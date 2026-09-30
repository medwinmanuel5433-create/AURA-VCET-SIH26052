"""Per-signal adaptive dictionaries, re-learned at every loop level.

For each loop k:
    y_in  -> MARK6.5 -> y_net
    removed_k = y_in - y_net                 (the noise this pass found)
    frames are classified from the signal itself:
        voice-only : DSRA says voice AND this pass removed little there
        noise      : this pass removed a lot there, or DSRA says no voice
    W_v <- NMF on |Y_net| in voice-only frames   (THIS speaker's voice)
    W_n <- NMF on |removed_k| everywhere it is non-trivial
            + |Y_net| in non-voice frames        (the noise at THIS level,
                                                   including what is left)
    y_out = semi-supervised NMF mask with [W_v, W_n] applied to Y_net

`causal=True` restricts every dictionary to frames strictly before the block
being processed (1 s blocks), which is what a headset could actually do.
"""
import numpy as np
import torch
from .losses import stft, istft
from .dict_refine import kl_nmf, EPS
from .dsra_level import voice_activity, level_voice


def _decompose(mag, Wv, Wn, iters=60):
    W = torch.cat([Wv, Wn], 1); kv = Wv.shape[1]
    H = torch.full((W.shape[1], mag.shape[1]), 0.1)
    den = W.T @ torch.ones_like(mag) + EPS
    for _ in range(iters):
        H *= (W.T @ (mag / (W @ H + EPS))) / den
    return Wv @ H[:kv], Wn @ H[kv:]


def _learn(frames, k, fallback, iters=80):
    if frames.shape[1] < max(8, k):
        return fallback
    return kl_nmf(frames.contiguous(), k, iters=iters, seed=0)


def adaptive_pass(y_in, model, cfg, Wv0, Wn0, kv=24, kn=24, floor=0.2, power=2.0,
                  causal=False, block_s=1.0):
    n = len(y_in)
    X = stft(torch.from_numpy(y_in).unsqueeze(0), cfg)[0]
    with torch.no_grad():
        Y = model(X.unsqueeze(0))["est"][0]
    R = X - Y                                              # removed this pass
    y_net = istft(Y.unsqueeze(0), cfg, length=n)[0].numpy()
    vad, _ = voice_activity(y_net)
    T = Y.shape[0]; vad = np.pad(vad, (0, max(0, T - len(vad))))[:T]
    ey = (Y.abs() ** 2).sum(1).numpy(); er = (R.abs() ** 2).sum(1).numpy()
    voice_only = vad & (er < 0.1 * ey)
    noise_fr = (er > 0.5 * ey) | (~vad)
    magY, magR = Y.abs().T, R.abs().T                      # (F, T)
    rkeep = er > er.max() * 1e-4

    def dicts(upto):
        vsel = np.where(voice_only[:upto])[0]
        nsel_r = np.where(rkeep[:upto])[0]
        nsel_y = np.where(noise_fr[:upto])[0]
        Wv = _learn(magY[:, vsel], kv, Wv0)
        Nf = torch.cat([magR[:, nsel_r], magY[:, nsel_y]], 1)
        Wn = _learn(Nf, kn, Wn0)
        # keep the generic atoms too, so rare sounds are still covered
        return torch.cat([Wv, Wv0], 1), torch.cat([Wn, Wn0], 1)

    M = torch.ones_like(magY)
    if causal:
        B = max(1, int(block_s * cfg.sig.sr / cfg.sig.hop))
        for s in range(0, T, B):
            e = min(T, s + B)
            Wv, Wn = dicts(s) if s > 0 else (Wv0, Wn0)
            V, N = _decompose(magY[:, s:e], Wv, Wn)
            M[:, s:e] = V ** power / (V ** power + N ** power + EPS)
    else:
        Wv, Wn = dicts(T)
        V, N = _decompose(magY, Wv, Wn)
        M = V ** power / (V ** power + N ** power + EPS)
    M = torch.clamp(M, min=floor)
    Yo = Y * M.T
    est_noise_db = 10 * np.log10(float(((magY * (1 - M)) ** 2).sum() / ((magY * M) ** 2).sum() + EPS))
    return istft(Yo.unsqueeze(0), cfg, length=n)[0].numpy(), est_noise_db


def adaptive_loop(mix, model, cfg, Wv0, Wn0, max_iter=4, min_gain_db=0.5,
                  dsra_boost_db=None, trace_fn=None, **kw):
    """Loop: each level re-learns voice and noise patterns from the current signal."""
    y = mix.astype(np.float32); trace = []; prev = None
    for k in range(max_iter):
        y, est = adaptive_pass(y, model, cfg, Wv0, Wn0, **kw)
        trace.append(est if trace_fn is None else (est, trace_fn(y)))
        if prev is not None and prev - est < min_gain_db:
            break
        prev = est
    if dsra_boost_db:
        y = level_voice(y, max_boost_db=dsra_boost_db)[0]
    return y.astype(np.float32), trace
