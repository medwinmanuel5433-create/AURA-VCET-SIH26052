"""MARK7 = MARK6.5 network (unchanged) wrapped in a real-time hybrid pipeline.

Stages (all causal, frame by frame, except the optional library stage):

  L  library reference cancellation (from v2) - OFFLINE ONLY, optional.
     If a known recording matches the input (confidence > ref_conf), it is
     subtracted by short-time least squares before the network.
  N  MARK6.5 network -> Y.
  G  change gate (Stage 0). Per frame, how much did the network remove?
        r = |X - Y|^2 / |X|^2.
     Little removed -> the frame is voice-only or silence -> output = INPUT,
     byte-exact (no coloration). Much removed -> noise present -> use Y.
     Instant attack, `hold_ms` release, so blasts are never let through and
     edges are crossfaded automatically.
  C  voice ceiling (ComTac/AMP-style limiter, but per frequency and tied to
     THIS speaker). From the frames the gate called clean voice, a running
     voice profile P(f) is kept. In noisy frames no bin may exceed
     P(f) * margin -> anything louder than the speaker's own voice in that
     band is noise and is squashed; voice below the ceiling is untouched.
  D  voice dictionary cleanup in noisy frames only.
     Voice atoms = universal (all training speakers) + LIVE atoms learned
     from this speaker's own clean frames seen so far (never the clean
     reference). Noise atoms = universal + LIVE atoms learned from what the
     network removed so far. Soft mask with a floor.
  S  output safety limiter (time domain, 2 ms lookahead): peaks may not
     exceed the speaker's clean-voice peak * limit_margin (hearing protection).
"""
import numpy as np
import torch
from scipy.signal import fftconvolve
from .losses import stft, istft
from .dict_refine import kl_nmf, EPS
from .dsra_level import voice_activity
from . import v2ref

MODES = {"clarity": {}, "max_quiet": {"dict": True, "floor": 0.5}}

DEFAULTS = dict(
    use_ref=True, ref_conf=0.5,
    gate=True, gate_lo_db=-20.0, gate_hi_db=-10.0, hold_ms=120.0, lvl_smooth=5,
    ceiling=False, ceil_margin_db=12.0, ceil_tau_s=2.0,
    dict=False, floor=0.5, power=1.5, live_k=16, live_block_s=1.0, nmf_iters=40, smooth=3,
    limiter=True, limit_margin_db=3.0,
)


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def _fast_match(x, library, max_refs=None):
    """Best normalized cross-correlation match over the library (FFT correlation)."""
    best = (0, 0, 0.0, None)
    x2c = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    for ref in library[:max_refs]:
        L = len(ref)
        if L >= len(x) or L < 64:
            continue
        c = fftconvolve(x, ref[::-1], mode="valid")
        seg = np.sqrt(np.maximum(x2c[L:] - x2c[:-L], 1e-12))[:len(c)]
        nc = np.abs(c) / (seg * np.sqrt(np.sum(ref ** 2)) + 1e-9)
        k = int(np.argmax(nc))
        if nc[k] > best[2]:
            best = (k, k + L, float(nc[k]), ref)
    return best


class Mark7:
    def __init__(self, model, cfg, W_v, W_n, library=None, **kw):
        self.m, self.cfg = model, cfg
        self.Wv0, self.Wn0 = W_v.float(), W_n.float()
        self.library = library or []
        self.p = dict(DEFAULTS); self.p.update(kw)

    # ------------------------------------------------------------------
    def __call__(self, x, return_info=False, ref=None):
        ref_mic = ref
        """ref: optional reference-mic signal (dual-mic MARK7-DM network)."""
        p, cfg = self.p, self.cfg
        x = np.asarray(x, np.float32); n = len(x); sr = cfg.sig.sr; hop = cfg.sig.hop
        info = {"ref_used": False, "ref_conf": 0.0}

        # L: library reference cancellation (offline only)
        x_in = x
        if p["use_ref"] and self.library:
            on, off, conf, lref = _fast_match(x, self.library)
            info["ref_conf"] = conf
            if conf > p["ref_conf"]:
                x_in = v2ref.reference_cancel(x.astype(np.float64), lref, on, off, sr).astype(np.float32)
                info["ref_used"] = True

        # N: network
        X = stft(torch.from_numpy(x_in).unsqueeze(0), cfg)[0]            # (T, F) complex
        with torch.no_grad():
            if ref_mic is None:
                Y = self.m(X.unsqueeze(0))["est"][0]
            else:
                Rs = stft(torch.from_numpy(np.asarray(ref_mic, np.float32)).unsqueeze(0), cfg)
                Y = self.m(X.unsqueeze(0), Rs)["est"][0]
        Xn, Yn = X.numpy(), Y.numpy()
        T = Xn.shape[0]
        fps = sr / hop

        # G: change gate, causal. The network also re-levels the voice (undoes
        # AGC pumping), so "change" is measured AFTER the best per-frame gain:
        # a frame the network only re-leveled counts as unchanged, and is
        # output as the INPUT re-leveled (exact input timbre, no coloration).
        num = np.real((np.conj(Xn) * Yn).sum(1)); eX = (np.abs(Xn) ** 2).sum(1) + 1e-12
        gl = np.clip(num / eX, 0.0, 4.0)
        if p["lvl_smooth"] > 1:                                    # causal smoothing of the gain
            a_ = 2.0 / (p["lvl_smooth"] + 1); s_ = gl[0]; gs = np.empty_like(gl)
            for t in range(T):
                s_ = (1 - a_) * s_ + a_ * gl[t]; gs[t] = s_
            gl = gs
        Xl = Xn * gl[:, None]
        if p["gate"]:
            eY = (np.abs(Yn) ** 2).sum(1) + 1e-12
            eR = (np.abs(Xl - Yn) ** 2).sum(1)
            r = _db(eR / np.maximum(eY, 1e-4 * eY.max()))
            wr = np.clip((r - p["gate_lo_db"]) / (p["gate_hi_db"] - p["gate_lo_db"]), 0, 1)
            hold = int(p["hold_ms"] / 1000 * fps); rel = 1.0 / max(1, hold)
            w = np.zeros(T); last = 0.0; cnt = 0
            for t in range(T):
                if wr[t] >= last:
                    last = wr[t]; cnt = hold
                elif cnt > 0:
                    cnt -= 1
                else:
                    last = max(wr[t], last - rel)
                w[t] = last
        else:
            w = np.ones(T)
        Z = Xl + w[:, None] * (Yn - Xl)
        info["gate_w"] = w

        # voice-only reference frames (causal knowledge of this speaker)
        vad, _ = voice_activity(x_in)
        vad = np.pad(vad, (0, max(0, T - len(vad))))[:T].astype(bool)
        clean_v = vad & (w < 0.05)
        info["clean_voice_frac"] = float(clean_v.mean())

        # C: per-frequency voice ceiling (running voice profile, causal)
        if p["ceiling"]:
            a = 1.0 / max(1.0, p["ceil_tau_s"] * fps)
            prof = None; marg = 10 ** (p["ceil_margin_db"] / 20)
            g = np.ones_like(np.abs(Z))
            magX = np.abs(Xl)
            for t in range(T):
                if prof is not None and w[t] > 0:
                    c = np.sqrt(prof) * marg
                    mz = np.abs(Z[t]) + 1e-12
                    g[t] = 1 - w[t] + w[t] * np.minimum(1.0, c / mz)
                if clean_v[t]:
                    pw = magX[t] ** 2
                    prof = pw.copy() if prof is None else np.maximum((1 - a) * prof + a * pw, 0)
                    # peak-tracking: follow loud voice quickly, decay slowly
                    prof = np.maximum(prof, 0.5 * pw)
            Z = Z * g

        # D: dictionary cleanup in noisy frames only, live atoms (causal blocks)
        if p["dict"] and (w > 0.05).any():
            mag = torch.from_numpy(np.abs(Z).T.astype(np.float32))                    # (F, T)
            magX = torch.from_numpy(np.abs(Xl).T.astype(np.float32))
            magR = torch.from_numpy(np.abs(Xn - Yn).T.astype(np.float32))
            noisy = w > 0.05
            B = max(1, int(p["live_block_s"] * fps))
            M = np.ones((mag.shape[0], T), np.float32)
            k = p["live_k"]
            for s in range(0, T, B):
                e = min(T, s + B)
                idx = np.where(noisy[s:e])[0] + s
                if len(idx) == 0:
                    continue
                Wv, Wn = [self.Wv0], [self.Wn0]
                cv = np.where(clean_v[:s])[0]
                if len(cv) >= max(12, k):
                    Wv.append(kl_nmf(magX[:, cv].contiguous(), k, iters=60, seed=0))
                rn = np.where(noisy[:s])[0]
                if len(rn) >= max(12, k):
                    Wn.append(kl_nmf(magR[:, rn].contiguous(), k, iters=60, seed=1))
                Wv = torch.cat(Wv, 1); Wn = torch.cat(Wn, 1)
                W = torch.cat([Wv, Wn], 1); kv = Wv.shape[1]
                V = mag[:, idx]
                H = torch.full((W.shape[1], V.shape[1]), 0.1)
                den = W.T @ torch.ones_like(V) + EPS
                for _ in range(p["nmf_iters"]):
                    H *= (W.T @ (V / (W @ H + EPS))) / den
                Ve = (Wv @ H[:kv]) ** p["power"]; Ne = (Wn @ H[kv:]) ** p["power"]
                M[:, idx] = (Ve / (Ve + Ne + EPS)).numpy()
            M = np.maximum(M, p["floor"])
            if p["smooth"] > 1:                                   # causal moving average
                Mc = np.cumsum(np.pad(M, ((0, 0), (p["smooth"] - 1, 0)), mode="edge"), 1)
                Mc = np.concatenate([np.zeros((M.shape[0], 1)), Mc], 1)
                M = (Mc[:, p["smooth"]:] - Mc[:, :-p["smooth"]]) / p["smooth"]
            G = 1 - w[None, :] + w[None, :] * M
            Z = Z * G.T

        y = istft(torch.from_numpy(Z).unsqueeze(0), cfg, length=n)[0].numpy()
        # S: output safety limiter
        if p["limiter"]:
            y = self._limit(y, x, clean_v, hop, sr)
        info["gate_frac"] = float((w > 0.05).mean())
        y = y.astype(np.float32)
        return (y, info) if return_info else y

    def _limit(self, y, x, clean_v, hop, sr):
        n = len(y)
        # running clean-voice peak (causal)
        cv = np.repeat(clean_v, hop)[:n]; cv = np.pad(cv, (0, n - len(cv)))
        pk = np.maximum.accumulate(np.where(cv, np.abs(x), 0.0))
        if pk[-1] <= 0:
            return y
        ceil = np.where(pk > 0, pk, np.inf) * 10 ** (self.p["limit_margin_db"] / 20)
        la = int(0.002 * sr); rel = np.exp(-1 / (0.05 * sr))
        need = np.minimum(1.0, ceil / (np.abs(y) + 1e-9))
        # lookahead: gain at t is the min over the next `la` samples
        from scipy.ndimage import minimum_filter1d
        need = minimum_filter1d(need, size=2 * la + 1, origin=0)
        g = np.empty(n); cur = 1.0
        for i in range(n):
            cur = need[i] if need[i] < cur else min(need[i], 1 - (1 - cur) * rel)
            g[i] = cur
        return y * g
