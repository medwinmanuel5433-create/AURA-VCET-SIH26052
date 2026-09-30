"""Recoverability-aware training objective.

The central idea
----------------
At the instant a blast arrives, the speech that coincided with it is not
attenuated in the recording -- it is *absent*. The capsule saturated, the ADC
clipped, and whatever remained sits tens of dB below the residual floor. No
mask, filter or network can recover information the recording does not
contain.

Earlier pipelines applied a uniform reconstruction loss over every T-F bin,
which meant a large share of the gradient was spent demanding the impossible.
The network's only way to reduce that loss was to hallucinate broadband energy
or, more usually, to over-suppress -- which is precisely the suppression-versus-
muffling trade-off those runs reported, and the reason they plateaued.

Here each T-F bin is scored by how much speech information actually survived:

    snr(t,f) = |Target(t,f)|^2 / |Mix(t,f) - Target(t,f)|^2
    w(t,f)   = sigmoid((snr_db + center) / scale)

Bins with w near 1 are *recoverable* and carry the reconstruction loss. Bins
with w near 0 are *destroyed*; demanding exact reconstruction there is noise,
so they are handed to a separate magnitude-inpainting objective which is judged
on plausibility rather than on sample-exact recovery.
"""
import torch
import torch.nn.functional as F

from .config import Cfg, DEFAULT


def stft(x, cfg: Cfg):
    win = torch.hann_window(cfg.sig.win, device=x.device)
    X = torch.stft(x, n_fft=cfg.sig.n_fft, hop_length=cfg.sig.hop,
                   win_length=cfg.sig.win, window=win, center=True,
                   return_complex=True)
    return X.transpose(1, 2)          # (B, T, F)


def istft(X, cfg: Cfg, length=None):
    win = torch.hann_window(cfg.sig.win, device=X.device)
    return torch.istft(X.transpose(1, 2), n_fft=cfg.sig.n_fft,
                       hop_length=cfg.sig.hop, win_length=cfg.sig.win,
                       window=win, center=True, length=length)


def recoverability(T_spec, R_spec, cfg: Cfg):
    """Per-bin weight in [0,1]: how much speech information survived.

    T_spec: target spectrogram, R_spec: residual (mix - target) spectrogram.
    """
    tp = T_spec.abs() ** 2
    rp = R_spec.abs() ** 2
    snr_db = 10.0 * torch.log10((tp + 1e-10) / (rp + 1e-10))
    c = cfg.train.rec_center_db
    s = cfg.train.rec_scale_db
    return torch.sigmoid((snr_db + c) / s)


def si_sdr(est, ref, eps=1e-8):
    """Scale-invariant SDR. Reported only -- NOT used as a training loss.

    SI-SDR is invariant to both scale and polarity, and is unbounded above.
    Used as an objective here it dominates the sign-sensitive terms and admits
    a degenerate optimum at est = -ref, which scores perfectly on SI-SDR,
    STOI, PESQ and multi-resolution STFT (all polarity-blind) while being
    -6 dB on true SNR. That failure was observed in run 1 of this model
    (correlation -0.99 against the target). Train with `snr_loss` instead.
    """
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj.pow(2).sum(-1) + eps) / (noise.pow(2).sum(-1) + eps))


def snr_loss(est, ref, eps=1e-8, max_db: float = 35.0):
    """Scale- and polarity-dependent SNR, clamped.

    The target here is an exact achievable waveform, not an arbitrarily scaled
    source, so scale invariance is not merely unnecessary -- it is wrong. The
    clamp stops a well-reconstructed clip from producing an unbounded reward
    that swamps every other term.
    """
    noise = est - ref
    ref_e = ref.pow(2).sum(-1)
    snr = 10 * torch.log10((ref_e + eps) / (noise.pow(2).sum(-1) + eps))
    loss = -torch.clamp(snr, max=max_db)
    # A crop whose target is silent (e.g. LibriSpeech zero-padding, now that
    # the target carries no mic noise) has no defined SNR: the ratio goes to
    # -inf and the loss to +80 dB or more. Such items contribute zero here;
    # the magnitude and clean-frame terms still push their output to silence.
    valid = (ref_e > 1e-4 * ref.shape[-1] * 1e-4).float()
    return loss * valid


def multi_res_stft(est, ref, sizes=((256, 64), (512, 128), (1024, 256))):
    loss = 0.0
    for n_fft, hop in sizes:
        win = torch.hann_window(n_fft, device=est.device)
        E = torch.stft(est, n_fft, hop, n_fft, win, return_complex=True).abs()
        R = torch.stft(ref, n_fft, hop, n_fft, win, return_complex=True).abs()
        rn = torch.norm(R, p="fro", dim=(-2, -1))
        # Spectral convergence is a ratio to the target's own norm; for a
        # silent target it divides by ~0 (measured: 4e9 on one crop). Floor
        # the denominator and drop silent items from this term.
        valid = (rn > 1e-2).float()
        sc = torch.norm(R - E, p="fro", dim=(-2, -1)) / (rn + 1.0) * valid
        mag = F.l1_loss(torch.log(E + 1e-5), torch.log(R + 1e-5))
        loss = loss + sc.mean() + mag
    return loss / len(sizes)


def mark6_loss(out, batch, cfg: Cfg = DEFAULT):
    """Full objective. Returns (total, dict_of_terms)."""
    tc = cfg.train
    mix_w = batch["mix"]                     # (B, N)
    tgt_w = batch["target"]                  # (B, N)
    n = mix_w.shape[-1]

    X = batch["X"]                           # (B,T,F) complex, mixture
    T_spec = batch["T"]                      # (B,T,F) complex, target
    R_spec = X - T_spec                      # residual actually present

    w = recoverability(T_spec, R_spec, cfg)  # (B,T,F) in [0,1]
    Y = out["est"]

    # ---- reconstruction on recoverable bins ---------------------------
    ri = ((Y.real - T_spec.real).abs() + (Y.imag - T_spec.imag).abs()) * w
    l_ri = ri.sum() / (w.sum() * 2 + 1e-6)

    ymag = torch.sqrt(Y.real ** 2 + Y.imag ** 2 + 1e-9)
    tmag = torch.sqrt(T_spec.real ** 2 + T_spec.imag ** 2 + 1e-9)
    # Magnitude is constrained on EVERY bin, with a floor under the weight.
    # Phase is unrecoverable inside the blast, but magnitude still is -- and
    # leaving destroyed bins almost unweighted (as a pure `w` weighting does)
    # lets output energy run away exactly where event-window SNR is measured.
    w_mag = 0.15 + 0.85 * w
    lm = (torch.log(ymag + 1e-5) - torch.log(tmag + 1e-5)).abs() * w_mag
    l_mag = lm.sum() / (w_mag.sum() + 1e-6)

    # ---- inpainting on destroyed bins ---------------------------------
    wn = 1.0 - w
    inp = (out["inpaint_logmag"] - torch.log(tmag + 1e-5)).abs() * wn
    l_inp = inp.sum() / (wn.sum() + 1e-6)

    # ---- recoverability prediction ------------------------------------
    # The blend weight must be inferable at runtime; supervise it against the
    # value the corpus generator knows.
    l_rec = F.binary_cross_entropy_with_logits(out["rec_logit"], w.detach())

    # ---- waveform-domain terms ----------------------------------------
    y_wav = istft(Y, cfg, length=n)
    l_wav = F.l1_loss(y_wav, tgt_w)
    _snr = snr_loss(y_wav, tgt_w)                 # sign-aware, clamped
    _nv = (tgt_w.pow(2).sum(-1) > 1e-4 * tgt_w.shape[-1] * 1e-4).float().sum()
    l_sisdr = _snr.sum() / _nv.clamp(min=1.0)
    l_mr = multi_res_stft(y_wav, tgt_w)

    # ---- detector ------------------------------------------------------
    # frame labels derived from the signals themselves, not from filenames
    with torch.no_grad():
        tframe = tmag.pow(2).sum(-1)
        rframe = (R_spec.abs() ** 2).sum(-1)
        speech_lab = (10 * torch.log10(tframe + 1e-10) >
                      10 * torch.log10(tframe.amax(dim=1, keepdim=True) + 1e-10) - 35).float()
        # An "event" frame must be dominated by the residual AND the residual
        # must be loud in absolute terms. Without the second condition, speech
        # pauses -- where the target is near-silent and the mic hiss is all
        # that is left -- are labelled as events.
        r_db = 10 * torch.log10(rframe + 1e-10)
        t_db = 10 * torch.log10(tframe + 1e-10)
        peak_db = 10 * torch.log10(tframe.amax(dim=1, keepdim=True) + 1e-10)
        event_lab = ((r_db > t_db - 6) & (r_db > peak_db - 25)).float()
    det = out["det_logits"]
    l_det = F.binary_cross_entropy_with_logits(
        det, torch.stack([speech_lab, event_lab], dim=-1))

    # ---- clean-frame preservation --------------------------------------
    # On frames with no event, match the clean TARGET with extra weight. This
    # used to pull toward the input X, which was right while the target
    # contained the same hiss and reverb as the input; with a dry, hiss-free
    # target, pulling toward X would directly fight hiss and reverb removal.
    # The guarantee it provides is unchanged: frames with no event must not
    # be damaged.
    quiet = (1.0 - event_lab).unsqueeze(-1)            # (B,T,1)
    pt = ((Y.real - T_spec.real).abs() + (Y.imag - T_spec.imag).abs()) * quiet
    l_pass = pt.sum() / (quiet.sum() * 2 * X.shape[-1] + 1e-6)

    total = (tc.w_ri * l_ri + tc.w_mag * l_mag + tc.w_wav * l_wav
             + tc.w_sisdr * l_sisdr + tc.w_mrstft * l_mr + tc.w_det * l_det
             + tc.w_pass * l_pass + tc.w_inpaint * l_inp + tc.w_rec * l_rec)

    return total, {
        "ri": l_ri.item(), "mag": l_mag.item(), "wav": l_wav.item(),
        "sisdr": l_sisdr.item(), "mrstft": l_mr.item(), "det": l_det.item(),
        "pass": l_pass.item(), "inpaint": l_inp.item(), "rec": l_rec.item(),
        "rec_frac": w.mean().item(),
        "rec_hat": out["rec_hat"].mean().item(),
        "total": total.item(),
    }
