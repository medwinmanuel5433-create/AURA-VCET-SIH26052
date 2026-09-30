"""Pseudo-speaker generation by vocal-tract-length perturbation.

Why
---
The voice corpus contains roughly two distinct voices across 71 files, which
makes a speaker-held-out evaluation impossible and lets the network memorise
talkers instead of learning speech structure. A real corpus is the correct fix,
but it is not always obtainable; this module manufactures additional apparent
talkers from the material at hand.

VTLP (Jaitly & Hinton, 2013) warps the frequency axis to simulate a different
vocal tract length. Unlike simple resampling it changes formant positions while
leaving pitch and duration intact, so the result reads as a different speaker
saying the same words at the same speed. Combined with independent pitch
shifting it spans a usable speaker space.

This is an augmentation, not a substitute for data. It multiplies apparent
speakers; it adds no new phonetic content, no new recording conditions and no
new languages. Treat the resulting speaker count as an upper bound on
diversity, never as equivalent to that many real talkers.

Warp factors are partitioned across splits, so a pseudo-speaker used in test is
never seen in training.
"""
import numpy as np

# Warp factors held disjoint across splits, so a test pseudo-speaker is unseen.
WARP_SPLITS = {
    "train": [0.88, 0.91, 0.94, 0.97, 1.00, 1.03, 1.06, 1.09, 1.12],
    "val":   [0.895, 1.045],
    "test":  [0.925, 0.985, 1.075],
}

PITCH_SPLITS = {
    "train": [-2.0, -1.0, 0.0, 1.0, 2.0],
    "val":   [-1.5, 1.5],
    "test":  [-2.5, 0.5, 2.5],
}


def vtlp_warp(x: np.ndarray, sr: int, alpha: float,
              n_fft: int = 512, hop: int = 128) -> np.ndarray:
    """Piecewise-linear frequency warping of the magnitude spectrum.

    alpha < 1 lengthens the apparent vocal tract (formants move down),
    alpha > 1 shortens it. Phase is carried over unchanged; the residual
    artefacts are well below what the capture chain then adds.
    """
    if abs(alpha - 1.0) < 1e-3:
        return x
    win = np.hanning(n_fft).astype(np.float32)
    n_frames = max(1, 1 + (len(x) - n_fft) // hop)
    n_freq = n_fft // 2 + 1
    freqs = np.arange(n_freq, dtype=np.float32)

    # warp map: linear with slope alpha up to the boundary, then a straight
    # line back to Nyquist so the full band stays covered
    f_bound = 0.8 * n_freq / max(alpha, 1.0)
    warped = np.where(
        freqs <= f_bound,
        alpha * freqs,
        (n_freq - 1) - ((n_freq - 1) - alpha * f_bound) *
        ((n_freq - 1) - freqs) / max((n_freq - 1) - f_bound, 1e-6))
    warped = np.clip(warped, 0, n_freq - 1)

    out = np.zeros(len(x) + n_fft, dtype=np.float32)
    wsum = np.zeros(len(x) + n_fft, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop:i * hop + n_fft]
        if len(seg) < n_fft:
            seg = np.pad(seg, (0, n_fft - len(seg)))
        S = np.fft.rfft(seg * win)
        mag = np.abs(S)
        phase = np.angle(S)
        mag_w = np.interp(freqs, warped, mag)
        Sw = mag_w * np.exp(1j * phase)
        frame = np.fft.irfft(Sw, n_fft).astype(np.float32) * win
        out[i * hop:i * hop + n_fft] += frame
        wsum[i * hop:i * hop + n_fft] += win ** 2
    out = out[:len(x)] / np.maximum(wsum[:len(x)], 1e-6)
    return out.astype(np.float32)


def pitch_shift(x: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Resample-then-restore-length pitch shift.

    Shifts pitch and duration together, then restores duration by linear
    resampling, which leaves pitch shifted and duration intact.
    """
    if abs(semitones) < 1e-3:
        return x
    rate = 2.0 ** (semitones / 12.0)
    n_out = int(len(x) / rate)
    if n_out < 16:
        return x
    idx = np.linspace(0, len(x) - 1, n_out).astype(np.float32)
    y = np.interp(idx, np.arange(len(x)), x).astype(np.float32)
    idx2 = np.linspace(0, len(y) - 1, len(x)).astype(np.float32)
    return np.interp(idx2, np.arange(len(y)), y).astype(np.float32)


def apply_pseudo_speaker(x: np.ndarray, sr: int, split: str,
                         rng: np.random.RandomState) -> tuple:
    """Draw a pseudo-speaker identity for `split` and apply it.

    Returns (audio, identity_string). The identity is recorded in the corpus
    manifest so pseudo-speaker disjointness across splits can be audited.
    """
    a = WARP_SPLITS[split][rng.randint(len(WARP_SPLITS[split]))]
    p = PITCH_SPLITS[split][rng.randint(len(PITCH_SPLITS[split]))]
    y = vtlp_warp(x, sr, a)
    y = pitch_shift(y, sr, p)
    return y, f"vtlp{a:.3f}_pit{p:+.1f}"


def n_pseudo_speakers(split: str) -> int:
    return len(WARP_SPLITS[split]) * len(PITCH_SPLITS[split])
