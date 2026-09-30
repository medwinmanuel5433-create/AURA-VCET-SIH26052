"""Output loudness stage: make the cleaned voice loud and steady.

  1. (optional, off by default: it cost ~1.4 dB level-aligned SNR in testing)
     voice-gated slow AGC  - measures level ONLY on frames the DSRA detector
     calls voice, moves slowly (tau ~0.8 s), holds its gain in pauses and
     blasts, so leftover noise is never pumped up; max +12 dB / -6 dB.
  2. loudness normalisation to a target in LUFS (ITU-R BS.1770 K-weighting).
     Offline: integrated loudness of the clip. On the device the AGC target
     already sits at this level, so this step becomes a fixed gain.
  3. peak limiter, 2 ms lookahead, ceiling -1 dBFS, 60 ms release.
Level-only processing: PESQ and STOI are level-invariant, so they do not
change; raw SNR against a fixed-level reference does (use level-aligned SNR).
"""
import numpy as np
from scipy.ndimage import minimum_filter1d
from .dsra_level import voice_activity

SR, HOP = 16000, 128


def voice_agc(y, target_db=-20.0, tau_s=0.8, max_up_db=12.0, max_down_db=6.0):
    vad, f = voice_activity(y)
    e_db = 10 * np.log10(f["energy"] / (256 ** 2) + 1e-12)      # frame power, ~dBFS
    a = 1 - np.exp(-HOP / (tau_s * SR))
    g_db = np.zeros(len(vad)); lvl = None; cur = 0.0
    for t in range(len(vad)):
        if vad[t]:
            lvl = e_db[t] if lvl is None else (1 - a) * lvl + a * e_db[t]
            cur = float(np.clip(target_db - lvl, -max_down_db, max_up_db))
        g_db[t] = cur
    if lvl is None:
        return y
    # first voice frames: use the first measured gain instead of 0
    first = np.argmax(vad); g_db[:first] = g_db[first]
    g = 10 ** (np.repeat(g_db, HOP)[: len(y)] / 20)
    g = np.pad(g, (0, len(y) - len(g)), mode="edge")
    return (y * g).astype(np.float32)


def lufs_normalize(y, target_lufs=-18.0):
    import pyloudnorm as pyln
    meter = pyln.Meter(SR, block_size=0.4)
    try:
        L = meter.integrated_loudness(y.astype(np.float64))
    except Exception:
        return y
    if not np.isfinite(L):
        return y
    return (y * 10 ** ((target_lufs - L) / 20)).astype(np.float32)


def peak_limit(y, ceiling_db=-1.0, look_ms=2.0, release_ms=60.0):
    c = 10 ** (ceiling_db / 20); la = int(look_ms / 1000 * SR); rel = np.exp(-1 / (release_ms / 1000 * SR))
    need = np.minimum(1.0, c / (np.abs(y) + 1e-9))
    need = minimum_filter1d(need, size=2 * la + 1)
    g = np.empty_like(need); cur = 1.0
    for i in range(len(y)):
        cur = need[i] if need[i] < cur else min(need[i], 1 - (1 - cur) * rel)
        g[i] = cur
    return (y * g).astype(np.float32)


def make_loud(y, target_lufs=-18.0, agc=False):
    y = np.asarray(y, np.float32)
    if agc:
        y = voice_agc(y, target_db=target_lufs - 2.0)
    return peak_limit(lufs_normalize(y, target_lufs))
