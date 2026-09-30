"""DSRA voice detector + voice leveling.

DSRA features are the per-frame descriptors from the V9 notebook
(`dsra_features`): spectral centroid, flatness, 85% rolloff, low/speech/high
band energies, frame energy and its delta. Here they drive a voice-activity
decision, and the voice segments are then raised toward the level of the
loudest voice segment in the clip.

Voice frame := loud enough (within 35 dB of the clip's loudest frame)
             AND harmonic (spectral flatness low)
             AND speech-band dominant (300-3400 Hz carries most energy)
             AND centroid inside the speech range,
then smoothed with a hangover so word endings are not cut.
"""
import numpy as np

SR, NFFT, HOP = 16000, 512, 128


def dsra_features(x):
    win = np.hanning(NFFT + 1)[:-1].astype(np.float32)
    xp = np.pad(x, (NFFT // 2, NFFT // 2))
    nf = 1 + (len(xp) - NFFT) // HOP
    fr = np.lib.stride_tricks.sliding_window_view(xp, NFFT)[::HOP][:nf] * win
    mag = np.abs(np.fft.rfft(fr, axis=-1)).T + 1e-8                     # (F, T)
    freqs = np.fft.rfftfreq(NFFT, 1 / SR)
    den = mag.sum(0) + 1e-8
    centroid = (freqs[:, None] * mag).sum(0) / den
    flat = np.exp(np.mean(np.log(mag), 0)) / (np.mean(mag, 0) + 1e-8)
    cs = np.cumsum(mag, 0); roll = freqs[np.argmax(cs >= 0.85 * den[None, :], 0)]
    pw = mag ** 2
    low = pw[(freqs >= 20) & (freqs < 300)].sum(0)
    speech = pw[(freqs >= 300) & (freqs < 3400)].sum(0)
    high = pw[(freqs >= 3400)].sum(0)
    en = pw.sum(0)
    de = np.concatenate([[0], np.diff(np.log(en + 1e-12))])
    return dict(centroid=centroid, flat=flat, roll=roll, low=low, speech=speech,
                high=high, energy=en, delta=de)


def voice_activity(x, floor_db=35.0, flat_max=0.35, ratio_min=0.45,
                   hang_frames=8):
    f = dsra_features(x)
    e_db = 10 * np.log10(f["energy"] + 1e-12)
    loud = e_db > e_db.max() - floor_db
    harmonic = f["flat"] < flat_max
    ratio = f["speech"] / (f["energy"] + 1e-12)
    speechy = (ratio > ratio_min) & (f["centroid"] > 250) & (f["centroid"] < 3500)
    v = loud & harmonic & speechy
    # hangover: keep voice on for a few frames after it was last detected
    out = v.copy(); c = 0
    for t in range(len(v)):
        c = hang_frames if v[t] else max(0, c - 1)
        out[t] = c > 0
    return out, f


def level_voice(y, max_boost_db=12.0, seg_min_frames=6, attack_ms=10.0,
                release_ms=80.0):
    """Raise every voice segment toward the level of the loudest one.

    Non-voice frames get unity gain (they are not boosted, so residual noise
    between words is not lifted). Boost per segment is capped at
    `max_boost_db` and the gain curve is smoothed to avoid clicks.
    Returns (leveled signal, per-frame voice mask, per-frame gain in dB).
    """
    vad, f = voice_activity(y)
    T = len(vad)
    fl_e = f["energy"]
    segs, t = [], 0
    while t < T:
        if vad[t]:
            s = t
            while t < T and vad[t]:
                t += 1
            if t - s >= seg_min_frames:
                segs.append((s, t))
        else:
            t += 1
    gain_db = np.zeros(T)
    if segs:
        lv = np.array([10 * np.log10(fl_e[s:e].mean() + 1e-12) for s, e in segs])
        top = lv.max()
        for (s, e), l in zip(segs, lv):
            gain_db[s:e] = min(top - l, max_boost_db)
    # smooth gain (attack / release) at frame rate, then to samples
    a_up = 1 - np.exp(-HOP / (SR * attack_ms / 1000))
    a_dn = 1 - np.exp(-HOP / (SR * release_ms / 1000))
    sm = np.zeros(T); cur = 0.0
    for t in range(T):
        tgt = gain_db[t]
        cur += (a_up if tgt > cur else a_dn) * (tgt - cur)
        sm[t] = cur
    g = 10 ** (sm / 20)
    gs = np.interp(np.arange(len(y)), np.arange(T) * HOP, g)
    return (y * gs).astype(np.float32), vad, sm
