"""Physics-correct mixture generation.

Why this exists
---------------
Earlier pipelines built training data as `mix = voice + noise`, then rescaled
the mixture to avoid clipping while leaving the target at full level. That
creates physically impossible examples: it implies a capsule that recorded a
blast and speech simultaneously with unlimited headroom and perfect linearity,
and it makes the target unreachable (the network is asked to produce energy
that the mixture never contained at that level).

Here the mixture is produced by pushing the *sum* through a real capture chain

    source -> distance/air -> room -> SUM -> capsule saturation -> AGC -> ADC clip

and the target is defined as *the same chain applied to the voice alone, using
the gain trajectory that the mixture produced*. That target is by construction
achievable: it is exactly what the microphone would have delivered had the
event not fired, at the level the AGC actually settled on.

The generator also records, per example, the information needed to compute a
per-T-F *recoverability* weight. Where the chain hard-clipped, or where the
event buries speech far below the residual floor, the speech information is
not present in the recording in any form. Training the network to reconstruct
it anyway is what produces the suppression-vs-muffling trade-off seen in
earlier runs. Those bins are routed to a separate objective instead.
"""
import os
import json
import glob
import numpy as np
from typing import Dict, List, Optional

from .config import Cfg, DEFAULT
from . import audio as A


# --------------------------------------------------------------------------
# chain components
# --------------------------------------------------------------------------

def one_pole_lowpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """First-order low-pass, used as a cheap air-absorption model."""
    if cutoff_hz >= sr / 2:
        return x
    a = np.exp(-2.0 * np.pi * cutoff_hz / sr)
    y = np.empty_like(x)
    acc = 0.0
    for i in range(len(x)):
        acc = (1 - a) * x[i] + a * acc
        y[i] = acc
    return y


def _fast_lowpass(x: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    """Vectorised equivalent of the one-pole filter via scipy when available."""
    if cutoff_hz >= sr / 2:
        return x
    try:
        from scipy.signal import lfilter
        a = float(np.exp(-2.0 * np.pi * cutoff_hz / sr))
        return lfilter([1 - a], [1.0, -a], x).astype(np.float32)
    except Exception:
        return one_pole_lowpass(x, cutoff_hz, sr)


RT60 = {"small": 0.25, "medium": 0.55, "large": 1.1}
EARLY_MS = 50.0   # direct path + reflections inside this window count as "the voice"


def synth_rir_pair(room: str, sr: int, rng: np.random.RandomState,
                   drr_db: float):
    """Calibrated room impulse response.

    Returns (full, early):
      full  = direct path + early reflections + late diffuse tail
      early = direct path + early reflections only (first EARLY_MS)

    The mixture is rendered with `full`; the voice TARGET with `early`. That
    is the convention used for dereverberation targets (e.g. the DNS
    challenge): early reflections are perceived as part of the voice, the late
    tail is what makes speech sound like it is in an empty hall.

    `drr_db` fixes the direct-to-reverberant ratio. The previous generator had
    no such control: its "direct path dominates" comment was wrong -- a
    unit-variance noise tail over thousands of samples carries hundreds of
    times the energy of the unit direct impulse, and the measured DRR of the
    old targets was -9.9 / -12.3 / -14.9 dB (small / medium / large). A headset
    microphone a few cm from the mouth is in the +10..+25 dB region.
    """
    if room == "anechoic":
        one = np.array([1.0], dtype=np.float32)
        return one, one
    rt60 = RT60[room]
    n = int(sr * min(rt60 * 1.2, 1.5))
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / rt60)
    ne = int(sr * EARLY_MS / 1000)

    # sparse early reflections, 1..EARLY_MS
    early = np.zeros(n, np.float32)
    for _ in range(rng.randint(4, 10)):
        k = rng.randint(int(sr * 0.001), ne)
        early[k] += rng.randn() * decay[k]
    # diffuse late tail from EARLY_MS on, with a short fade-in
    late = (rng.randn(n) * decay).astype(np.float32)
    fade = np.clip((t - EARLY_MS / 1000 * 0.6) / (EARLY_MS / 1000 * 0.4), 0, 1)
    late *= fade.astype(np.float32)
    late[:int(ne * 0.6)] = 0.0

    # calibrate: reverberant energy = 10^(-drr/10) x direct energy (=1),
    # split between early and late
    e_frac = rng.uniform(0.3, 0.6)
    rev = 10 ** (-drr_db / 10.0)
    early *= np.sqrt(e_frac * rev / (np.sum(early ** 2) + 1e-12))
    late *= np.sqrt((1 - e_frac) * rev / (np.sum(late ** 2) + 1e-12))

    full = late + early
    full[0] += 1.0
    early_rir = early[:ne].copy()
    early_rir[0] += 1.0
    return full.astype(np.float32), early_rir.astype(np.float32)


def voice_drr(rng) -> float:
    """Close-talk / headset microphone: a boom mic a few cm from the mouth."""
    return rng.uniform(12.0, 25.0)


def dc_block(x: np.ndarray, sr: int, fc: float = 40.0) -> np.ndarray:
    """2nd-order high-pass, as every real microphone/ADC chain has.

    Source files carry small DC offsets (LibriSpeech up to ~0.01). The AGC
    multiplies them by up to 16x, and the full and early room responses pass
    different amounts of DC, so an unfiltered offset turns into a slow
    low-frequency wander that is in the mix but not the target (measured:
    one validation clip with target DC -0.146).
    """
    try:
        from scipy.signal import butter, lfilter
        b, a = butter(2, fc / (sr / 2), btype="high")
        return lfilter(b, a, x).astype(np.float32)
    except Exception:
        return (x - np.mean(x)).astype(np.float32)


def event_drr(distance: str, rng) -> float:
    """Farther sources are more reverberant."""
    lo, hi = {"near": (4.0, 12.0), "mid": (-2.0, 6.0), "far": (-8.0, 2.0)}[distance]
    return rng.uniform(lo, hi)


def synth_rir(room: str, sr: int, rng: np.random.RandomState) -> np.ndarray:
    """Legacy generator (MARK6.0-6.2 corpora). Kept only so old corpora can be
    reproduced; do not use for new data -- see synth_rir_pair."""
    if room == "anechoic":
        return np.array([1.0], dtype=np.float32)
    rt60 = {"small": 0.25, "medium": 0.55, "large": 1.1}[room]
    n = int(sr * min(rt60 * 1.2, 1.5))
    t = np.arange(n) / sr
    decay = np.exp(-6.908 * t / rt60)          # -60 dB at rt60
    tail = rng.randn(n).astype(np.float32) * decay
    # direct path dominates
    pre = int(sr * 0.002)
    tail[:pre] *= 0.2
    rir = tail
    rir[0] += 1.0
    rir /= (np.sqrt(np.sum(rir ** 2)) + 1e-9)
    return rir.astype(np.float32)


def convolve_rir(x: np.ndarray, rir: np.ndarray) -> np.ndarray:
    if len(rir) == 1:
        return x * rir[0]
    try:
        from scipy.signal import fftconvolve
        y = fftconvolve(x, rir)[:len(x)]
    except Exception:
        y = np.convolve(x, rir)[:len(x)]
    return y.astype(np.float32)


def distance_filter(x: np.ndarray, distance: str, sr: int,
                    cfg: Cfg, rng: np.random.RandomState) -> np.ndarray:
    """Distance attenuation + frequency-dependent air absorption."""
    rng_m = {"near": (0.5, 3.0), "mid": (3.0, 20.0), "far": (20.0, 120.0)}[distance]
    r = rng.uniform(*rng_m)
    atten = 1.0 / max(r, 0.5)
    frac = np.clip((r - 0.5) / 120.0, 0.0, 1.0)
    lp = cfg.chain.air_lp_hz_near + frac * (cfg.chain.air_lp_hz_far - cfg.chain.air_lp_hz_near)
    y = _fast_lowpass(x, lp, sr) * atten
    return y.astype(np.float32), r


def capsule_saturate(x: np.ndarray, headroom: float) -> np.ndarray:
    """Soft saturation of the microphone capsule / preamp front end."""
    return (headroom * np.tanh(x / max(headroom, 1e-6))).astype(np.float32)


def agc_gain(x: np.ndarray, cfg: Cfg, sr: int) -> np.ndarray:
    """Compute the AGC gain trajectory g(t) from the signal envelope.

    Returned separately from its application so the *same* trajectory can be
    applied to the voice-only path, which is what makes the target achievable.
    """
    att = np.exp(-1.0 / (sr * cfg.chain.agc_attack_ms / 1000.0))
    rel = np.exp(-1.0 / (sr * cfg.chain.agc_release_ms / 1000.0))
    env = np.empty_like(x)
    acc = 0.0
    ax = np.abs(x)
    for i in range(len(x)):
        a = att if ax[i] > acc else rel
        acc = (1 - a) * ax[i] + a * acc
        env[i] = acc
    g = cfg.chain.agc_target_rms / (env + 1e-6)
    gmax = 10 ** (cfg.chain.agc_max_gain_db / 20)
    gmin = 10 ** (cfg.chain.agc_min_gain_db / 20)
    return np.clip(g, gmin, gmax).astype(np.float32)


def _agc_gain_fast(x: np.ndarray, cfg: Cfg, sr: int) -> np.ndarray:
    """Vectorised envelope follower (scipy lfilter, single release constant).

    Uses an asymmetric follower approximated by a fast attack stage followed by
    a slow release stage; matches the sample loop closely enough for training
    data and is ~100x faster.
    """
    try:
        from scipy.signal import lfilter
        ax = np.abs(x).astype(np.float32)
        a_att = float(np.exp(-1.0 / (sr * cfg.chain.agc_attack_ms / 1000.0)))
        a_rel = float(np.exp(-1.0 / (sr * cfg.chain.agc_release_ms / 1000.0)))
        fast = lfilter([1 - a_att], [1.0, -a_att], ax)
        env = lfilter([1 - a_rel], [1.0, -a_rel], fast)
        env = np.maximum(env, fast * 0.35)
        g = cfg.chain.agc_target_rms / (env + 1e-6)
        gmax = 10 ** (cfg.chain.agc_max_gain_db / 20)
        gmin = 10 ** (cfg.chain.agc_min_gain_db / 20)
        return np.clip(g, gmin, gmax).astype(np.float32)
    except Exception:
        return agc_gain(x, cfg, sr)


# Capsule headroom is set relative to speech at normal level (rms 0.03-0.09).
# MARK6.0-6.4 used (0.7,1.1) / (0.35,0.6), which saturated the wearer's own
# voice (measured voice-only distortion -36 / -26 dB): a mic that distorts on
# normal speech is broken hardware, not a harsh environment. Now the voice
# alone saturates only mildly on loud peaks; blasts still drive the capsule
# hard, which is the realistic failure.
CHAIN_PRESETS = {
    "clean":    dict(headroom=(1.2, 1.6), adc_clip=1.0, bits=16, agc=False),
    "moderate": dict(headroom=(1.0, 1.5), adc_clip=1.0, bits=16, agc=True),
    "severe":   dict(headroom=(0.6, 0.9), adc_clip=0.85, bits=12, agc=True),
}


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    if bits >= 16:
        q = 32768.0
    else:
        q = float(2 ** (bits - 1))
    return (np.round(x * q) / q).astype(np.float32)


# --------------------------------------------------------------------------
# source pools
# --------------------------------------------------------------------------

def speaker_of(path: str) -> str:
    """LibriSpeech: <spk>-<chapter>-<utt>.flac -> <spk>.
    Other sources: the file itself is the only identity we have."""
    b = os.path.basename(path)
    parts = b.split("-")
    if len(parts) == 3 and parts[0].isdigit() and b.endswith(".flac"):
        return "ls" + parts[0]
    return "file:" + b


class SourcePool:
    """Indexes the raw dataset and provides disjoint train/val/test partitions.

    Weapons are partitioned by *folder*, so a weapon that appears in test never
    contributes a single sample to training. This is the split that earlier
    pipelines lost by flattening the weapon directories with rglob.
    """

    def __init__(self, voice_dir: str, gun_dir: str, exp_dir: str, cfg: Cfg = DEFAULT,
                 voice_split_override: Optional[Dict[str, List[str]]] = None):
        self.cfg = cfg
        if voice_dir is None and voice_split_override:
            voice_dir = os.path.dirname(next(iter(voice_split_override.values()))[0])
        self.voice_files = sorted(
            [p for p in glob.glob(os.path.join(voice_dir, "**", "*"), recursive=True)
             if os.path.isfile(p) and os.path.splitext(p)[1].lower()
             in (".wav", ".m4a", ".ogg", ".opus", ".mp3", ".flac")])
        self.weapons = sorted(
            [d for d in os.listdir(gun_dir) if os.path.isdir(os.path.join(gun_dir, d))])
        self.gun_by_weapon = {
            w: sorted(glob.glob(os.path.join(gun_dir, w, "**", "*.wav"), recursive=True))
            for w in self.weapons}
        self.exp_files = sorted(
            glob.glob(os.path.join(exp_dir, "**", "*.wav"), recursive=True))

        c = cfg.corpus
        self.test_w = [w for w in self.weapons if w in c.test_weapons]
        self.val_w = [w for w in self.weapons if w in c.val_weapons]
        self.train_w = [w for w in self.weapons
                        if w not in self.test_w and w not in self.val_w]

        rng = np.random.RandomState(c.seed)
        # voice + explosion partitioned by file
        vi = rng.permutation(len(self.voice_files))
        n_v = len(vi)
        self.voice_split = {
            "test": [self.voice_files[i] for i in vi[:max(1, int(0.15 * n_v))]],
            "val": [self.voice_files[i] for i in vi[int(0.15 * n_v):int(0.30 * n_v)]],
            "train": [self.voice_files[i] for i in vi[int(0.30 * n_v):]],
        }
        ei = rng.permutation(len(self.exp_files))
        n_e = len(ei)
        self.exp_split = {
            "test": [self.exp_files[i] for i in ei[:max(1, int(0.15 * n_e))]],
            "val": [self.exp_files[i] for i in ei[int(0.15 * n_e):int(0.30 * n_e)]],
            "train": [self.exp_files[i] for i in ei[int(0.30 * n_e):]],
        }
        self.weapon_split = {"train": self.train_w, "val": self.val_w, "test": self.test_w}

        # Speaker-disjoint voice splits supplied from outside (e.g. LibriSpeech
        # speaker IDs). Replaces the by-file split, which cannot be
        # speaker-disjoint when speaker identity is unknown.
        if voice_split_override:
            self.voice_split = {k: list(v) for k, v in voice_split_override.items()}
            self.voice_files = sorted({p for v in self.voice_split.values() for p in v})
            # any split without its own weapons/explosions reuses test's
            for k in self.voice_split:
                self.weapon_split.setdefault(k, self.test_w)
                self.exp_split.setdefault(k, self.exp_split["test"])

    def guns(self, split: str) -> List[str]:
        out = []
        for w in self.weapon_split[split]:
            out.extend(self.gun_by_weapon[w])
        return out

    def summary(self) -> Dict:
        def spk(p):
            return speaker_of(p)
        return {
            "voice_files": len(self.voice_files),
            "voice_split": {k: len(v) for k, v in self.voice_split.items()},
            "speakers_per_split": {k: len({spk(p) for p in v})
                                   for k, v in self.voice_split.items()},
            "speaker_overlap": sorted(
                ({spk(p) for p in self.voice_split.get("train", [])} &
                 ({spk(p) for p in self.voice_split.get("test", [])} |
                  {spk(p) for p in self.voice_split.get("val", [])}))),
            "weapons": self.weapons,
            "weapon_split": self.weapon_split,
            "gun_files": {k: len(self.guns(k)) for k in ("train", "val", "test")},
            "explosion_split": {k: len(v) for k, v in self.exp_split.items()},
        }


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------

def place_event(n: int, ev_len: int, overlap: str,
                rng: np.random.RandomState) -> Optional[int]:
    """Choose the start sample for an event given the requested overlap mode."""
    if overlap == "none":
        return None
    if ev_len >= n:
        return 0
    if overlap == "onset":
        hi = max(1, int(0.2 * n))
        return rng.randint(0, hi)
    if overlap == "mid":
        lo, hi = int(0.3 * n), max(int(0.3 * n) + 1, int(0.7 * n) - ev_len)
        return rng.randint(lo, max(lo + 1, hi))
    return rng.randint(0, n - ev_len)


def make_example(voice_path: str, gun_paths: List[str], exp_paths: List[str],
                 situation: Dict, cfg: Cfg, rng: np.random.RandomState,
                 split: str = "train", pseudo_speaker: bool = True) -> Dict:
    """Render one (mix, target) pair plus metadata for a given situation."""
    sr = cfg.sig.sr
    n = cfg.sig.n_samples

    # ---- voice ----------------------------------------------------------
    v = dc_block(A.load_audio(voice_path, sr), sr)
    v = A.fit_length(v, n, mode="pad" if len(v) < n else "crop")

    # Pseudo-speaker identity, drawn from a warp set that is disjoint across
    # splits so a test talker is never heard during training.
    spk_id = "native"
    if pseudo_speaker:
        from .speaker_aug import apply_pseudo_speaker
        v, spk_id = apply_pseudo_speaker(v, sr, split, rng)

    v = A.set_rms(v, rng.uniform(0.03, 0.09))

    room = situation["room"]
    dist = situation["distance"]
    v, _ = distance_filter(v, "near", sr, cfg, rng)          # talker is close
    drr_v = voice_drr(rng)
    rir_v_full, rir_v_early = synth_rir_pair(room, sr, rng, drr_v)
    v_room = convolve_rir(v, rir_v_full)     # what the microphone hears
    v_early = convolve_rir(v, rir_v_early)   # what the TARGET keeps: direct + early
    drr_e = event_drr(dist, rng)
    rir_e_full, _ = synth_rir_pair(room, sr, rng, drr_e)

    # ---- event ----------------------------------------------------------
    ev = np.zeros(n, dtype=np.float32)
    ev_span = np.zeros(n, dtype=bool)
    et = situation["event_type"]
    chosen = []
    if et in ("gunshot", "both") and gun_paths:
        chosen.append(("gunshot", gun_paths[rng.randint(len(gun_paths))]))
    if et in ("explosion", "both") and exp_paths:
        chosen.append(("explosion", exp_paths[rng.randint(len(exp_paths))]))

    for kind, path in chosen:
        e = dc_block(A.load_audio(path, sr), sr)
        if len(e) > n:
            e = e[:n]
        e = e / (np.max(np.abs(e)) + 1e-9)
        e_room = convolve_rir(e, rir_e_full)
        e_room, radius = distance_filter(e_room, dist, sr, cfg, rng)
        start = place_event(n, len(e_room), situation["overlap"], rng)
        if start is None:
            continue
        end = min(n, start + len(e_room))
        ev[start:end] += e_room[:end - start]
        ev_span[start:end] = True

    # ---- SNR scaling, measured inside the event span --------------------
    if ev_span.any() and np.any(np.abs(ev) > 0):
        snr = situation["snr_db"]
        v_pow = np.mean(v_room[ev_span] ** 2) + 1e-12
        e_pow = np.mean(ev[ev_span] ** 2) + 1e-12
        scale = np.sqrt(v_pow / (e_pow * (10 ** (snr / 10.0))))
        ev = ev * scale
    else:
        ev_span[:] = False

    # ---- capture chain --------------------------------------------------
    preset = CHAIN_PRESETS[situation["chain"]]
    headroom = rng.uniform(*preset["headroom"])

    # Microphone self-noise. Real capsules have a noise floor; adding it here
    # also prevents zero-padded speech from creating digital silence, which
    # earlier pipelines mislabelled as high-confidence "event absent" frames.
    nf_db = rng.uniform(-72.0, -58.0)
    nfloor = rng.randn(n).astype(np.float32) * (10 ** (nf_db / 20.0))

    # The mic self-noise goes into the MIXTURE only. Previously it was added
    # to the voice path and so ended up in the target, where the AGC lifted it
    # ~17 dB in pauses (measured: quietest target frames -43 dBFS on AGC
    # chains, -61 dBFS without) -- the model was being taught to keep hiss.
    x_lin = v_room + ev + nfloor
    x_sat = capsule_saturate(x_lin, headroom)
    # (MARK6.4 note, superseded below) Target kept the voice as the mic heard it,
    # including its (small, realistic) room ambience. A dereverberation target
    # (direct + early only) was tried in MARK6.3 and measured worse: on clean
    # clips the filter path reached 19.8 dB against an input of 24.2 dB,
    # because a 5-tap causal filter (40 ms of history) cannot cancel reverb
    # that is 50-1000 ms old, and suppressing it costs speech. The "empty
    # room" sound came from the miscalibrated DRR (-10..-15 dB), not from the
    # presence of any reverb at all.
    # (voice-only saturation is no longer part of the target -- see below)

    if preset["agc"]:
        g = _agc_gain_fast(x_lin, cfg, sr)          # AGC reacts to what it hears
    else:
        g = np.ones(n, dtype=np.float32)

    adc = preset["adc_clip"]
    mix = np.clip(x_sat * g, -adc, adc)

    # TARGET = the linear voice at a steady level.
    # Previously the target was the voice pushed through the same capsule
    # saturation, AGC and ADC clip as the mixture. That taught the model to
    # KEEP two things that sound like a radio: harmonic grit from saturation
    # (voice-only distortion -24..-32 dB) and AGC pumping (voice ducked ~7 dB
    # after every blast, recovering over ~250 ms). A closed-form check
    # (check_desat.py) showed that target capped the output at ~16 dB SNR on
    # blast clips, while the network's own output stage can reach ~29 dB
    # against the linear voice.
    # Level: an ideal steady AGC -- one gain per clip that puts the talker's
    # active speech at the AGC's own target level. No pumping, and no boosting
    # of pauses (which a real fast AGC does, and which would re-inject noise).
    if preset["agc"]:
        vv = v_room
        fl = 256; nfr = len(vv) // fl
        fe = (vv[:nfr * fl].reshape(nfr, fl) ** 2).mean(1)
        act = fe > fe.max() * 1e-4                       # within 40 dB of peak
        v_act_rms = np.sqrt(fe[act].mean() + 1e-12) if act.any() else 1e-3
        G = cfg.chain.agc_target_rms / v_act_rms
        G = float(np.clip(G, 10 ** (cfg.chain.agc_min_gain_db / 20),
                          10 ** (cfg.chain.agc_max_gain_db / 20)))
    else:
        G = 1.0
    tgt = (v_room * G).astype(np.float32)
    v_sat = None
    pk = float(np.max(np.abs(tgt)))
    if pk > 0.98:                    # keep the linear target representable in 16-bit wav;
        sc = 0.98 / pk               # scale mix identically so their relation is unchanged
        tgt = tgt * sc
        mix = mix * sc

    clipped = (np.abs(x_sat * g) >= adc * 0.999)

    mix = quantize(mix, preset["bits"])
    # the target is not pushed through the ADC's reduced bit depth either:
    # quantisation noise is part of the degradation, not of the voice

    meta = {
        "voice_file": os.path.basename(voice_path),
        "speaker": speaker_of(voice_path),
        "pseudo_speaker": spk_id,
        "event_type": et,
        "event_sources": [os.path.basename(p) for _, p in chosen],
        "weapons": [os.path.basename(os.path.dirname(p))
                    for k, p in chosen if k == "gunshot"],
        "snr_db": situation["snr_db"] if ev_span.any() else None,
        "overlap": situation["overlap"],
        "distance": dist,
        "room": room,
        "chain": situation["chain"],
        "headroom": round(float(headroom), 4),
        "drr_voice_db": round(float(drr_v), 2),
        "drr_event_db": round(float(drr_e), 2),
        "target": "linear close-talk voice, steady level (no saturation, no clip, no AGC pumping, no hiss)",
        "event_span": [int(np.argmax(ev_span)), int(n - np.argmax(ev_span[::-1]))]
                      if ev_span.any() else None,
        "clipped_frac": round(float(clipped.mean()), 5),
        "agc_gain_db_range": [round(float(20 * np.log10(g.min() + 1e-9)), 2),
                              round(float(20 * np.log10(g.max() + 1e-9)), 2)],
    }
    return {"mix": mix, "target": tgt, "meta": meta}


def sample_situation(rng: np.random.RandomState, cfg: Cfg) -> Dict:
    c = cfg.corpus
    et = c.event_types[rng.randint(len(c.event_types))]
    ov = "none" if et == "none" else c.overlap[rng.randint(1, len(c.overlap))]
    return {
        "event_type": et,
        "snr_db": float(c.snr_db[rng.randint(len(c.snr_db))]),
        "overlap": ov,
        "distance": c.distance[rng.randint(len(c.distance))],
        "room": c.room[rng.randint(len(c.room))],
        "chain": c.chain[rng.randint(len(c.chain))],
    }


def build_corpus(pool: SourcePool, out_dir: str, cfg: Cfg = DEFAULT,
                 counts: Optional[Dict[str, int]] = None, verbose: bool = True):
    """Render the full situation corpus to disk.

    Layout:
        out_dir/<split>/<id>/mix.wav      what the model receives
        out_dir/<split>/<id>/target.wav   what it must produce
        out_dir/<split>/manifest.jsonl    one metadata record per example

    The residual (mix - target) is recoverable at load time, so it is not
    stored separately.
    """
    counts = counts or {"train": cfg.corpus.n_train,
                        "val": cfg.corpus.n_val,
                        "test": cfg.corpus.n_test}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "source_split.json"), "w") as f:
        json.dump(pool.summary(), f, indent=2)

    for split, count in counts.items():
        rng = np.random.RandomState(cfg.corpus.seed + hash(split) % 10000)
        vps = pool.voice_split[split]
        gps = pool.guns(split)
        eps = pool.exp_split[split]
        sdir = os.path.join(out_dir, split)
        os.makedirs(sdir, exist_ok=True)
        man = open(os.path.join(sdir, "manifest.jsonl"), "w")
        for i in range(count):
            sit = sample_situation(rng, cfg)
            vp = vps[rng.randint(len(vps))]
            try:
                ex = make_example(vp, gps, eps, sit, cfg, rng)
            except Exception as e:
                if verbose:
                    print(f"  skip {split}/{i}: {e}")
                continue
            eid = f"{i:06d}"
            d = os.path.join(sdir, eid)
            A.save_wav(os.path.join(d, "mix.wav"), ex["mix"], cfg.sig.sr)
            A.save_wav(os.path.join(d, "target.wav"), ex["target"], cfg.sig.sr)
            rec = dict(ex["meta"]); rec["id"] = eid; rec["split"] = split
            man.write(json.dumps(rec) + "\n")
            if verbose and (i + 1) % 100 == 0:
                print(f"  {split}: {i+1}/{count}")
        man.close()
        if verbose:
            print(f"[corpus] {split}: {count} examples -> {sdir}")
