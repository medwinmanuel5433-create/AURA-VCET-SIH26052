"""Dual-microphone capture: the primary (boom / voice) mic exactly as in
physmix.make_example, plus a REFERENCE mic on the outside of the ear cup.

Reference-mic physics (what differs from the primary mic):
  * voice: the mouth is ~3x farther away -> voice arrives 10..18 dB weaker,
    with its own room response (lower direct-to-reverberant ratio);
  * event: same source, its own position -> same direct sound (slightly
    different gain and a 0..0.4 ms time offset), early reflections re-drawn,
    late diffuse tail only PARTLY shared with the primary mic
    (coherence rho 0.5..0.9, spacing of a few cm);
  * its own self-noise and its own capsule saturation, NO AGC (ANC/ENC
    reference mics are read raw), same ADC clip and bit depth.
Everything that makes the primary channel is identical to MARK6/MARK7's
corpus generator, so the single-mic numbers stay comparable.
"""
import os
import numpy as np
RHO = tuple(float(v) for v in os.environ.get("REF_RHO", "0.5,0.9").split(","))
from . import physmix as P
from . import audio as A


def _ref_rir(rir_full, room, sr, rng, drr_db, rho, delay):
    """Reference-mic room response correlated with the primary one."""
    if room == "anechoic" or len(rir_full) == 1:
        r = np.zeros(delay + 1, np.float32); r[delay] = rng.uniform(0.8, 1.25)
        return r
    own, _ = P.synth_rir_pair(room, sr, rng, drr_db)
    n = max(len(own), len(rir_full))
    a = np.zeros(n, np.float32); a[:len(rir_full)] = rir_full; a[0] -= 1.0          # primary reverb only
    b = np.zeros(n, np.float32); b[:len(own)] = own; b[0] -= 1.0                     # independent reverb
    b *= np.sqrt((np.sum(a ** 2) + 1e-12) / (np.sum(b ** 2) + 1e-12))
    rev = rho * a + np.sqrt(max(0.0, 1 - rho ** 2)) * b
    out = np.zeros(n + delay, np.float32)
    out[delay] += rng.uniform(0.8, 1.25)                                             # direct sound
    out[delay:delay + n] += rev
    return out


def make_dual_example(voice_path, gun_paths, exp_paths, situation, cfg, rng, split="test"):
    sr, n = cfg.sig.sr, cfg.sig.n_samples
    v = P.dc_block(A.load_audio(voice_path, sr), sr)
    v = A.fit_length(v, n, mode="pad" if len(v) < n else "crop")
    v = A.set_rms(v, rng.uniform(0.03, 0.09))
    room, dist = situation["room"], situation["distance"]
    v, _ = P.distance_filter(v, "near", sr, cfg, rng)
    drr_v = P.voice_drr(rng)
    rir_v_full, _ = P.synth_rir_pair(room, sr, rng, drr_v)
    v_room = P.convolve_rir(v, rir_v_full)
    drr_e = P.event_drr(dist, rng)
    rir_e_full, _ = P.synth_rir_pair(room, sr, rng, drr_e)

    # reference-mic paths
    v_leak_db = rng.uniform(-18.0, -10.0)
    rir_v_ref, _ = P.synth_rir_pair(room, sr, rng, drr_v - 8.0)
    v_ref = P.convolve_rir(v, rir_v_ref) * 10 ** (v_leak_db / 20)
    rho = rng.uniform(*RHO); delay = rng.randint(0, 7)          # 0..0.4 ms
    rir_e_ref = _ref_rir(rir_e_full, room, sr, rng, drr_e, rho, delay)

    ev = np.zeros(n, np.float32); ev_r = np.zeros(n, np.float32); span = np.zeros(n, bool)
    et = situation["event_type"]; chosen = []
    if et in ("gunshot", "both") and gun_paths:
        chosen.append(("gunshot", gun_paths[rng.randint(len(gun_paths))]))
    if et in ("explosion", "both") and exp_paths:
        chosen.append(("explosion", exp_paths[rng.randint(len(exp_paths))]))
    for kind, path in chosen:
        e = P.dc_block(A.load_audio(path, sr), sr)[:n]
        e = e / (np.max(np.abs(e)) + 1e-9)
        st = rng.get_state()
        e_room, _ = P.distance_filter(P.convolve_rir(e, rir_e_full), dist, sr, cfg, rng)
        rng.set_state(st)                                            # same distance draw for the ref mic
        e_ref, _ = P.distance_filter(P.convolve_rir(e, rir_e_ref), dist, sr, cfg, rng)
        start = P.place_event(n, len(e_room), situation["overlap"], rng)
        if start is None:
            continue
        end = min(n, start + len(e_room))
        ev[start:end] += e_room[:end - start]; ev_r[start:end] += e_ref[:end - start]; span[start:end] = True

    if span.any() and np.any(np.abs(ev) > 0):
        scale = np.sqrt((np.mean(v_room[span] ** 2) + 1e-12) /
                        ((np.mean(ev[span] ** 2) + 1e-12) * 10 ** (situation["snr_db"] / 10)))
        ev *= scale; ev_r *= scale
    else:
        span[:] = False

    pre = P.CHAIN_PRESETS[situation["chain"]]
    head = rng.uniform(*pre["headroom"]); head_r = rng.uniform(*pre["headroom"])
    nf = rng.randn(n).astype(np.float32) * 10 ** (rng.uniform(-72, -58) / 20)
    nf_r = rng.randn(n).astype(np.float32) * 10 ** (rng.uniform(-72, -58) / 20)
    x_lin = v_room + ev + nf
    x_sat = P.capsule_saturate(x_lin, head)
    g = P._agc_gain_fast(x_lin, cfg, sr) if pre["agc"] else np.ones(n, np.float32)
    adc = pre["adc_clip"]
    mix = np.clip(x_sat * g, -adc, adc)
    ref = np.clip(P.capsule_saturate(v_ref + ev_r + nf_r, head_r), -adc, adc)

    if pre["agc"]:
        fl = 256; nfr = n // fl
        fe = (v_room[:nfr * fl].reshape(nfr, fl) ** 2).mean(1); act = fe > fe.max() * 1e-4
        G = cfg.chain.agc_target_rms / (np.sqrt(fe[act].mean() + 1e-12) if act.any() else 1e-3)
        G = float(np.clip(G, 10 ** (cfg.chain.agc_min_gain_db / 20), 10 ** (cfg.chain.agc_max_gain_db / 20)))
    else:
        G = 1.0
    tgt = (v_room * G).astype(np.float32)
    pk = float(np.max(np.abs(tgt)))
    if pk > 0.98:
        tgt *= 0.98 / pk; mix *= 0.98 / pk
    mix = P.quantize(mix, pre["bits"]); ref = P.quantize(ref, pre["bits"])
    meta = dict(voice_file=os.path.basename(voice_path), event_type=et,
                event_sources=[os.path.basename(p) for _, p in chosen],
                weapons=[os.path.basename(os.path.dirname(p)) for k, p in chosen if k == "gunshot"],
                snr_db=situation["snr_db"] if span.any() else None, overlap=situation["overlap"],
                distance=dist, room=room, chain=situation["chain"],
                ref_voice_leak_db=round(float(v_leak_db), 1), ref_rho=round(float(rho), 2),
                event_span=[int(np.argmax(span)), int(n - np.argmax(span[::-1]))] if span.any() else None)
    return {"mix": mix.astype(np.float32), "ref": ref.astype(np.float32), "target": tgt, "meta": meta}
