"""Evaluation metrics.

Two rules carried over from the earlier post-mortems:

1. Scoring happens inside the *event window*, not over the whole clip. A 7 s
   clip containing a 300 ms blast will show an excellent full-clip SNR while
   the speech under the blast is destroyed. Full-clip numbers are reported
   alongside, but the gate uses the event window.

2. A metric that fails to compute is a failure, not a skip. Earlier code
   averaged over successful PESQ calls only, so clips where PESQ crashed --
   typically the hardest ones -- silently improved the mean. Here failures are
   counted and surfaced.
"""
import numpy as np

try:
    from pystoi import stoi as _stoi
    HAVE_STOI = True
except Exception:
    HAVE_STOI = False

try:
    from pesq import pesq as _pesq
    HAVE_PESQ = True
except Exception:
    HAVE_PESQ = False


def snr_db(est, ref, eps=1e-10):
    ref = ref[:len(est)]
    est = est[:len(ref)]
    noise = est - ref
    return 10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum(noise ** 2) + eps))


def si_sdr_db(est, ref, eps=1e-10):
    ref = ref[:len(est)] - np.mean(ref[:len(est)])
    est = est[:len(ref)] - np.mean(est[:len(ref)])
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    proj = alpha * ref
    return 10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum((est - proj) ** 2) + eps))


def event_window(meta, n, sr, pad_ms=150.0):
    """Sample range to score, expanded slightly around the event."""
    span = meta.get("event_span")
    if not span:
        return None
    pad = int(sr * pad_ms / 1000)
    return max(0, span[0] - pad), min(n, span[1] + pad)


def evaluate_clip(est, ref, mix, meta, sr=16000):
    """Return a dict of metrics for one clip. NaN means 'could not compute',
    and is propagated rather than dropped."""
    n = min(len(est), len(ref), len(mix))
    est, ref, mix = est[:n], ref[:n], mix[:n]
    out = {}

    out["snr_full"] = snr_db(est, ref)
    out["snr_in_full"] = snr_db(mix, ref)
    out["sisdr_full"] = si_sdr_db(est, ref)

    # Polarity guard. STOI, PESQ and SI-SDR are all blind to a sign flip, so a
    # fully inverted output can score well on every perceptual metric while
    # being -6 dB on true SNR. Run 1 of this model did exactly that. This is
    # reported per clip and enforced in the gate.
    den = (np.linalg.norm(est) * np.linalg.norm(ref)) + 1e-12
    out["polarity_corr"] = float(np.dot(est, ref) / den)

    # Whether this clip contains an event decides which question is being
    # asked, and the two must never be averaged together. On an event clip the
    # question is "how much did you improve it". On a clean clip the input is
    # already near-perfect (input SNR can exceed 60 dB), so *any* processing
    # scores as a large loss, and mixing the two produces a headline number
    # that is dominated by clean clips and says nothing about either case.
    out["has_event"] = 1.0 if meta.get("event_span") else 0.0

    win = event_window(meta, n, sr)
    if win:
        a, b = win
        if b - a > sr * 0.05:
            out["snr_evt"] = snr_db(est[a:b], ref[a:b])
            out["snr_in_evt"] = snr_db(mix[a:b], ref[a:b])
            out["sisdr_evt"] = si_sdr_db(est[a:b], ref[a:b])
    else:
        # no event: the metric that matters is how little was damaged
        out["snr_evt"] = out["snr_full"]
        out["snr_in_evt"] = out["snr_in_full"]
        out["sisdr_evt"] = out["sisdr_full"]

    # Impulse window: only the frames where the event actually dominates the
    # speech. The event window above spans the whole source file (median
    # 4.9 s of a 7 s clip, measured), most of which is the quiet tail after a
    # ~100 ms shot -- so it mostly measures easy material. This is the hard
    # part, reported separately.
    fl = 256
    nfr = n // fl
    if nfr > 0:
        res = (mix[:nfr * fl] - ref[:nfr * fl]).reshape(nfr, fl)
        tg = ref[:nfr * fl].reshape(nfr, fl)
        dom = (res ** 2).sum(1) > (tg ** 2).sum(1)
        dom = dom | np.r_[False, dom[:-1]] | np.r_[dom[1:], False]
        if dom.sum() * fl >= sr * 0.05:
            m = np.repeat(dom, fl)
            e_, r_, x_ = est[:nfr * fl][m], ref[:nfr * fl][m], mix[:nfr * fl][m]
            out["snr_imp"] = snr_db(e_, r_)
            out["snr_in_imp"] = snr_db(x_, r_)
            out["delta_snr_imp"] = out["snr_imp"] - out["snr_in_imp"]
            out["imp_frac"] = float(dom.mean())

    # Pause noise: what is left in the output where the target voice is
    # silent, relative to the voice's own active level. This is the number
    # for "mild noise spread across the whole signal". Lower is better.
    fl2 = 256; nf2 = n // fl2
    if nf2 > 0:
        te = (ref[:nf2 * fl2].reshape(nf2, fl2) ** 2).mean(1)
        ee = (est[:nf2 * fl2].reshape(nf2, fl2) ** 2).mean(1)
        tdb = 10 * np.log10(te + 1e-14)
        act = tdb > tdb.max() - 30
        pau = tdb < tdb.max() - 45
        if act.sum() >= 5 and pau.sum() >= 5:
            out["pause_noise_db"] = float(10 * np.log10(ee[pau].mean() + 1e-14)
                                          - 10 * np.log10(te[act].mean() + 1e-14))

    if HAVE_STOI:
        try:
            out["stoi"] = float(_stoi(ref, est, sr, extended=False))
        except Exception:
            out["stoi"] = float("nan")
        if win and win[1] - win[0] >= sr * 0.5:
            try:
                a, b = win
                out["stoi_evt"] = float(_stoi(ref[a:b], est[a:b], sr, extended=False))
            except Exception:
                out["stoi_evt"] = float("nan")
    else:
        out["stoi"] = float("nan")

    if HAVE_PESQ:
        try:
            out["pesq"] = float(_pesq(sr, ref, est, "wb"))
        except Exception:
            out["pesq"] = float("nan")
    else:
        out["pesq"] = float("nan")

    for k in ("snr_evt", "snr_in_evt"):
        if k in out and not np.isfinite(out[k]):
            out[k] = float("nan")
    if "snr_evt" in out and "snr_in_evt" in out:
        out["delta_snr_evt"] = out["snr_evt"] - out["snr_in_evt"]
    return out


def aggregate_split(rows):
    """Aggregate separately for event and no-event clips.

    Returns {"event": {...}, "clean": {...}, "all": {...}}. The gate reads the
    first two; "all" is kept only for continuity with older reports and must
    not be used to judge the system.
    """
    ev = [r for r in rows if r.get("has_event", 0.0) >= 0.5]
    cl = [r for r in rows if r.get("has_event", 0.0) < 0.5]
    return {"event": aggregate(ev), "clean": aggregate(cl),
            "all": aggregate(rows),
            "n_event": len(ev), "n_clean": len(cl)}


def aggregate(rows):
    """Aggregate clip metrics, counting failures explicitly."""
    if not rows:
        return {}
    keys = set()
    for r in rows:
        keys |= set(r.keys())
    agg = {}
    for k in sorted(keys):
        vals = np.array([r.get(k, np.nan) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        agg[k] = {
            "mean": float(np.mean(vals[finite])) if finite.any() else float("nan"),
            "median": float(np.median(vals[finite])) if finite.any() else float("nan"),
            "n_ok": int(finite.sum()),
            "n_fail": int((~finite).sum()),
        }
    return agg


def pass_gate_split(split_agg, min_delta_snr=10.0, min_stoi=0.85,
                    min_pesq=2.5, min_clean_snr=25.0, n_total=None):
    """The real gate: event clips must improve, clean clips must survive.

    A system that scores well on events by damaging clean audio is not
    acceptable, and neither is the reverse. Both conditions are required.
    """
    ev, cl = split_agg.get("event", {}), split_agg.get("clean", {})
    ok_ev, why_ev = pass_gate(ev, min_delta_snr, min_stoi, min_pesq,
                              n_total=split_agg.get("n_event"))
    reasons = [f"[event] {r}" for r in why_ev]
    cs = cl.get("snr_evt", {})
    if cl and np.isfinite(cs.get("mean", np.nan)) and cs["mean"] < min_clean_snr:
        reasons.append(
            f"[clean] passthrough SNR {cs['mean']:.2f} dB < {min_clean_snr} "
            "-- the model is damaging audio that contained no event")
    return (len(reasons) == 0), reasons


def pass_gate(agg, min_delta_snr=10.0, min_stoi=0.85, min_pesq=2.5,
              max_fail_frac=0.02, n_total=None):
    """The PASS rule. Uses event-window improvement, and refuses to pass if a
    meaningful share of clips failed to score."""
    reasons = []
    pc = agg.get("polarity_corr", {})
    if np.isfinite(pc.get("mean", np.nan)) and pc["mean"] < 0.0:
        reasons.append(
            f"polarity inverted (corr {pc['mean']:.3f}) -- output is sign-flipped; "
            "perceptual metrics cannot detect this")
    d = agg.get("delta_snr_evt", {})
    if not np.isfinite(d.get("mean", np.nan)):
        reasons.append("delta_snr_evt not computable")
    elif d["mean"] < min_delta_snr:
        reasons.append(f"delta_snr_evt {d['mean']:.2f} < {min_delta_snr}")
    s = agg.get("stoi", {})
    if np.isfinite(s.get("mean", np.nan)) and s["mean"] < min_stoi:
        reasons.append(f"stoi {s['mean']:.3f} < {min_stoi}")
    p = agg.get("pesq", {})
    if np.isfinite(p.get("mean", np.nan)) and p["mean"] < min_pesq:
        reasons.append(f"pesq {p['mean']:.3f} < {min_pesq}")
    if n_total:
        for k in ("stoi", "pesq", "delta_snr_evt"):
            kk = agg.get(k, {})
            if kk and kk.get("n_fail", 0) / max(n_total, 1) > max_fail_frac:
                reasons.append(f"{k} failed on {kk['n_fail']}/{n_total} clips")
    return (len(reasons) == 0), reasons
