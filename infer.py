"""AURA inference: clean one recording (voice mic + reference mic) from the command line.

    python infer.py --voice boom_mic.wav --ref reference_mic.wav --out clean.wav
    python infer.py --voice boom_mic.wav --out clean.wav                 # 1-mic MARK7 (no reference mic)
    python infer.py --voice boom_mic.wav --ref reference_mic.wav --out clean.wav --model v3 --no-loud

Models:  v4 (default) = MARK7-DM v3 + reference-mic post-filter (attenuation-only)
         v3           = MARK7-DM v3 network output
         (no --ref)   = MARK7 single-mic pipeline
Input audio is resampled to 16 kHz mono. Runs on CPU.
"""
import argparse, os, sys, shutil

REPO = os.path.dirname(os.path.abspath(__file__))
RT = os.path.join(REPO, ".runtime")
WEIGHTS = {"mark7_network.pt": "m65_best.pt", "dictionaries.pt": "dicts_m65.pt",
           "mark7dm_network.pt": "m7dm_final.pt", "mark7dm2_network.pt": "m7dm2_final.pt",
           "mark7dm3_network.pt": "m7dm3_final.pt"}


def _link(src, dst):
    if os.path.lexists(dst):
        return
    try:
        os.symlink(src, dst)
    except OSError:                                   # e.g. Windows without symlink rights
        (shutil.copytree if os.path.isdir(src) else shutil.copy)(src, dst)


def build_runtime():
    """Lay out the runtime folder the MARK7 loader expects (weights/, library/, aura/)."""
    os.makedirs(os.path.join(RT, "weights"), exist_ok=True)
    for dst, src in WEIGHTS.items():
        _link(os.path.join(REPO, "weights", src), os.path.join(RT, "weights", dst))
    _link(os.path.join(REPO, "mark7_bundle", "library"), os.path.join(RT, "library"))
    _link(os.path.join(REPO, "code", "aura"), os.path.join(RT, "aura"))
    shutil.copy(os.path.join(REPO, "mark7_bundle", "run_mark7.py"), RT)
    sys.path.insert(0, RT)


def load(path, sr=16000):
    import numpy as np, soundfile as sf
    x, s = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if s != sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(s, sr); x = resample_poly(x, sr // g, s // g).astype(np.float32)
    return x


def main():
    ap = argparse.ArgumentParser(description="AURA voice-preserving blast suppression")
    ap.add_argument("--voice", required=True, help="boom (voice) mic recording")
    ap.add_argument("--ref", help="outer reference mic recording (same length, same start)")
    ap.add_argument("--out", required=True, help="output .wav")
    ap.add_argument("--model", choices=["v4", "v3"], default="v4")
    ap.add_argument("--no-loud", action="store_true", help="skip -18 LUFS loudness normalisation")
    a = ap.parse_args()

    build_runtime()
    import numpy as np, soundfile as sf, torch
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    import run_mark7 as R

    x = load(a.voice)
    if a.ref:
        r = load(a.ref); n = min(len(x), len(r)); x, r = x[:n], r[:n]
        pipe = (R.Mark7V4 if a.model == "v4" else R.Mark7V3)(RT, loud=not a.no_loud)
        y = pipe(x, r)
    else:
        pipe = R.load_mark7(RT)
        y = pipe(x)
        if not a.no_loud:
            from aura.loudness import make_loud
            y = make_loud(np.asarray(y, np.float32))
    sf.write(a.out, np.asarray(y, np.float32), 16000, subtype="PCM_16")
    print(f"saved {a.out}  ({len(y) / 16000:.1f} s, model {'MARK7 1-mic' if not a.ref else a.model})")


if __name__ == "__main__":
    main()
