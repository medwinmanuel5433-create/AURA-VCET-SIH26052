"""AURA MARK7 - load and run.

    from run_mark7 import load_mark7, enhance_file
    pipe = load_mark7(".")                      # mode "clarity" (default) or "max_quiet"
    y = pipe(x)                                 # x: float32 mono 16 kHz numpy array

CLI:  python run_mark7.py input.wav output.wav [--mode max_quiet] [--no-library] [--ref reference_mic.wav]
"""
import os, sys, glob, argparse
import numpy as np, torch, soundfile as sf
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from aura.config import DEFAULT as cfg
from aura.model import Mark6
from aura.mark7 import Mark7, MODES
from aura.audio import load_audio

SR = 16000


def load_mark7(root=ROOT, mode="clarity", use_library=True, threads=None, dual=False, **overrides):
    """dual=True loads MARK7-DM (two mics): call pipe(voice_mic, ref=reference_mic)."""
    if threads:
        torch.set_num_threads(threads)
    if dual:
        from aura.model_dm import Mark7DM
        net = Mark7DM(cfg); wf = "mark7dm_network.pt"
    else:
        net = Mark6(cfg); wf = "mark7_network.pt"
    net.load_state_dict(torch.load(os.path.join(root, "weights", wf), map_location="cpu")["model"])
    net.eval()
    D = torch.load(os.path.join(root, "weights", "dictionaries.pt"), map_location="cpu")
    lib = []
    if use_library:
        for f in sorted(glob.glob(os.path.join(root, "library", "*.wav"))):
            x, sr = sf.read(f, dtype="float32")
            lib.append(x)
    kw = dict(MODES[mode]); kw.update(overrides)
    return Mark7(net, cfg, D["W_v"], D["W_r"], library=lib, use_ref=bool(lib), **kw)


def enhance_file(pipe, src, dst, ref=None):
    x = load_audio(src, SR)                     # any ffmpeg-readable format -> 16 kHz mono
    if ref is not None:
        r = load_audio(ref, SR); n = min(len(x), len(r)); x, r = x[:n], r[:n]
        y = pipe(x, ref=r)
    else:
        y = pipe(x)
    sf.write(dst, y, SR, subtype="PCM_16")
    return x, y


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("dst")
    ap.add_argument("--mode", default="clarity", choices=list(MODES))
    ap.add_argument("--no-library", action="store_true")
    ap.add_argument("--ref", default=None, help="reference (outer) mic recording -> uses MARK7-DM")
    a = ap.parse_args()
    enhance_file(load_mark7(mode=a.mode, use_library=not a.no_library, dual=a.ref is not None), a.src, a.dst, a.ref)
    print("wrote", a.dst)


# ---------------------------------------------------------------- MARK7-DM v2
class Mark7V2:
    """Best model: two mics + 16 ms lookahead + live speaker voiceprint + loudness.
        v2 = Mark7V2(ROOT); y = v2(voice_mic, reference_mic)          # loud=True -> -18 LUFS
    `enroll` (optional): a few seconds of the wearer's clean voice; otherwise the
    voiceprint is learned live from input frames the MARK7 gate finds clean."""
    def __init__(self, root=ROOT, loud=True):
        from aura.model_dm2 import Mark7DM2
        self.net = Mark7DM2(cfg)
        self.net.load_state_dict(torch.load(os.path.join(root, "weights", "mark7dm2_network.pt"), map_location="cpu")["model"])
        self.net.eval(); self.gate = load_mark7(root, use_library=False); self.loud = loud

    def __call__(self, x, ref, enroll=None):
        from aura.model_dm2 import voiceprint
        from aura.loudness import make_loud
        from aura.losses import stft, istft
        x = np.asarray(x, np.float32); ref = np.asarray(ref, np.float32)
        if enroll is not None:
            vp = voiceprint(np.asarray(enroll, np.float32))
        else:
            _, info = self.gate(x, return_info=True); vp = voiceprint(x, info["gate_w"] < 0.05)
        S = lambda s: stft(torch.from_numpy(s).unsqueeze(0), cfg)
        with torch.no_grad():
            Y = self.net(S(x), S(ref), vp=torch.from_numpy(vp).unsqueeze(0))["est"]
        y = istft(Y, cfg, length=len(x))[0].numpy()
        return make_loud(y) if self.loud else y


# ---------------------------------------------------------------- MARK7-DM v3
class Mark7V3(Mark7V2):   # (MARK7-DM v2 weights are not shipped; v3 supersedes them)
    """Best model: MARK7-DM v2 + DSRA cues from both mics; voiceprint from frames that are
    blast-free (gate) AND voice (DSRA).   v3 = Mark7V3(ROOT); y = v3(voice_mic, reference_mic)"""
    def __init__(self, root=ROOT, loud=True):
        from aura.model_dm3 import build_v3
        self.net = build_v3(cfg)
        self.net.load_state_dict(torch.load(os.path.join(root, "weights", "mark7dm3_network.pt"), map_location="cpu")["model"])
        self.net.eval(); self.gate = load_mark7(root, use_library=False); self.loud = loud

    def __call__(self, x, ref, enroll=None):
        from aura.model_dm2 import voiceprint
        from aura.model_dm3 import dsra_vec, dsra_voice_mask
        from aura.loudness import make_loud
        from aura.losses import stft, istft
        x = np.asarray(x, np.float32); ref = np.asarray(ref, np.float32)
        if enroll is not None:
            vp = voiceprint(np.asarray(enroll, np.float32))
        else:
            _, info = self.gate(x, return_info=True); gc = info["gate_w"] < 0.05; vm = dsra_voice_mask(x); k = min(len(gc), len(vm))
            vp = voiceprint(x, gc[:k] & vm[:k])
        ds = torch.from_numpy(np.concatenate([dsra_vec(x), dsra_vec(ref)], 1)).unsqueeze(0)
        S = lambda s: stft(torch.from_numpy(s).unsqueeze(0), cfg)
        with torch.no_grad():
            Y = self.net(S(x), S(ref), vp=torch.from_numpy(vp).unsqueeze(0), ds=ds)["est"]
        y = istft(Y, cfg, length=len(x))[0].numpy()
        return make_loud(y) if self.loud else y


# ---------------------------------------------------------------- MARK7-DM v4
class Mark7V4(Mark7V3):
    """MARK7-DM v3 + reference-mic residual post-filter.
    A voice-steered subband NLMS canceller on the network output estimates the residual blast
    from the reference mic; it is applied as an attenuation-only gain.
    Voice mic read raw (automatic gain AFTER cleaning); outer mic close to the voice mic.   v4 = Mark7V4(ROOT); y = v4(voice_mic, reference_mic)"""
    def __call__(self, x, ref, enroll=None):
        from aura.model_dm2 import voiceprint
        from aura.model_dm3 import dsra_vec, dsra_voice_mask
        from aura.ref_cancel import cancel_reference
        from aura.loudness import make_loud
        from aura.losses import stft, istft
        x = np.asarray(x, np.float32); ref = np.asarray(ref, np.float32)
        if enroll is not None:
            vp = voiceprint(np.asarray(enroll, np.float32))
        else:
            _, info = self.gate(x, return_info=True); gc = info["gate_w"] < 0.05; vm = dsra_voice_mask(x); k = min(len(gc), len(vm))
            vp = voiceprint(x, gc[:k] & vm[:k])
        ds = torch.from_numpy(np.concatenate([dsra_vec(x), dsra_vec(ref)], 1)).unsqueeze(0)
        S = lambda s: stft(torch.from_numpy(np.asarray(s, np.float32)).unsqueeze(0), cfg)
        with torch.no_grad():
            Y = self.net(S(x), S(ref), vp=torch.from_numpy(vp).unsqueeze(0), ds=ds)["est"][0].numpy().astype(np.complex128)
        X = S(x)[0].numpy().astype(np.complex128)
        R = S(ref)[0].numpy().astype(np.complex128)
        m = np.clip(np.abs(Y) / (np.abs(X) + 1e-9), 0, 1)
        Z, _ = cancel_reference(Y, R, p_voice=m)
        y = istft(torch.from_numpy(Z.astype(np.complex64)).unsqueeze(0), cfg, length=len(x))[0].numpy()
        return make_loud(y) if self.loud else y
