"""Render a dual-mic corpus: val / test / field (evaluation only; the canceller needs no training)."""
import sys, os, glob, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from multiprocessing import Pool
from aura.config import DEFAULT as cfg
from aura import physmix as P, audio as A
from aura.dualmic import make_dual_example
OUT = os.environ.get("DUAL_OUT", "/content/aura_home/corpus_dual")
SNR_OFF = float(os.environ.get("SNR_OFFSET", "0"))   # extra dB added to the mixing SNR (noise attenuation) for the evaluation condition
EXT = (".wav", ".flac", ".m4a")
files = lambda d: sorted(p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True) if p.lower().endswith(EXT))
tr = files("/content/aura_home/data/libri/LibriSpeech/train-clean-100")
spks = sorted({P.speaker_of(p) for p in tr}); val_s = set(np.random.RandomState(cfg.corpus.seed).permutation(spks)[:25])
override = {"train": [p for p in tr if P.speaker_of(p) not in val_s], "val": [p for p in tr if P.speaker_of(p) in val_s],
            "test": files("/content/aura_home/data/libri/dev-clean"), "field": files("/content/aura_home/data/voice_raw/VOICE")}
# Fixed 7-file stub pool: the explosion split does not depend on which copy of train-clean-100 is present.
_STUB = "/content/aura_home/.pool_stub"; os.makedirs(_STUB, exist_ok=True)
for _i in range(7):
    open(os.path.join(_STUB, f"stub{_i}.flac"), "a").close()
if not override["train"]:
    override["train"] = override["val"] = []
pool = P.SourcePool(_STUB, "/content/aura_home/data/gun_raw", "/content/aura_home/data/explosion_raw", cfg, voice_split_override=override)
def work(a):
    split, i, seed = a
    np.random.seed(seed % (2**32))                     # audio.fit_length crops with the global RNG -> seed it per clip (deterministic reruns)
    rng = np.random.RandomState(seed); sit = P.sample_situation(rng, cfg)
    if sit["event_type"] == "none" and split != "train":                         # evaluate the canceller where there is something to cancel
        sit["event_type"] = ["gunshot", "explosion", "both"][rng.randint(3)]; sit["overlap"] = ["onset", "mid", "full"][rng.randint(3)]
    vps = pool.voice_split[split]
    sit["snr_db"] = float(sit["snr_db"]) + SNR_OFF
    if os.environ.get("CHAIN_FORCE"): sit["chain"] = os.environ["CHAIN_FORCE"]
    try:
        ex = make_dual_example(vps[rng.randint(len(vps))], pool.guns(split), pool.exp_split[split], sit, cfg, rng, split)
    except Exception as e:
        return None
    d = f"{OUT}/{split}/{i:06d}"; os.makedirs(d, exist_ok=True)
    for k in ("mix", "ref", "target"):
        A.save_wav(f"{d}/{k}.wav", ex[k], cfg.sig.sr)
    r = dict(ex["meta"]); r["id"] = f"{i:06d}"; return r
if __name__ == "__main__":
    todo = [("val", int(sys.argv[1])), ("test", int(sys.argv[2])), ("field", int(sys.argv[3]))]
    if len(sys.argv) > 4: todo = [("train", int(sys.argv[4]))]
    for split, n in todo:
        base = 777 + sum(map(ord, split)) * 1000
        with Pool(2) as pp:
            recs = [r for r in pp.map(work, [(split, i, base + i * 7919) for i in range(n)]) if r]
        os.makedirs(f"{OUT}/{split}", exist_ok=True)
        with open(f"{OUT}/{split}/manifest.jsonl", "w") as f:
            for r in sorted(recs, key=lambda r: r["id"]):
                f.write(json.dumps(r) + "\n")
        print(split, len(recs), flush=True)
