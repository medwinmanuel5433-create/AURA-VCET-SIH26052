"""Warm-start MARK7-DM v2 (lookahead + voiceprint) from MARK7-DM on the dual-mic corpus."""
import sys, os, json, glob, time, random, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
from aura.config import DEFAULT as cfg
from aura.model_dm2 import Mark7DM2, voiceprint
from aura.losses import mark6_loss
from aura import audio as A
from aura.physmix import speaker_of
import train_dm as T0
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
LIB = "/content/aura_home/data/libri/LibriSpeech/train-clean-100"
BY_SPK = {}
for p in glob.glob(f"{LIB}/*/*/*.flac"):
    BY_SPK.setdefault(speaker_of(p), []).append(p)

def enroll_other(vf):
    s = speaker_of(vf); c = [p for p in BY_SPK.get(s, []) if os.path.basename(p) != vf]
    if not c: return np.zeros(80, np.float32)
    return voiceprint(A.load_audio(random.choice(c), 16000)[: 16000 * 6])

def live_vp(mix, span, n):
    m = np.ones(1 + (n - 512) // 128, bool)
    if span:
        a, b = span[0] // 128 - 4, span[1] // 128 + 4; m[max(0, a):max(0, b)] = False
    return voiceprint(mix, m)

class DS2(T0.DualDS):
    def __getitem__(self, i):
        r = self.recs[i]; n = cfg.sig.n_samples
        u = random.random()
        if u < 0.45:
            vp = enroll_other(r["voice_file"])
        elif u < 0.9:
            mix = np.pad(A.read_wav(f"{self.d}/{r['id']}/mix.wav")[0], (0, n))[:n]
            vp = live_vp(mix, r.get("event_span"), n)
        else:
            vp = np.zeros(80, np.float32)
        it = super().__getitem__(i); it["vp"] = torch.from_numpy(vp); return it

def collate(b):
    out = T0.collate(b); out["vp"] = torch.stack([x["vp"] for x in b]); return out

if __name__ == "__main__":
    minutes, out = float(sys.argv[1]), sys.argv[2]
    model = Mark7DM2(cfg)
    model.load_state_dict(torch.load("/content/aura_home/runs/m7dm_final.pt", map_location="cpu")["model"], strict=False)
    new = [p for n, p in model.named_parameters() if n.startswith(("la_", "vp_"))]
    old = [p for n, p in model.named_parameters() if not n.startswith(("la_", "vp_"))]
    opt = torch.optim.AdamW([{"params": new, "lr": 1e-3}, {"params": old, "lr": 7e-5}], weight_decay=1e-4)
    dl = DataLoader(DS2("train", crop=3.0), batch_size=4, shuffle=True, collate_fn=collate, drop_last=True)
    t0 = time.time(); step = 0; model.train(); done = False
    while not done:
        for b in dl:
            o = model(b["X"], b["R"], b["spk"], vp=b["vp"])
            loss, terms = mark6_loss(o, b, cfg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            step += 1
            if step % 25 == 0:
                print(f"step {step} loss {terms['total']:.3f} {(time.time()-t0)/60:.1f} min", flush=True)
                torch.save({"model": model.state_dict(), "step": step}, out)
            if (time.time() - t0) / 60 > minutes:
                done = True; break
    torch.save({"model": model.state_dict(), "step": step}, out)
    print("saved", out, step, flush=True)
