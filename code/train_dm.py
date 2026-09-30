"""Warm-start MARK7-DM from MARK6.5 on the dual-mic corpus."""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import Dataset, DataLoader
from aura.config import DEFAULT as cfg
from aura.model_dm import Mark7DM
from aura.losses import stft, mark6_loss
from aura import audio as A
torch.set_num_threads(int(os.environ.get("THREADS", "2")))
ROOT = "/content/aura_home/corpus_dual"

class DualDS(Dataset):
    def __init__(self, split, crop=None):
        self.d = f"{ROOT}/{split}"; self.recs = [json.loads(l) for l in open(f"{self.d}/manifest.jsonl")]
        self.crop = int(crop * cfg.sig.sr) if crop else None
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]; n = cfg.sig.n_samples
        w = {k: np.pad(A.read_wav(f"{self.d}/{r['id']}/{k}.wav")[0], (0, n))[:n] for k in ("mix", "ref", "target")}
        if self.crop:
            sp = r.get("event_span")
            if sp and np.random.rand() < 0.7:
                s = int(np.clip((sp[0] + sp[1]) // 2 - self.crop // 2, 0, n - self.crop))
            else:
                s = np.random.randint(0, n - self.crop + 1)
            w = {k: v[s:s + self.crop] for k, v in w.items()}
            r = dict(r); 
            if sp: r["event_span"] = [max(0, sp[0] - s), max(0, min(self.crop, sp[1] - s))] if sp[1] > s and sp[0] < s + self.crop else None
        return {k: torch.from_numpy(v.astype(np.float32)) for k, v in w.items()} | {"meta": r}

def collate(b):
    out = {k: torch.stack([x[k] for x in b]) for k in ("mix", "ref", "target")}
    out["meta"] = [x["meta"] for x in b]; out["spk"] = torch.zeros(len(b), cfg.model.spk_dim)
    out["X"] = stft(out["mix"], cfg); out["R"] = stft(out["ref"], cfg); out["T"] = stft(out["target"], cfg)
    return out

if __name__ == "__main__":
    minutes, out = float(sys.argv[1]), sys.argv[2]
    model = Mark7DM(cfg)
    miss = model.load_state_dict(torch.load("/content/aura_home/runs/m65/best.pt", map_location="cpu")["model"], strict=False)
    print("new params:", len(miss.missing_keys), flush=True)
    new = [p for n, p in model.named_parameters() if n.startswith("ref_")]
    old = [p for n, p in model.named_parameters() if not n.startswith("ref_")]
    opt = torch.optim.AdamW([{"params": new, "lr": 1e-3}, {"params": old, "lr": 1e-4}], weight_decay=1e-4)
    dl = DataLoader(DualDS("train", crop=3.0), batch_size=4, shuffle=True, collate_fn=collate, drop_last=True)
    t0 = time.time(); step = 0; model.train(); done = False
    while not done:
        for b in dl:
            o = model(b["X"], b["R"], b["spk"])
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
