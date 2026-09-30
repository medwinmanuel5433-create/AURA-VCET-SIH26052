"""Warm-start MARK7-DM v3 (DSRA conditioning) from v2."""
import sys, os, random, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torch.utils.data import DataLoader
import train_dm2 as T2
from train_dm2 import cfg, mark6_loss, A, voiceprint, enroll_other
from aura.model_dm3 import build_v3, dsra_vec, dsra_voice_mask

def live_vp3(mix, span, n):
    m = np.ones(1 + (n - 512) // 128, bool)
    if span:
        a, b = span[0] // 128 - 4, span[1] // 128 + 4; m[max(0, a):max(0, b)] = False
    v = dsra_voice_mask(mix); k = min(len(m), len(v)); m = m[:k] & v[:k]
    return voiceprint(mix, m)

class DS3(T2.DS2):
    def __getitem__(self, i):
        r = self.recs[i]; n = cfg.sig.n_samples; u = random.random()
        if u < 0.45: vp = enroll_other(r["voice_file"])
        elif u < 0.9:
            mix = np.pad(A.read_wav(f"{self.d}/{r['id']}/mix.wav")[0], (0, n))[:n]; vp = live_vp3(mix, r.get("event_span"), n)
        else: vp = np.zeros(80, np.float32)
        it = T2.T0.DualDS.__getitem__(self, i); it["vp"] = torch.from_numpy(vp)
        it["ds"] = torch.from_numpy(np.concatenate([dsra_vec(it["mix"].numpy()), dsra_vec(it["ref"].numpy())], 1))
        return it

def collate(b):
    out = T2.collate(b); out["ds"] = torch.stack([x["ds"] for x in b]); return out

if __name__ == "__main__":
    minutes, out = float(sys.argv[1]), sys.argv[2]
    torch.set_num_threads(2)
    model = build_v3(cfg)
    sd = torch.load("/content/aura_home/runs/m7dm2_final.pt", map_location="cpu")["model"]
    model.load_state_dict({("cond_mix.base." + k[9:] if k.startswith("cond_mix.") else k): v for k, v in sd.items()}, strict=False)
    new = [p for n, p in model.named_parameters() if n.startswith("ds_")]
    old = [p for n, p in model.named_parameters() if not n.startswith("ds_")]
    opt = torch.optim.AdamW([{"params": new, "lr": 1e-3}, {"params": old, "lr": 5e-5}], weight_decay=1e-4)
    dl = DataLoader(DS3("train", crop=3.0), batch_size=4, shuffle=True, collate_fn=collate, drop_last=True)
    t0 = time.time(); step = 0; model.train(); done = False
    while not done:
        for b in dl:
            o = model(b["X"], b["R"], b["spk"], vp=b["vp"], ds=b["ds"])
            loss, terms = mark6_loss(o, b, cfg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step(); step += 1
            if step % 25 == 0:
                print(f"step {step} loss {terms['total']:.3f} {(time.time()-t0)/60:.1f} min", flush=True)
                torch.save({"model": model.state_dict(), "step": step}, out)
            if (time.time() - t0) / 60 > minutes: done = True; break
    torch.save({"model": model.state_dict(), "step": step}, out); print("saved", out, step, flush=True)
