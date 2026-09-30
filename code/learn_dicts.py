"""Learn voice and residual-noise dictionaries on corpus6 TRAIN clips only."""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(2)
from aura.config import DEFAULT as cfg
from aura.model import Mark6
from aura.losses import stft, istft
from aura.data import CorpusDataset, collate
from aura.dict_refine import kl_nmf

m = Mark6(cfg); m.load_state_dict(torch.load("/content/aura_home/runs/m65/best.pt", map_location="cpu")["model"]); m.eval()
ds = CorpusDataset("/content/aura_home/corpus6", "train", cfg)
rng = np.random.RandomState(0)
idx = rng.permutation(len(ds))[:240]
V_voice, V_noise = [], []
for s in range(0, len(idx), 8):
    b = collate([ds[i] for i in idx[s:s + 8]], cfg)
    with torch.no_grad():
        Y = m(b["X"], b["spk"])["est"]
    T = b["T"]
    raw_noise = (b["X"] - T).abs()            # what the mic adds
    left = (Y - T).abs()                      # what MARK6.5 leaves behind
    for j in range(len(T)):
        tm = T[j].abs()
        act = tm.sum(1) > tm.sum(1).max() * 1e-3
        V_voice.append(tm[act])
        for src in (raw_noise[j], left[j]):
            e = src.sum(1); keep = e > e.max() * 1e-3
            V_noise.append(src[keep])
Vv = torch.cat(V_voice); Vn = torch.cat(V_noise)
sel_v = torch.from_numpy(rng.choice(len(Vv), min(40000, len(Vv)), replace=False))
sel_n = torch.from_numpy(rng.choice(len(Vn), min(40000, len(Vn)), replace=False))
print("frames: voice", len(Vv), "noise", len(Vn))
W_v = kl_nmf(Vv[sel_v].T.contiguous(), 64, iters=200, seed=1)
W_r = kl_nmf(Vn[sel_n].T.contiguous(), 48, iters=200, seed=2)
torch.save({"W_v": W_v, "W_r": W_r}, "/content/aura_home/runs/dicts_m65.pt")
print("saved dictionaries", tuple(W_v.shape), tuple(W_r.shape))
