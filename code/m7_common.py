import sys, os, glob, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(1)
from aura.config import DEFAULT as cfg
from aura.model import Mark6
from aura.data import CorpusDataset
from aura.mark7 import Mark7
from aura.losses import stft, istft
from aura import metrics as M, audio as A

m = Mark6(cfg); m.load_state_dict(torch.load("/content/aura_home/runs/m65/best.pt", map_location="cpu")["model"]); m.eval()
D = torch.load("/content/aura_home/runs/dicts_m65.pt")

def mark65(x):
    with torch.no_grad():
        return istft(m(stft(torch.from_numpy(x).unsqueeze(0), cfg))["est"], cfg, length=len(x))[0].numpy()

def cuts(mix, y, tgt):
    X = stft(torch.from_numpy(mix).unsqueeze(0), cfg)[0].numpy()
    Y = stft(torch.from_numpy(np.asarray(y, np.float32)).unsqueeze(0), cfg)[0].numpy()
    T = stft(torch.from_numpy(tgt).unsqueeze(0), cfg)[0].numpy()
    tp, npw = np.abs(T) ** 2, np.abs(X - T) ** 2
    vd = (tp > npw) & (tp > tp.max() * 1e-5); nd = npw > 9 * tp
    vc = 10 * np.log10(np.sum(np.abs(Y[vd]) ** 2) / (np.sum(tp[vd]) + 1e-12) + 1e-12) if vd.any() else np.nan
    nc = 10 * np.log10(np.sum(np.abs(Y[nd]) ** 2) / (np.sum(np.abs(X[nd]) ** 2) + 1e-12) + 1e-12) if nd.any() else np.nan
    return vc, nc

g = lambda d, k: d.get(k, {}).get("mean", float("nan"))

def summarize(rows, cc):
    a = M.aggregate_split(rows); e, cl = a["event"], a["clean"]
    c = np.nanmean(np.array(cc), 0) if cc else [np.nan, np.nan]
    return dict(out_snr=g(e, "snr_evt"), dsnr=g(e, "delta_snr_evt"), imp=g(e, "delta_snr_imp"),
                stoi=g(e, "stoi"), pesq=g(e, "pesq"), gap=g(e, "pause_noise_db"),
                voice=float(c[0]), noise=float(c[1]), heard=float(100 * 2 ** (c[1] / 10)),
                cl_snr=g(cl, "snr_evt"), cl_stoi=g(cl, "stoi"), cl_pesq=g(cl, "pesq"))

HDR = f"{'variant':34s} {'outSNR':>6s} {'dSNR':>6s} {'imp':>6s} {'STOI':>6s} {'PESQ':>5s} {'gap':>6s} {'voice':>6s} {'heard':>5s} | {'clSNR':>6s} {'clSTOI':>6s} {'clPESQ':>6s}"
def line(k, o):
    return (f"{k:34s} {o['out_snr']:6.2f} {o['dsnr']:6.2f} {o['imp']:6.2f} {o['stoi']:6.3f} {o['pesq']:5.2f} {o['gap']:6.1f} "
            f"{o['voice']:+6.1f} {o['heard']:4.0f}% | {o['cl_snr']:6.1f} {o['cl_stoi']:6.3f} {o['cl_pesq']:6.2f}")

def run_split(split, n, variants, verbose=True, start=0):
    ds = CorpusDataset("/content/aura_home/corpus6", split, cfg)
    rows = {k: [] for k in variants}; cc = {k: [] for k in variants}; tm = {k: 0.0 for k in variants}
    for i in range(start, start + n):
        it = ds[i]; mix, tgt = it["mix"].numpy(), it["target"].numpy()
        for k, fn in variants.items():
            t0 = time.time(); y = fn(mix, it); tm[k] += time.time() - t0
            rows[k].append(M.evaluate_clip(np.asarray(y, np.float32), tgt, mix, it["meta"]))
            if it["meta"].get("event_span"):
                cc[k].append(cuts(mix, y, tgt))
    out = {}
    if verbose: print(HDR)
    for k in variants:
        out[k] = summarize(rows[k], cc[k]); out[k]["sec_per_clip"] = tm[k] / n
        if verbose: print(line(k, out[k]) + f"  {tm[k]/n:.2f}s", flush=True)
    return out
