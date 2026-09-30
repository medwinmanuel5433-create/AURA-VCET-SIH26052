from m7_common import *
from aura.canceller import subband_nlms
sys.path.insert(0, "/content/aura_home/deliver/AURA_MARK7")
from run_mark7 import load_mark7
P7 = load_mark7("/content/aura_home/deliver/AURA_MARK7")

def load_dual(split):
    root = f"/content/aura_home/corpus_dual/{split}"
    for l in open(f"{root}/manifest.jsonl"):
        r = json.loads(l); d = f"{root}/{r['id']}"
        yield {k: A.read_wav(f"{d}/{k}.wav")[0].astype(np.float32) for k in ("mix", "ref", "target")} | {"meta": r}

def S(x):
    return stft(torch.from_numpy(np.asarray(x, np.float32)).unsqueeze(0), cfg)[0].numpy().astype(np.complex128)
def iS(Z, n):
    return istft(torch.from_numpy(Z.astype(np.complex64)).unsqueeze(0), cfg, length=n)[0].numpy()

def p_voice(X):
    with torch.no_grad():
        Y = m(torch.from_numpy(X.astype(np.complex64)).unsqueeze(0))["est"][0].numpy()
    return np.clip(np.abs(Y) ** 2 / (np.abs(X) ** 2 + 1e-10), 0, 1)

def cancel(mix, ref, steer=True, **kw):
    X, R = S(mix), S(ref)
    E, _ = subband_nlms(X, R, steer=p_voice(X) if steer else None, **kw)
    return iS(E, len(mix))

def run_dual(split, n, V, start=0, verbose=True):
    rows = {k: [] for k in V}; cc = {k: [] for k in V}; tm = {k: 0.0 for k in V}
    for j, it in enumerate(load_dual(split)):
        if j < start: continue
        if j >= start + n: break
        mix, ref, tgt = it["mix"], it["ref"], it["target"]
        for k, fn in V.items():
            t0 = time.time(); y = np.asarray(fn(mix, ref), np.float32); tm[k] += time.time() - t0
            rows[k].append(M.evaluate_clip(y, tgt, mix, it["meta"])); cc[k].append(cuts(mix, y, tgt))
    out = {}
    if verbose: print(HDR)
    for k in V:
        out[k] = summarize(rows[k], cc[k]); out[k]["sec_per_clip"] = tm[k] / n
        if verbose: print(line(k, out[k]) + f"  {tm[k]/n:.2f}s", flush=True)
    return out

def cancel_blend(mix, ref, beta=1.0, kw=None, pv=None):
    """AI-steered adaptation AND AI-steered output: in voice-dominant bins keep the
    primary mic (no voice damage), in noise-dominated bins take the cancelled
    signal, and never let the cancellation ADD energy to a bin."""
    kw = kw or {}
    X, R = S(mix), S(ref)
    pv = p_voice(X) if pv is None else pv
    E, _ = subband_nlms(X, R, steer=pv, **kw)
    Em = np.where(np.abs(E) < np.abs(X), E, X)
    a = pv ** beta
    return iS(a * X + (1 - a) * Em, len(mix))

def net_Y(x):
    X = S(x)
    with torch.no_grad():
        return m(torch.from_numpy(X.astype(np.complex64)).unsqueeze(0))["est"][0].numpy().astype(np.complex128)

def dm_fuse(mix, ref, mode="min", beta=0.5, kw=None):
    """Two network estimates -- from the primary mic and from the cancelled
    signal -- fused per bin: the one with less energy wins (each bin's extra
    energy is residual noise, since both estimates target the same voice)."""
    c = cancel_blend(mix, ref, beta, kw or dict(gamma=3.0))
    Yp, Yc = net_Y(mix), net_Y(c)
    if mode == "min":
        Z = np.where(np.abs(Yc) < np.abs(Yp), Yc, Yp)
    else:
        Z = 0.5 * (Yp + Yc)
    return iS(Z, len(mix))
