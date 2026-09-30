"""Hardware scenario (raw voice mic: AGC after cleaning; close reference mic, coherence 0.97-0.995).
MARK7-DM v3 vs v4 (v3 + reference-mic post-filter) on the evaluation corpus."""
import sys
from dual_common import *
from aura.model_dm2 import voiceprint
from aura.model_dm3 import build_v3, dsra_vec, dsra_voice_mask
from aura.ref_cancel import cancel_reference
from aura.loudness import make_loud
split, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d3 = build_v3(cfg); d3.load_state_dict(torch.load("/content/aura_home/runs/m7dm3_final.pt", map_location="cpu")["model"]); d3.eval()
Tt = lambda Z: torch.from_numpy(Z.astype(np.complex64)).unsqueeze(0)
def v3(x, r, vp):
    ds = torch.from_numpy(np.concatenate([dsra_vec(x), dsra_vec(r)], 1)).unsqueeze(0)
    with torch.no_grad(): return d3(Tt(S(x)), Tt(S(r)), vp=torch.from_numpy(vp).unsqueeze(0), ds=ds)["est"][0].numpy().astype(np.complex128)
def lm(y, t, meta):
    g = np.dot(y, t) / (np.dot(y, y) + 1e-12); a, b = M.event_window(meta, len(t), 16000); return M.snr_db(g * y[a:b], t[a:b])
def v4(x, r):
    _, inf = P7(x, return_info=True); gc = inf["gate_w"] < 0.05; vm = dsra_voice_mask(x); k_ = min(len(gc), len(vm)); vp = voiceprint(x, gc[:k_] & vm[:k_])
    X = S(x); R = S(r); Y = v3(x, r, vp); m = np.clip(np.abs(Y) / (np.abs(X) + 1e-9), 0, 1)
    Z, E = cancel_reference(Y, R, p_voice=m)
    return Y, E, Z
res = {}
for root in ("/content/aura_home/corpus_hw_eval",):
    rows, cc, lmv, ins = {}, {}, {}, []
    for r_ in [json.loads(l) for l in open(f"{root}/{split}/manifest.jsonl")][:n]:
        d = f"{root}/{split}/{r_['id']}"; x, r, t = (A.read_wav(f"{d}/{k}.wav")[0].astype(np.float32) for k in ("mix", "ref", "target"))
        Y, E, Z = v4(x, r); L = len(x); z = iS(Z, L).astype(np.float32)
        outs = {"input": x, "MARK7-DM v3": iS(Y, L), "residual canceller": iS(E, L), "MARK7-DM v4 (v3 + reference post-filter)": z, "MARK7-DM v4 + loud (-18 LUFS)": make_loud(z)}
        for k, y in outs.items():
            y = np.asarray(y, np.float32); gm = np.dot(y, t) / (np.dot(y, y) + 1e-12)
            rows.setdefault(k, []).append(M.evaluate_clip(y, t, x, r_)); cc.setdefault(k, []).append(cuts(x, gm * y, t)); lmv.setdefault(k, []).append(lm(y, t, r_))
        ins.append(r_["snr_db"])
    key = "evaluation"; res[key] = {}
    print(f"\n=== {split} | HARDWARE scenario | {len(ins)} clips ===\n{'variant':36s} {'SNR*':>6s} {'STOI':>6s} {'PESQ':>5s} {'heard':>6s} | pass SNR STOI PESQ all3", flush=True)
    for k in rows:
        s = summarize(rows[k], cc[k]); lv_ = np.array(lmv[k]); s["lm_snr"] = float(np.nanmean(lv_))
        st = np.array([q.get("stoi", np.nan) for q in rows[k]]); pq = np.array([q.get("pesq", np.nan) for q in rows[k]])
        s["pass_snr"], s["pass_stoi"], s["pass_pesq"] = float(np.mean(lv_ > 15)), float(np.mean(st > .85)), float(np.mean(pq > 2.5))
        s["pass_all"] = float(np.mean((lv_ > 15) & (st > .85) & (pq > 2.5)))
        ia = np.array(ins); s["by_in_snr"] = {f"{v:.1f}": float(np.nanmean(lv_[ia == v])) for v in sorted(set(ins))}
        res[key][k] = s
        print(f"{k:36s} {s['lm_snr']:6.2f} {s['stoi']:6.3f} {s['pesq']:5.2f} {s['heard']:5.0f}% | {100*s['pass_snr']:4.0f}% {100*s['pass_stoi']:4.0f}% {100*s['pass_pesq']:4.0f}% {100*s['pass_all']:4.0f}%", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
