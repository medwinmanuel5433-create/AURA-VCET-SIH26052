"""Current-headset evaluation: MARK7 1-mic, MARK7-DM v2, v3 (DSRA) on the evaluation corpus."""
import sys, glob, random
from dual_common import *
from aura.model_dm2 import Mark7DM2, voiceprint
from aura.model_dm3 import build_v3, dsra_vec, dsra_voice_mask
from aura.loudness import make_loud
from aura.physmix import speaker_of
ck3 = sys.argv[1]; split = sys.argv[2]; n = int(sys.argv[3]); out = sys.argv[4]
d2 = Mark7DM2(cfg); d2.load_state_dict(torch.load("/content/aura_home/runs/m7dm2_final.pt", map_location="cpu")["model"]); d2.eval()
d3 = build_v3(cfg); d3.load_state_dict(torch.load(ck3, map_location="cpu")["model"]); d3.eval()
DEV = {}
for p in glob.glob("/content/aura_home/data/libri/dev-clean/*/*/*.flac"):
    DEV.setdefault(speaker_of(p), []).append(p)
T = lambda Z: torch.from_numpy(Z.astype(np.complex64)).unsqueeze(0)
def enroll(meta):
    c = [p for p in DEV.get(speaker_of(meta["voice_file"]), []) if os.path.basename(p) != meta["voice_file"]]
    random.seed(meta["id"]); return voiceprint(A.load_audio(random.choice(c), 16000)[:16000 * 6]) if c else None
def run2(x, r, vp):
    with torch.no_grad(): return iS(d2(T(S(x)), T(S(r)), vp=torch.from_numpy(vp).unsqueeze(0))["est"][0].numpy(), len(x))
def run3(x, r, vp):
    ds = torch.from_numpy(np.concatenate([dsra_vec(x), dsra_vec(r)], 1)).unsqueeze(0)
    with torch.no_grad(): return iS(d3(T(S(x)), T(S(r)), vp=torch.from_numpy(vp).unsqueeze(0), ds=ds)["est"][0].numpy(), len(x))
def lm_snr(y, t, meta):
    g = np.dot(y, t) / (np.dot(y, y) + 1e-12); a, b = M.event_window(meta, len(t), 16000); return M.snr_db(g * y[a:b], t[a:b])
LEVELS = {"evaluation": "/content/aura_home/corpus_dual_eval"}
res = {}
for lv, root in LEVELS.items():
    rows, cc, lm, ins = {}, {}, {}, []
    for r in [json.loads(l) for l in open(f"{root}/{split}/manifest.jsonl")][:n]:
        d = f"{root}/{split}/{r['id']}"; x, rf, t = (A.read_wav(f"{d}/{k}.wav")[0].astype(np.float32) for k in ("mix", "ref", "target"))
        _, inf = P7(x, return_info=True); gate_clean = inf["gate_w"] < 0.05
        vm = dsra_voice_mask(x); k_ = min(len(gate_clean), len(vm))
        vp2 = voiceprint(x, gate_clean); vp3 = voiceprint(x, gate_clean[:k_] & vm[:k_])
        y3 = run3(x, rf, vp3)
        outs = {"input": x, "MARK7 1-mic": P7(x), "MARK7-DM v2": run2(x, rf, vp2), "MARK7-DM v3 (DSRA)": y3}
        if split == "test":
            ve = enroll(r)
            if ve is not None: outs["MARK7-DM v3 + clean-voice enrollment"] = run3(x, rf, ve)
        outs["MARK7-DM v3 + loud (-18 LUFS)"] = make_loud(y3)
        for k, y in outs.items():
            y = np.asarray(y, np.float32); gm = np.dot(y, t) / (np.dot(y, y) + 1e-12)
            q = M.evaluate_clip(y, t, x, r); q["lm"] = lm_snr(y, t, r)
            rows.setdefault(k, []).append(q); cc.setdefault(k, []).append(cuts(x, gm * y, t))
        ins.append(r["snr_db"])
    res[lv] = {}
    print(f"\n=== {split} | current headset | {len(ins)} clips ===\n{'variant':38s} {'SNR*':>6s} {'STOI':>6s} {'PESQ':>5s} {'heard':>6s} | pass SNR STOI PESQ all3", flush=True)
    for k in rows:
        s = summarize(rows[k], cc[k]); lmv = np.array([q["lm"] for q in rows[k]]); s["lm_snr"] = float(np.nanmean(lmv))
        st = np.array([q.get("stoi", np.nan) for q in rows[k]]); pq = np.array([q.get("pesq", np.nan) for q in rows[k]])
        s["pass_snr"], s["pass_stoi"], s["pass_pesq"] = float(np.mean(lmv > 15)), float(np.mean(st > .85)), float(np.mean(pq > 2.5))
        s["pass_all"] = float(np.mean((lmv > 15) & (st > .85) & (pq > 2.5)))
        ia = np.array(ins); s["by_in_snr"] = {f"{v:.1f}": float(np.nanmean(lmv[ia == v])) for v in sorted(set(ins))}
        res[lv][k] = s
        print(f"{k:38s} {s['lm_snr']:6.2f} {s['stoi']:6.3f} {s['pesq']:5.2f} {s['heard']:5.0f}% | {100*s['pass_snr']:4.0f}% {100*s['pass_stoi']:4.0f}% {100*s['pass_pesq']:4.0f}% {100*s['pass_all']:4.0f}%", flush=True)
    json.dump(res, open(out, "w"), indent=1, default=float)
