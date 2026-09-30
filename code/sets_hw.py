"""Demo audio: 4 field sets, hardware scenario (input, v3, v4, clean voice + spectrograms)."""
import shutil
exec(open("eval_hw.py").read().split("res = {}")[0].replace("split, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]", ""))
import soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
dst = "/content/aura_home/deliver/mark7dm4_sets"; shutil.rmtree(dst, ignore_errors=True); os.makedirs(dst + "/spec")
picks = [("000028", "set1_gunshot"), ("000007", "set2_gun+explosion_far"), ("000022", "set3_WORST_gun+explosion"), ("000047", "set4_gun+explosion_near")]
def spec(y, path):
    fig = plt.figure(figsize=(7, 1.6), dpi=110); ax = fig.add_axes([0, 0, 1, 1])
    ax.specgram(y + 1e-7, NFFT=512, Fs=16000, noverlap=384, vmin=-120, vmax=-15, cmap="magma"); ax.axis("off"); ax.set_ylim(0, 8000)
    fig.savefig(path + ".png"); plt.close(fig); Image.open(path + ".png").convert("RGB").save(path + ".jpg", quality=70); os.remove(path + ".png")
summary = []
for cid, name in picks:
    for root in ("/content/aura_home/corpus_hw_eval",):
        d = f"{root}/field/{cid}"
        meta = [json.loads(l) for l in open(f"{root}/field/manifest.jsonl") if json.loads(l)["id"] == cid][0]
        x, r, t = (A.read_wav(f"{d}/{k}.wav")[0].astype(np.float32) for k in ("mix", "ref", "target"))
        Y, E, Z = v4(x, r); L = len(x); z = iS(Z, L).astype(np.float32)
        outs = {"1_INPUT_voice_mic": x, "2_OUTPUT_MARK7-DM-v3": iS(Y, L), "3_OUTPUT_MARK7-DM-v4_loud": make_loud(z), "0_clean_voice": t}
        rec = {"set": name, "meta": meta, "metrics": {}}
        for k, s in outs.items():
            s = np.asarray(s, np.float32); fn = f"{name}_{k}"
            sf.write(f"{dst}/{fn}.wav", s, 16000, subtype="PCM_16"); spec(s, f"{dst}/spec/{fn}")
            q = M.evaluate_clip(s, t, x, meta); gm = np.dot(s, t) / (np.dot(s, s) + 1e-12)
            rec["metrics"][k] = dict(lm_snr=float(lm(s, t, meta)), stoi=float(q.get("stoi", np.nan)), pesq=float(q.get("pesq", np.nan)),
                                     heard=float(100 * 2 ** (cuts(x, gm * s, t)[1] / 10)) if k != "0_clean_voice" else 0.0)
        shutil.copy(f"{d}/ref.wav", f"{dst}/{name}_1b_INPUT_reference_mic.wav")
        summary.append(rec); print(name, {k: (round(v["lm_snr"], 1), round(v["stoi"], 2), round(v["pesq"], 2)) for k, v in rec["metrics"].items() if k != "0_clean_voice"}, flush=True)
json.dump(summary, open(f"{dst}/sets.json", "w"), indent=1)
