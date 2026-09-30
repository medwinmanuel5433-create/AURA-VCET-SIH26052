"""Pick 5 PASS and 5 FAIL cases (SNR > 15 dB, STOI > 0.85, PESQ > 2.5), every case a different signal.
Metrics are computed here (nothing typed in). A case passes if MARK7-DM v3 or v4_loud passes all three;
a case fails only if BOTH fail. Different signal = different clip id AND different voice recording
AND different noise-source combination."""
import shutil, zipfile
exec(open("eval_hw.py").read().split("res = {}")[0].replace("split, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]", ""))
import soundfile as sf
N_EACH = int(os.environ.get("N_EACH", "5"))
OUT = os.environ.get("CASES_OUT", "/content/AURA_PASS_FAIL_CASES")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT + "/PASS"); os.makedirs(OUT + "/FAIL")
OK = lambda m: m["snr"] > 15 and m["stoi"] > 0.85 and m["pesq"] > 2.5

def score(s, t, x, meta):
    q = M.evaluate_clip(s, t, x, meta)
    return dict(snr=float(lm(s, t, meta)), stoi=float(q.get("stoi", np.nan)), pesq=float(q.get("pesq", np.nan)))

ROOT = "/content/aura_home/corpus_hw_eval"
cands = []
for split in ("field", "test"):
    if os.path.isfile(f"{ROOT}/{split}/manifest.jsonl"):
        cands += [(split, json.loads(l)) for l in open(f"{ROOT}/{split}/manifest.jsonl")]
used_id, used_voice, used_noise = set(), set(), set()
picked = {"PASS": [], "FAIL": []}
t0 = time.time()
for split, meta in cands:
    if all(len(v) >= N_EACH for v in picked.values()):
        break
    key = (split, meta["id"]); nz = tuple(sorted(meta["event_sources"]))
    if key in used_id or meta["voice_file"] in used_voice or nz in used_noise:
        continue
    d = f"{ROOT}/{split}/{meta['id']}"
    x, r, t = (A.read_wav(f"{d}/{k}.wav")[0].astype(np.float32) for k in ("mix", "ref", "target"))
    Y, E, Z = v4(x, r); L = len(x)
    outs = {"MARK7-DM-v4_loud": np.asarray(make_loud(iS(Z, L).astype(np.float32)), np.float32),
            "MARK7-DM-v3": np.asarray(iS(Y, L), np.float32)}
    sc = {k: score(s, t, x, meta) for k, s in outs.items()}
    good = [k for k in outs if OK(sc[k])]
    grp = "PASS" if good else "FAIL"
    if len(picked[grp]) >= N_EACH:
        continue
    name = good[0] if good else "MARK7-DM-v4_loud"
    i = len(picked[grp]) + 1
    tag = f"{grp.lower()}{i}_{meta['event_type']}_{meta['distance']}"
    sf.write(f"{OUT}/{grp}/{tag}_1_INPUT_voice_mic.wav", x, 16000, subtype="PCM_16")
    sf.write(f"{OUT}/{grp}/{tag}_2_OUTPUT_{name}.wav", outs[name], 16000, subtype="PCM_16")
    picked[grp].append(dict(case=f"{grp} {i}", tag=tag, split=split, id=meta["id"], voice=meta["voice_file"],
                            noise=list(nz), event=meta["event_type"], distance=meta["distance"], output=name,
                            metrics=sc[name], other={k: v for k, v in sc.items() if k != name},
                            input=f"{grp}/{tag}_1_INPUT_voice_mic.wav", out_file=f"{grp}/{tag}_2_OUTPUT_{name}.wav"))
    used_id.add(key); used_voice.add(meta["voice_file"]); used_noise.add(nz)
    m = sc[name]; print(f"{grp} {i}: {split}/{meta['id']}  SNR {m['snr']:.2f}  STOI {m['stoi']:.3f}  PESQ {m['pesq']:.2f}  [{name}]  ({time.time()-t0:.0f}s)", flush=True)
json.dump(picked, open(f"{OUT}/cases.json", "w"), indent=1)
with zipfile.ZipFile(OUT + ".zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root_, _, fs in os.walk(OUT):
        for f in fs:
            z.write(os.path.join(root_, f), os.path.relpath(os.path.join(root_, f), os.path.dirname(OUT)))
print("found", {k: len(v) for k, v in picked.items()}, "->", OUT + ".zip")
