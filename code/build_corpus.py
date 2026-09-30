"""Render the AURA-MARK6 situation corpus.

Two voice modes:

  --voice DIR
      one directory, split by file (use only when speaker identity is unknown;
      the resulting test set is NOT speaker-disjoint).

  --libri-train DIR --libri-test DIR [--val-speakers N] [--field DIR]
      LibriSpeech. Train/val come from --libri-train with val drawn from
      speakers that never appear in train; test comes from --libri-test
      (LibriSpeech dev/test sets share no speakers with train-* by design).
      --field adds a fourth split built from your own recordings, evaluated
      only, to measure transfer to the deployment domain.

Output:
    <out>/<split>/<id>/mix.wav      model input
    <out>/<split>/<id>/target.wav   achievable clean reference
    <out>/<split>/manifest.jsonl    per-example situation metadata
    <out>/source_split.json         what went where, incl. speaker overlap
"""
import argparse
import glob
import json
import os
import sys
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aura.config import DEFAULT
from aura import physmix as P
from aura import audio as A

AUDIO_EXT = (".wav", ".flac", ".m4a", ".ogg", ".opus", ".mp3")
_POOL = None
_PSEUDO = False


def _init(gun, exp, override, pseudo):
    global _POOL, _PSEUDO
    _POOL = P.SourcePool(None, gun, exp, DEFAULT, voice_split_override=override)
    _PSEUDO = pseudo


def _worker(args):
    split, idx, seed, out_dir = args
    cfg = DEFAULT
    rng = np.random.RandomState(seed)
    sit = P.sample_situation(rng, cfg)
    vps = _POOL.voice_split[split]
    vp = vps[rng.randint(len(vps))]
    try:
        ex = P.make_example(vp, _POOL.guns(split), _POOL.exp_split[split],
                            sit, cfg, rng, split=split if split in ("train", "val", "test") else "test",
                            pseudo_speaker=_PSEUDO)
    except Exception:
        return None
    eid = f"{idx:06d}"
    d = os.path.join(out_dir, split, eid)
    A.save_wav(os.path.join(d, "mix.wav"), ex["mix"], cfg.sig.sr)
    A.save_wav(os.path.join(d, "target.wav"), ex["target"], cfg.sig.sr)
    rec = dict(ex["meta"]); rec["id"] = eid; rec["split"] = split
    return rec


def files_in(d):
    return sorted(p for p in glob.glob(os.path.join(d, "**", "*"), recursive=True)
                  if os.path.isfile(p) and p.lower().endswith(AUDIO_EXT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice")
    ap.add_argument("--libri-train")
    ap.add_argument("--libri-test")
    ap.add_argument("--field")
    ap.add_argument("--val-speakers", type=int, default=25)
    ap.add_argument("--gun", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=DEFAULT.corpus.n_train)
    ap.add_argument("--val", type=int, default=DEFAULT.corpus.n_val)
    ap.add_argument("--test", type=int, default=DEFAULT.corpus.n_test)
    ap.add_argument("--field-n", type=int, default=150)
    ap.add_argument("--pseudo-speaker", action="store_true",
                    help="VTLP pseudo-speakers; only useful with very few real voices")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()

    counts = {"train": a.train, "val": a.val, "test": a.test}
    if a.libri_train:
        tr_files = files_in(a.libri_train)
        spks = sorted({P.speaker_of(p) for p in tr_files})
        rng = np.random.RandomState(DEFAULT.corpus.seed)
        val_s = set(rng.permutation(spks)[:a.val_speakers])
        override = {
            "train": [p for p in tr_files if P.speaker_of(p) not in val_s],
            "val": [p for p in tr_files if P.speaker_of(p) in val_s],
            "test": files_in(a.libri_test),
        }
        if a.field:
            override["field"] = files_in(a.field)
            counts["field"] = a.field_n
    else:
        vf = files_in(a.voice)
        rng = np.random.RandomState(DEFAULT.corpus.seed)
        vi = rng.permutation(len(vf)); n = len(vf)
        override = {"test": [vf[i] for i in vi[:max(1, int(.15 * n))]],
                    "val": [vf[i] for i in vi[int(.15 * n):int(.30 * n)]],
                    "train": [vf[i] for i in vi[int(.30 * n):]]}

    os.makedirs(a.out, exist_ok=True)
    pool = P.SourcePool(None, a.gun, a.exp, DEFAULT, voice_split_override=override)
    summary = pool.summary()
    summary["speaker_overlap_count"] = len(summary.pop("speaker_overlap"))
    summary["pseudo_speaker"] = a.pseudo_speaker
    with open(os.path.join(a.out, "source_split.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    if a.libri_train and summary["speaker_overlap_count"]:
        raise SystemExit("speaker overlap between train and val/test -- refusing to build")

    for split, n in counts.items():
        os.makedirs(os.path.join(a.out, split), exist_ok=True)
        base = DEFAULT.corpus.seed + sum(map(ord, split)) * 1000
        jobs = [(split, i, base + i * 7919, a.out) for i in range(n)]
        recs = []
        with Pool(a.workers, initializer=_init,
                  initargs=(a.gun, a.exp, override, a.pseudo_speaker)) as pp:
            for k, r in enumerate(pp.imap_unordered(_worker, jobs, chunksize=4)):
                if r is not None:
                    recs.append(r)
                if (k + 1) % 250 == 0:
                    print(f"  {split}: {k+1}/{n}", flush=True)
        recs.sort(key=lambda r: r["id"])
        with open(os.path.join(a.out, split, "manifest.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"[done] {split}: {len(recs)} examples", flush=True)


if __name__ == "__main__":
    main()
