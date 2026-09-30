"""AURA-MARK6 training and evaluation.

Usage:
    python train.py --corpus DIR --out DIR [--epochs N --batch N --limit N]
    python train.py --corpus DIR --out DIR --eval-only --ckpt PATH

Checkpoints, per-epoch metrics and the config are written to --out so a run is
reproducible and resumable. Resume state includes every key it checks for --
the previous pipeline wrote a state file that its own loader rejected, so every
restart silently began from scratch.
"""
import argparse
import json
import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aura.config import DEFAULT, Cfg
from aura.model import Mark6, assert_causal, count_params
from aura.losses import mark6_loss, istft, stft
from aura.data import CorpusDataset, collate, RefPool
from aura import metrics as M
from aura import audio as A

EXPERIMENT_VERSION = "mark6.5"


def make_loaders(corpus, cfg, batch, limit=None, refpool=None, workers=0,
                 crop_seconds=None):
    out = {}
    for split in ("train", "val", "test", "field"):
        p = os.path.join(corpus, split, "manifest.jsonl")
        if not os.path.exists(p):
            continue
        # crops only ever apply to training; val/test run at full length
        ds = CorpusDataset(corpus, split, cfg, refpool,
                           crop_seconds=crop_seconds if split == "train" else None)
        if limit:
            ds.records = ds.records[:limit]
        out[split] = DataLoader(
            ds, batch_size=batch, shuffle=(split == "train"),
            num_workers=workers, collate_fn=lambda b: collate(b, cfg),
            drop_last=(split == "train"))
    return out


@torch.no_grad()
def evaluate(model, loader, cfg, max_batches=None, save_dir=None):
    model.eval()
    rows, saved = [], 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        out = model(batch["X"], batch["spk"])
        y = istft(out["est"], cfg, length=batch["mix"].shape[-1]).cpu().numpy()
        mix = batch["mix"].cpu().numpy()
        tgt = batch["target"].cpu().numpy()
        for i in range(len(y)):
            r = M.evaluate_clip(y[i], tgt[i], mix[i], batch["meta"][i], cfg.sig.sr)
            r["_meta"] = batch["meta"][i]
            rows.append(r)
            if save_dir and saved < 6:
                d = os.path.join(save_dir, batch["id"][i])
                A.save_wav(os.path.join(d, "mix.wav"), mix[i], cfg.sig.sr)
                A.save_wav(os.path.join(d, "target.wav"), tgt[i], cfg.sig.sr)
                A.save_wav(os.path.join(d, "enhanced.wav"), y[i], cfg.sig.sr)
                saved += 1
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    return M.aggregate_split(clean), rows


def breakdown(rows, key, event_only=True):
    """Report delta SNR grouped by a situation axis, so a good average cannot
    hide a whole condition being broken."""
    groups = {}
    if event_only:
        rows = [r for r in rows if r.get("has_event", 0.0) >= 0.5]
    for r in rows:
        g = r["_meta"].get(key)
        g = "none" if g is None else str(g)
        groups.setdefault(g, []).append((r.get("delta_snr_evt", np.nan),
                                         r.get("delta_snr_imp", np.nan)))
    out = {}
    for g, vals in sorted(groups.items()):
        v = np.array([x[0] for x in vals], dtype=float)
        u = np.array([x[1] for x in vals], dtype=float)
        f, fu = np.isfinite(v), np.isfinite(u)
        out[g] = {"n": int(len(v)),
                  "delta_snr_evt": round(float(v[f].mean()), 3) if f.any() else None,
                  "delta_snr_imp": round(float(u[fu].mean()), 3) if fu.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--voice-dir", default=None,
                    help="raw voice dir, for building the reference pool")
    ap.add_argument("--epochs", type=int, default=DEFAULT.train.epochs)
    ap.add_argument("--batch", type=int, default=DEFAULT.train.batch_size)
    ap.add_argument("--lr", type=float, default=DEFAULT.train.lr)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--eval-batches", type=int, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop cleanly after this wall-clock budget")
    ap.add_argument("--crop", type=float, default=None,
                    help="train on random crops of this many seconds")
    a = ap.parse_args()

    torch.set_num_threads(a.threads)
    cfg = DEFAULT
    cfg.train.lr = a.lr
    os.makedirs(a.out, exist_ok=True)
    cfg.dump(os.path.join(a.out, "config.json"))

    refpool = None
    if a.voice_dir:
        import glob
        vf = [p for p in glob.glob(os.path.join(a.voice_dir, "**", "*"), recursive=True)
              if os.path.isfile(p)]
        refpool = RefPool(vf, cfg)
        print(f"[refpool] {len(refpool.raw)} voice references", flush=True)

    loaders = make_loaders(a.corpus, cfg, a.batch, a.limit, refpool,
                           crop_seconds=a.crop)
    print(f"[data] " + ", ".join(f"{k}={len(v.dataset)}" for k, v in loaders.items()),
          flush=True)

    model = Mark6(cfg)
    print(f"[model] {count_params(model)/1e6:.3f} M params", flush=True)
    ok, leak = assert_causal(model)
    print(f"[causality] {'PASS' if ok else 'FAIL'} (max leak {leak:.2e})", flush=True)
    if not ok:
        print("ABORT: model is not causal", flush=True)
        return

    if a.ckpt and os.path.exists(a.ckpt):
        sd = torch.load(a.ckpt, map_location="cpu")
        model.load_state_dict(sd["model"])
        print(f"[resume] loaded {a.ckpt} (epoch {sd.get('epoch')}, "
              f"version {sd.get('experiment_version')})", flush=True)

    if a.eval_only:
        for split in ("test", "field"):
            if split in loaders:
                agg, rows = evaluate(model, loaders[split], cfg, a.eval_batches,
                                     save_dir=os.path.join(a.out, f"samples_{split}"))
                report(agg, rows, a.out, split)
        return

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                            weight_decay=cfg.train.weight_decay)
    # Per-step linear warmup then cosine. The old per-epoch cosine over a
    # 14-epoch horizon kept the LR near its peak for most of a short run and
    # never annealed; the audit showed the model under-fitting its own
    # training set, which calls for a schedule that actually converges.
    steps_total = max(1, a.epochs * len(loaders["train"]))
    warm = min(300, steps_total // 10)
    import math
    def lr_lambda(step):
        if step < warm:
            return (step + 1) / warm
        p_ = (step - warm) / max(1, steps_total - warm)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(p_, 1.0)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    hist = []
    best = -1e9
    t_start = time.time()
    stop = False

    for ep in range(1, a.epochs + 1):
        model.train()
        agg_terms, nb = {}, 0
        t0 = time.time()
        for bi, batch in enumerate(loaders["train"]):
            out = model(batch["X"], batch["spk"])
            loss, terms = mark6_loss(out, batch, cfg)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            sched.step()
            for k, v in terms.items():
                agg_terms[k] = agg_terms.get(k, 0.0) + v
            nb += 1
            if (bi + 1) % 25 == 0:
                print(f"  ep{ep} step {bi+1}/{len(loaders['train'])} "
                      f"loss={terms['total']:.4f} rec={terms['rec_frac']:.3f}",
                      flush=True)
            if a.max_minutes and (time.time() - t_start) / 60 > a.max_minutes:
                print("[budget] wall-clock budget reached, stopping", flush=True)
                stop = True
                break
        tr = {k: v / max(nb, 1) for k, v in agg_terms.items()}

        vagg, vrows = evaluate(model, loaders["val"], cfg, a.eval_batches)
        ev = vagg.get("event", {}); cl = vagg.get("clean", {})
        dsnr = ev.get("delta_snr_evt", {}).get("mean", float("nan"))
        dimp = ev.get("delta_snr_imp", {}).get("mean", float("nan"))
        vpesq = ev.get("pesq", {}).get("mean", float("nan"))
        vpause = ev.get("pause_noise_db", {}).get("mean", float("nan"))
        stoi = ev.get("stoi", {}).get("mean", float("nan"))
        csnr = cl.get("snr_evt", {}).get("mean", float("nan"))
        print(f"[ep {ep}] train_loss={tr.get('total', float('nan')):.4f} "
              f"| EVENT dsnr={dsnr:+.2f} imp={dimp:+.2f} dB stoi={stoi:.3f} (n={vagg.get('n_event')})"
              f" pesq={vpesq:.3f} pause={vpause:.1f}dB"
              f" | CLEAN passthru={csnr:.1f} dB (n={vagg.get('n_clean')})"
              f" | {time.time()-t0:.0f}s", flush=True)

        hist.append({"epoch": ep, "train": tr, "val": vagg})
        with open(os.path.join(a.out, "history.json"), "w") as f:
            json.dump(hist, f, indent=2)

        # Select on the event window AND the impulse window, and refuse to
        # reward a checkpoint that buys event gains by damaging clean audio.
        # Selection covers every complaint that has a number: SNR in the event
        # and impulse windows, perceived quality (PESQ), residual noise in
        # speech pauses (lower is better), and damage to clean audio.
        score = (dsnr if np.isfinite(dsnr) else -50) + (dimp if np.isfinite(dimp) else -50)
        score += 4.0 * (vpesq if np.isfinite(vpesq) else 0.0)
        score += 0.2 * (-vpause if np.isfinite(vpause) else 0.0)
        if np.isfinite(csnr) and csnr < 25.0:
            score -= (25.0 - csnr)
        state = {"model": model.state_dict(), "opt": opt.state_dict(),
                 "epoch": ep, "experiment_version": EXPERIMENT_VERSION,
                 "score": score, "cfg": os.path.join(a.out, "config.json")}
        torch.save(state, os.path.join(a.out, "last.pt"))
        if score > best:
            best = score
            torch.save(state, os.path.join(a.out, "best.pt"))
            print(f"  new best ({best:.2f} dB) -> best.pt", flush=True)
        if stop:
            break

    # final test pass with the best checkpoint
    bp = os.path.join(a.out, "best.pt")
    if os.path.exists(bp):
        model.load_state_dict(torch.load(bp, map_location="cpu")["model"])
    for split in ("test", "field"):
        if split in loaders:
            agg, rows = evaluate(model, loaders[split], cfg, None,
                                 save_dir=os.path.join(a.out, f"samples_{split}"))
            report(agg, rows, a.out, split)


def report(agg, rows, out_dir, split):
    n = len(rows)
    passed, reasons = M.pass_gate_split(agg, n_total=n)
    rep = {
        "split": split, "n_clips": n, "aggregate": agg,
        "pass": passed, "fail_reasons": reasons,
        "by_event_type": breakdown(rows, "event_type"),
        "by_snr": breakdown(rows, "snr_db"),
        "by_chain": breakdown(rows, "chain"),
        "by_distance": breakdown(rows, "distance"),
        "by_room": breakdown(rows, "room"),
    }
    with open(os.path.join(out_dir, f"report_{split}.json"), "w") as f:
        json.dump(rep, f, indent=2)

    print(f"\n===== {split.upper()} ({n} clips) =====", flush=True)
    for grp, label in (("event", "EVENT CLIPS"), ("clean", "CLEAN CLIPS")):
        g = agg.get(grp, {})
        if not g:
            continue
        cnt = agg.get("n_event" if grp == "event" else "n_clean")
        print(f"\n  --- {label} (n={cnt}) ---", flush=True)
        for k in ("delta_snr_evt", "delta_snr_imp", "snr_evt", "snr_in_evt",
                  "sisdr_evt", "stoi", "stoi_evt", "pesq", "pause_noise_db",
                  "polarity_corr"):
            v = g.get(k)
            if v:
                print(f"    {k:16s} mean={v['mean']:8.3f} "
                      f"median={v['median']:8.3f} ok={v['n_ok']} "
                      f"fail={v['n_fail']}", flush=True)
    print(f"\n  GATE: {'PASS' if passed else 'FAIL'}", flush=True)
    for r in reasons:
        print(f"    - {r}", flush=True)
    print("\n  delta_snr_evt by event_type:", flush=True)
    for g, d in rep["by_event_type"].items():
        print(f"    {g:12s} n={d['n']:4d}  evt={d['delta_snr_evt']}  imp={d['delta_snr_imp']}", flush=True)
    print("  delta_snr_evt by input SNR:", flush=True)
    for g, d in sorted(rep["by_snr"].items(),
                       key=lambda kv: (kv[0] == "none", kv[0])):
        print(f"    {g:12s} n={d['n']:4d}  evt={d['delta_snr_evt']}  imp={d['delta_snr_imp']}", flush=True)
    print("  delta_snr_evt by chain:", flush=True)
    for g, d in rep["by_chain"].items():
        print(f"    {g:12s} n={d['n']:4d}  evt={d['delta_snr_evt']}  imp={d['delta_snr_imp']}", flush=True)


if __name__ == "__main__":
    main()
