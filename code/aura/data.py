"""Corpus dataset and the voice reference pool."""
import os
import json
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import Cfg, DEFAULT
from . import audio as A
from .losses import stft


class CorpusDataset(Dataset):
    """Reads the rendered corpus. The model receives mix; target is the
    achievable clean reference produced by the same capture chain."""

    def __init__(self, root: str, split: str, cfg: Cfg = DEFAULT,
                 refpool=None, crop_seconds: float = None):
        self.cfg = cfg
        # Training on random crops is safe here only because every time-axis
        # operation is causal and carries its own state: the model is length
        # invariant by construction. Evaluation always runs at full clip
        # length, and crops are never used for val/test.
        self.crop = int(cfg.sig.sr * crop_seconds) if crop_seconds else None
        self.dir = os.path.join(root, split)
        self.records = []
        mpath = os.path.join(self.dir, "manifest.jsonl")
        with open(mpath) as f:
            for line in f:
                self.records.append(json.loads(line))
        self.refpool = refpool

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        d = os.path.join(self.dir, r["id"])
        mix, _ = A.read_wav(os.path.join(d, "mix.wav"))
        tgt, _ = A.read_wav(os.path.join(d, "target.wav"))
        n = self.cfg.sig.n_samples
        mix = np.pad(mix, (0, max(0, n - len(mix))))[:n]
        tgt = np.pad(tgt, (0, max(0, n - len(tgt))))[:n]
        if self.crop and self.crop < n:
            # bias the crop toward the event so short crops still contain the
            # hard case rather than mostly quiet speech
            span = r.get("event_span")
            if span and np.random.rand() < 0.7:
                centre = (span[0] + span[1]) // 2
                s = int(np.clip(centre - self.crop // 2, 0, n - self.crop))
            else:
                s = np.random.randint(0, n - self.crop + 1)
            mix = mix[s:s + self.crop]
            tgt = tgt[s:s + self.crop]
        item = {
            "mix": torch.from_numpy(mix.astype(np.float32)),
            "target": torch.from_numpy(tgt.astype(np.float32)),
            "id": r["id"],
        }
        if self.refpool is not None:
            item["spk"] = self.refpool.embed_for(r["voice_file"])
        else:
            item["spk"] = torch.zeros(self.cfg.model.spk_dim)
        item["meta"] = r
        return item


def collate(batch, cfg: Cfg = DEFAULT):
    mix = torch.stack([b["mix"] for b in batch])
    tgt = torch.stack([b["target"] for b in batch])
    spk = torch.stack([b["spk"] for b in batch])
    out = {"mix": mix, "target": tgt, "spk": spk,
           "meta": [b["meta"] for b in batch],
           "id": [b["id"] for b in batch]}
    out["X"] = stft(mix, cfg)
    out["T"] = stft(tgt, cfg)
    return out


# --------------------------------------------------------------------------
# voice reference pool
# --------------------------------------------------------------------------

class RefPool:
    """A dictionary of voice references.

    Two jobs:
      1. Provide a speaker-conditioning embedding so the enhancer knows what
         the protected talker sounds like -- the single strongest cue for
         deciding what to preserve when speech and blast overlap.
      2. Report how many distinct voices the corpus actually contains, which
         determines whether a speaker-held-out evaluation is possible at all.

    The embedding is a log-mel statistics vector (mean + std), projected to
    `spk_dim`. It is deliberately not a learned speaker encoder: with the
    amount of speech available here a learned encoder would memorise rather
    than generalise. Swap in a pretrained ECAPA/x-vector encoder when a larger
    corpus is in place -- the interface does not change.
    """

    def __init__(self, voice_files, cfg: Cfg = DEFAULT, n_mels: int = 40):
        from .physmix import speaker_of
        self.cfg = cfg
        self.n_mels = n_mels
        self.files = list(voice_files)
        self.raw = {}
        self.spk_of = {}
        self._fb = self._mel_fb(n_mels, cfg.sig.n_fft, cfg.sig.sr)
        for p in self.files:
            try:
                k = os.path.basename(p)
                self.raw[k] = self._feat(p)
                self.spk_of[k] = speaker_of(p)
            except Exception:
                continue
        self.by_spk = {}
        for k, s in self.spk_of.items():
            self.by_spk.setdefault(s, []).append(k)
        if self.raw:
            M = np.stack(list(self.raw.values()))
            self.mu = M.mean(0)
            self.sd = M.std(0) + 1e-6
            self.proj = self._make_proj(M.shape[1], cfg.model.spk_dim)
        else:
            self.mu = self.sd = None
            self.proj = None

    @staticmethod
    def _mel_fb(n_mels, n_fft, sr):
        def hz2mel(f): return 2595 * np.log10(1 + f / 700)
        def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
        n_freq = n_fft // 2 + 1
        mels = np.linspace(hz2mel(50), hz2mel(sr / 2), n_mels + 2)
        hz = mel2hz(mels)
        bins = np.floor((n_fft + 1) * hz / sr).astype(int)
        fb = np.zeros((n_mels, n_freq), dtype=np.float32)
        for m in range(1, n_mels + 1):
            l, c, r = bins[m - 1], bins[m], bins[m + 1]
            if c == l: c = l + 1
            if r == c: r = c + 1
            r = min(r, n_freq - 1); c = min(c, n_freq - 1)
            for k in range(l, c):
                fb[m - 1, k] = (k - l) / max(c - l, 1)
            for k in range(c, r):
                fb[m - 1, k] = (r - k) / max(r - c, 1)
        return fb

    def _feat(self, path):
        x = A.load_audio(path, self.cfg.sig.sr)
        vad = A.active_speech_mask(x, self.cfg.sig.sr)
        if vad.sum() > self.cfg.sig.sr * 0.5:
            x = x[vad]
        x = A.set_rms(x, 0.06)
        n_fft, hop = self.cfg.sig.n_fft, self.cfg.sig.hop
        if len(x) < n_fft:
            x = np.pad(x, (0, n_fft - len(x)))
        fr = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop]
        S = np.abs(np.fft.rfft(fr * np.hanning(n_fft).astype(np.float32), n_fft, axis=-1))
        mel = np.log(S @ self._fb.T + 1e-5)
        return np.concatenate([mel.mean(0), mel.std(0)])

    @staticmethod
    def _make_proj(d_in, d_out, seed=0):
        rng = np.random.RandomState(seed)
        P = rng.randn(d_in, d_out).astype(np.float32) / np.sqrt(d_in)
        return P

    def embed_for(self, basename):
        """Speaker embedding from an ENROLLMENT set: the other utterances of the
        same speaker, never the file the target is made from.

        The previous version embedded the target's own clean source file, i.e.
        it handed the network a summary of the answer. (The audit showed the
        network ignored it -- dSNR +3.723 with, +3.732 zeroed -- but that is an
        accident of training, not a property to rely on.) With no other
        utterance of the speaker available, no enrollment exists and the
        embedding is zero, exactly as in deployment without enrollment.
        """
        if self.proj is None:
            return torch.zeros(self.cfg.model.spk_dim)
        s = self.spk_of.get(basename)
        others = [k for k in self.by_spk.get(s, []) if k != basename]
        if not others:
            return torch.zeros(self.cfg.model.spk_dim)
        v = np.mean([self.raw[k] for k in others], axis=0)
        z = (v - self.mu) / self.sd
        return torch.from_numpy((z @ self.proj).astype(np.float32))

    # ---- diagnostics --------------------------------------------------
    def estimate_speakers(self, max_k: int = 10):
        """Cluster the reference features and report a silhouette-style score
        per k, to estimate how many distinct voices the corpus contains."""
        if len(self.raw) < 4:
            return {"n_files": len(self.raw), "note": "too few files"}
        M = np.stack(list(self.raw.values()))
        Z = (M - M.mean(0)) / (M.std(0) + 1e-6)
        try:
            from scipy.cluster.vq import kmeans2
        except Exception:
            return {"n_files": len(M), "note": "scipy unavailable"}
        results = {}
        for k in range(2, min(max_k, len(M) - 1) + 1):
            try:
                cent, lab = kmeans2(Z, k, minit="++", seed=0)
            except Exception:
                continue
            # mean intra-cluster distance vs nearest-cluster distance
            sil = []
            for i in range(len(Z)):
                same = Z[lab == lab[i]]
                a = np.mean(np.linalg.norm(same - Z[i], axis=1)) if len(same) > 1 else 0.0
                b = np.inf
                for c in range(k):
                    if c == lab[i]:
                        continue
                    oth = Z[lab == c]
                    if len(oth) == 0:
                        continue
                    b = min(b, np.mean(np.linalg.norm(oth - Z[i], axis=1)))
                if np.isfinite(b) and max(a, b) > 0:
                    sil.append((b - a) / max(a, b))
            results[k] = round(float(np.mean(sil)), 4) if sil else None
        best = max((v for v in results.values() if v is not None), default=None)
        best_k = [k for k, v in results.items() if v == best]
        return {"n_files": len(M), "silhouette_by_k": results,
                "best_k": best_k[0] if best_k else None}
