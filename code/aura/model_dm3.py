"""MARK7-DM v3 = MARK7-DM v2 + DSRA conditioning.

The DSRA frame descriptors (from the V9 notebook: spectral centroid, flatness,
85% roll-off, low / speech / high band energy shares, frame energy and its
change) of BOTH microphones are fed, every frame, into the FiLM conditioning.
They are exactly the cues that separate a harmonic voice frame (low flatness,
speech-band energy, centroid 250-3500 Hz) from a blast frame (flat, broadband,
sudden energy jump), handed to the network explicitly. All causal.
The voiceprint is also built only from frames that are blast-free (gate) AND
voice (DSRA voice detector). New weights start at zero: at step 0 this is v2.
"""
import numpy as np
import torch
import torch.nn as nn
from .model_dm2 import Mark7DM2
from .dsra_level import dsra_features, voice_activity

N_DS = 8


def dsra_vec(x):
    """(T, 8) causal, level-independent DSRA descriptors."""
    f = dsra_features(np.asarray(x, np.float32)); e = f["energy"] + 1e-10
    le = np.log10(e); rel = le - np.maximum.accumulate(le)           # dB below the loudest frame so far
    v = np.stack([f["centroid"] / 4000.0, f["flat"], f["roll"] / 8000.0,
                  np.log10(f["low"] / e + 1e-6) / 3, np.log10(f["speech"] / e + 1e-6) / 3, np.log10(f["high"] / e + 1e-6) / 3,
                  np.clip(rel / 6.0, -2, 0), np.clip(f["delta"] / 5.0, -2, 2)], 1)
    return v.astype(np.float32)


def dsra_voice_mask(x):
    return voice_activity(np.asarray(x, np.float32))[0]


class Mark7DM3(Mark7DM2):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ds_proj = nn.Linear(2 * N_DS, self.cfg.model.cond_dim)
        with torch.no_grad():
            self.ds_proj.weight.zero_(); self.ds_proj.bias.zero_()

    def forward(self, X, R, spk_emb=None, vp=None, ds=None):
        self._ds = ds
        return super().forward(X, R, spk_emb, vp)

    # inject DSRA into the conditioning by wrapping cond_mix
    def _apply_ds(self, cond):
        if getattr(self, "_ds", None) is None:
            return cond
        ds = self._ds[:, :cond.shape[1]]
        if ds.shape[1] < cond.shape[1]:
            ds = torch.cat([ds, ds[:, -1:].expand(-1, cond.shape[1] - ds.shape[1], -1)], 1)
        return cond + self.ds_proj(ds)


class _CondMixDS(nn.Module):
    """Wraps the original cond_mix so v2's forward picks up the DSRA term."""
    def __init__(self, base, owner):
        super().__init__(); self.base = base; self.owner = [owner]
    def forward(self, z):
        return self.owner[0]._apply_ds(self.base(z))


def build_v3(cfg):
    m = Mark7DM3(cfg)
    m.cond_mix = _CondMixDS(m.cond_mix, m)
    return m
