"""MARK7-DM v2 = MARK7-DM + 16 ms lookahead + speaker voiceprint.

* Lookahead: the trunk also sees the features of the next 2 frames (voice mic
  and reference mic), so a gunshot's first frame is known before it has to be
  output. Output frame t is still aligned with input frame t; the device only
  buffers 2 hops (16 ms) of extra latency.
* Voiceprint: an 80-d log-mel mean/std of the wearer's clean voice (a short
  enrollment clip, or frames the gate found clean). It shifts the FiLM
  conditioning, telling the network whose voice to keep.
Every new weight starts at zero: at step 0 this is exactly MARK7-DM.
"""
import numpy as np
import torch
import torch.nn as nn
from .model import BandLinear, BandLayerNorm
from .model_dm import Mark7DM

N_MELS = 40
_FB = None


def _mel_fb(sr=16000, n_fft=512, n_mels=N_MELS):
    hz2mel = lambda f: 2595 * np.log10(1 + f / 700); mel2hz = lambda m: 700 * (10 ** (m / 2595) - 1)
    hz = mel2hz(np.linspace(hz2mel(50), hz2mel(sr / 2), n_mels + 2)); fr = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(fr)), np.float32)
    for m in range(1, n_mels + 1):
        l, c, r = hz[m - 1], hz[m], hz[m + 1]
        fb[m - 1] = np.clip(np.minimum((fr - l) / (c - l + 1e-9), (r - fr) / (r - c + 1e-9)), 0, None)
    return fb


def voiceprint(x, frames_mask=None):
    """80-d voiceprint: mean and std of log-mel over active (or given) frames."""
    global _FB
    if _FB is None:
        _FB = _mel_fb()
    x = np.asarray(x, np.float32)
    nf = 1 + (len(x) - 512) // 128
    if nf < 4:
        return np.zeros(2 * N_MELS, np.float32)
    fr = np.lib.stride_tricks.sliding_window_view(x, 512)[::128][:nf] * np.hanning(512).astype(np.float32)
    P = np.abs(np.fft.rfft(fr, axis=-1)) ** 2
    L = np.log(P @ _FB.T + 1e-8)                                           # (T, 40)
    e = P.sum(1)
    act = e > e.max() * 10 ** (-3.5)
    if frames_mask is not None:
        fm = np.asarray(frames_mask, bool)[:nf]; fm = np.pad(fm, (0, nf - len(fm)))
        act = act & fm
    if act.sum() < 8:
        return np.zeros(2 * N_MELS, np.float32)
    Lm = L[act]
    v = np.concatenate([Lm.mean(0) - Lm.mean(), Lm.std(0)])               # level-independent
    return v.astype(np.float32)


class Mark7DM2(Mark7DM):
    LA = 2

    def __init__(self, cfg):
        super().__init__(cfg)
        D = self.D
        self.la_in = nn.ModuleList()
        for k in range(self.LA):
            for src in ("x", "r"):
                self.la_in.append(nn.ModuleList([BandLinear(hi - lo, 3 * w, D) for w, lo, hi, _, _ in self.groups]))
        self.la_norm = nn.ModuleList([BandLayerNorm(hi - lo, 3 * w) for w, lo, hi, _, _ in self.groups])
        self.vp_proj = nn.Linear(2 * N_MELS, self.cfg.model.cond_dim)
        with torch.no_grad():
            for ml in self.la_in:
                for l in ml:
                    l.w.zero_(); l.b.zero_()
            self.vp_proj.weight.zero_(); self.vp_proj.bias.zero_()

    @staticmethod
    def _shift(X, k):
        return torch.cat([X[:, k:], torch.zeros_like(X[:, :k])], dim=1)

    def forward(self, X, R, spk_emb=None, vp=None):
        B, T, Fr = X.shape
        feats = self._feats(X, self.band_norm, self.band_in) + self._feats(R, self.ref_norm, self.ref_in)
        i = 0
        for k in range(1, self.LA + 1):
            for S in (X, R):
                feats = feats + self._feats(self._shift(S, k), self.la_norm, self.la_in[i]); i += 1
        det_h, _ = self.det_rnn(feats.mean(dim=2), None)
        det_logits = self.det_out(det_h)
        if spk_emb is None:
            spk_emb = torch.zeros(B, self.cfg.model.spk_dim, device=X.device)
        spk = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)
        cond = self.cond_mix(torch.cat([torch.sigmoid(det_logits), spk], dim=-1))
        if vp is not None:
            cond = cond + self.vp_proj(vp).unsqueeze(1)
        h = feats
        for blk in self.blocks:
            h, _ = blk(h, cond, None)
        z = self.decode(h)
        K = self.K
        coef = torch.tanh(z[..., :2 * K].reshape(B, T, Fr, K, 2)) * 2.0
        inp = z[..., 2 * K]; rec_hat = torch.sigmoid(z[..., 2 * K + 1])
        Y_df = self.deep_filter(X, coef)
        mag_inp = torch.exp(torch.clamp(inp, max=3.0)); ph = torch.angle(X)
        Y_inp = torch.complex(mag_inp * torch.cos(ph), mag_inp * torch.sin(ph))
        r = rec_hat
        Y = torch.complex(r * Y_df.real + (1 - r) * Y_inp.real, r * Y_df.imag + (1 - r) * Y_inp.imag)
        zz = torch.nn.functional.gelu(self.head_hid(self.head_norm(h)))
        cr = [self.ref_head[g](zz[:, :, lo:hi]).reshape(B, T, (hi - lo) * w, 2 * K) for g, (w, lo, hi, s, e) in enumerate(self.groups)]
        cref = torch.tanh(torch.cat(cr, dim=2).reshape(B, T, Fr, K, 2)) * 4.0
        N_ref = self.deep_filter(R, cref)
        Y = Y - torch.complex(r * N_ref.real, r * N_ref.imag)
        return {"est": Y, "est_df": Y_df, "det_logits": det_logits, "inpaint_logmag": inp,
                "rec_logit": z[..., 2 * K + 1], "rec_hat": rec_hat, "coef": coef, "cref": cref}
