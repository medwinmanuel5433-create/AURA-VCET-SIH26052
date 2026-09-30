"""MARK7-DM network: MARK6.5 + a reference-mic branch with a NEURAL CANCELLER.

Classical two-mic cancellation (NLMS/RLS) must *converge* to the acoustic path
between the mics; a gunshot is over before it does. Here the network predicts,
every frame and every bin, a K-tap complex filter that is applied to the
REFERENCE mic and subtracted from its own estimate:

    Y = blend(DF_x(X), inpaint)  -  r * DF_ref(R)

so the cancelling filter is re-estimated instantly from context instead of
by slow gradient adaptation -- an "instant adaptive filter". The reference
features also enter the trunk, so the voice/blast decision sees both mics.

Every new weight starts at zero: at step 0 the model is exactly MARK6.5, and
training can only add what the second mic is worth.
"""
import torch
import torch.nn as nn
from .model import Mark6, BandLinear, BandLayerNorm


class Mark7DM(Mark6):
    def __init__(self, cfg):
        super().__init__(cfg)
        D = self.D; K = self.K
        self.ref_norm = nn.ModuleList([BandLayerNorm(hi - lo, 3 * w) for w, lo, hi, _, _ in self.groups])
        self.ref_in = nn.ModuleList([BandLinear(hi - lo, 3 * w, D) for w, lo, hi, _, _ in self.groups])
        self.ref_head = nn.ModuleList([BandLinear(hi - lo, 2 * D, w * 2 * K) for w, lo, hi, _, _ in self.groups])
        with torch.no_grad():
            for l in list(self.ref_in) + list(self.ref_head):
                l.w.zero_(); l.b.zero_()

    def _feats(self, X, norms, lins):
        B, T, _ = X.shape
        re, im = X.real, X.imag
        lm = torch.log(torch.sqrt(re ** 2 + im ** 2 + 1e-9) + 1e-5)
        out = []
        for g, (w, lo, hi, s, e) in enumerate(self.groups):
            n = hi - lo
            b = torch.cat([re[..., s:e].reshape(B, T, n, w), im[..., s:e].reshape(B, T, n, w),
                           lm[..., s:e].reshape(B, T, n, w)], dim=-1)
            out.append(lins[g](norms[g](b)))
        return torch.cat(out, dim=2)

    def forward(self, X, R, spk_emb=None):
        B, T, Fr = X.shape
        feats = self._feats(X, self.band_norm, self.band_in) + self._feats(R, self.ref_norm, self.ref_in)
        det_h, _ = self.det_rnn(feats.mean(dim=2), None)
        det_logits = self.det_out(det_h)
        if spk_emb is None:
            spk_emb = torch.zeros(B, self.cfg.model.spk_dim, device=X.device)
        spk = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)
        cond = self.cond_mix(torch.cat([torch.sigmoid(det_logits), spk], dim=-1))
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
        # neural canceller on the reference mic
        zz = torch.nn.functional.gelu(self.head_hid(self.head_norm(h)))
        cr = []
        for g, (w, lo, hi, s, e) in enumerate(self.groups):
            cr.append(self.ref_head[g](zz[:, :, lo:hi]).reshape(B, T, (hi - lo) * w, 2 * K))
        cref = torch.tanh(torch.cat(cr, dim=2).reshape(B, T, Fr, K, 2)) * 4.0
        N_ref = self.deep_filter(R, cref)
        Y = Y - torch.complex(r * N_ref.real, r * N_ref.imag)
        return {"est": Y, "est_df": Y_df, "det_logits": det_logits, "inpaint_logmag": inp,
                "rec_logit": z[..., 2 * K + 1], "rec_hat": rec_hat, "coef": coef, "cref": cref}
