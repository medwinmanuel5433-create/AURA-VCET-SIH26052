"""AURA-MARK6 network.

Design choices and where they come from
---------------------------------------
* Band-split front end (BSRNN / TF-GridNet family). Rather than treating the
  257 frequency bins as a flat axis, bins are grouped into 31 bands with fine
  resolution at low frequency and coarse at high. This cuts the sequence the
  recurrent layers see by ~8x, which is what makes the model affordable on the
  target class of hardware, and it matches the fact that speech structure is
  concentrated below 4 kHz while blast energy is broadband.

* Dual-path blocks. Each block does two things: model *across bands* within a
  single frame (bidirectional over frequency is legitimate -- frequency is not
  time), then model *across time* within each band using a unidirectional GRU.
  This is the TF-GridNet decomposition, made causal on the time axis only.
  Earlier pipelines used a bidirectional GRU over time, which is why their
  causal variant could never inherit the weights.

* Deep filtering instead of a plain mask (DeepFilterNet family). The decoder
  emits, per bin and frame, a short *complex FIR filter* over the previous
  `df_order` frames rather than a single gain. For impulsive noise this is the
  decisive difference: a per-bin gain can only attenuate a corrupted bin,
  whereas a filter can reconstruct it from the frames immediately before the
  blast, where the speech was still clean.

* FiLM conditioning. A small detector produces per-frame event evidence, which
  together with a speaker-reference embedding modulates every block. This gives
  the behavioural switching of a hard gated chain without branching, and keeps
  the whole thing one differentiable graph.

* Explicit passthrough path. When the detector says no event is present the
  model is trained (via the passthrough loss) to reproduce its input. This is
  the structural clean-audio guarantee, kept as a trained behaviour rather than
  a hard bypass so it degrades gracefully at event boundaries.

Causality: every time-axis operation is unidirectional or uses only past
frames. `assert_causal()` verifies this numerically rather than by inspection.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Cfg, DEFAULT


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def build_band_edges(band_plan, n_freq):
    edges, start = [], 0
    for width, count in band_plan:
        for _ in range(count):
            end = min(start + width, n_freq)
            if start < end:
                edges.append((start, end))
            start = end
    if start < n_freq:
        edges.append((start, n_freq))
    return edges


class CausalGRU(nn.Module):
    """Unidirectional GRU over time with explicit state, for streaming."""

    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.gru = nn.GRU(d_in, d_hidden, batch_first=True)
        self.proj = nn.Linear(d_hidden, d_out)

    def forward(self, x, state=None):
        y, state = self.gru(x, state)
        return self.proj(y), state


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.to_gamma = nn.Linear(cond_dim, feat_dim)
        self.to_beta = nn.Linear(cond_dim, feat_dim)

    def forward(self, x, cond):
        # x: (B, T, NB, D)   cond: (B, T, C)
        g = self.to_gamma(cond).unsqueeze(2)
        b = self.to_beta(cond).unsqueeze(2)
        return x * (1.0 + g) + b


class DualPathBlock(nn.Module):
    """One band-axis pass (bidirectional) + one time-axis pass (causal)."""

    def __init__(self, width, rnn_hidden, cond_dim):
        super().__init__()
        self.norm_band = nn.LayerNorm(width)
        self.band_rnn = nn.GRU(width, rnn_hidden, batch_first=True,
                               bidirectional=True)
        self.band_proj = nn.Linear(2 * rnn_hidden, width)

        self.norm_time = nn.LayerNorm(width)
        self.time_rnn = CausalGRU(width, rnn_hidden, width)

        self.film = FiLM(cond_dim, width)

    def forward(self, x, cond, state=None):
        # x: (B, T, NB, D)
        B, T, NB, D = x.shape

        # ---- across bands, within each frame (frequency axis) ----
        h = self.norm_band(x).reshape(B * T, NB, D)
        h, _ = self.band_rnn(h)
        h = self.band_proj(h).reshape(B, T, NB, D)
        x = x + h

        # ---- across time, within each band (causal) ----
        h = self.norm_time(x).permute(0, 2, 1, 3).reshape(B * NB, T, D)
        h, state = self.time_rnn(h, state)
        h = h.reshape(B, NB, T, D).permute(0, 2, 1, 3)
        x = x + h

        x = self.film(x, cond)
        return x, state


# --------------------------------------------------------------------------
# main model
# --------------------------------------------------------------------------

class Mark6(nn.Module):
    def __init__(self, cfg: Cfg = DEFAULT):
        super().__init__()
        self.cfg = cfg
        mc = cfg.model
        self.n_freq = cfg.sig.n_freq
        self.edges = build_band_edges(mc.band_plan, self.n_freq)
        self.NB = len(self.edges)
        self.D = mc.width
        self.K = mc.df_order

        # ---- band split: per-band encoder over [real, imag, logmag] --------
        self.band_norm = nn.ModuleList()
        self.band_in = nn.ModuleList()
        for (s, e) in self.edges:
            w = e - s
            self.band_norm.append(nn.LayerNorm(3 * w))
            self.band_in.append(nn.Linear(3 * w, self.D))

        # ---- detector (causal, operates on pooled band features) ----------
        self.det_rnn = CausalGRU(self.D, mc.rnn_hidden, mc.rnn_hidden)
        self.det_out = nn.Linear(mc.rnn_hidden, 2)   # [speech, event] logits

        # ---- conditioning -------------------------------------------------
        self.spk_proj = nn.Linear(mc.spk_dim, mc.cond_dim)
        self.cond_mix = nn.Linear(2 + mc.cond_dim, mc.cond_dim)

        # ---- dual-path trunk ----------------------------------------------
        self.blocks = nn.ModuleList([
            DualPathBlock(self.D, mc.rnn_hidden, mc.cond_dim)
            for _ in range(mc.n_blocks)])

        # ---- deep-filter decoder: per band -> 2*K coeffs per bin ----------
        self.band_out = nn.ModuleList()
        for (s, e) in self.edges:
            w = e - s
            self.band_out.append(nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D * 2),
                nn.GELU(),
                nn.Linear(self.D * 2, w * 2 * self.K),
            ))

        # ---- inpainting head: magnitude only, for unrecoverable bins ------
        self.inpaint = nn.ModuleList()
        for (s, e) in self.edges:
            w = e - s
            self.inpaint.append(nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D),
                nn.GELU(),
                nn.Linear(self.D, w),
            ))

        # ---- recoverability head ------------------------------------------
        # Predicts, per bin, how much speech information survived the capture
        # chain. At training time the true value is known from the corpus; at
        # inference it must be estimated, because it decides which of the two
        # synthesis paths is trusted. Without this the inpainting head is
        # supervised but never actually drives the output, leaving the model
        # unconstrained in exactly the bins where the blast lands.
        self.rec_head = nn.ModuleList()
        for (s, e) in self.edges:
            w = e - s
            self.rec_head.append(nn.Sequential(
                nn.LayerNorm(self.D),
                nn.Linear(self.D, self.D),
                nn.GELU(),
                nn.Linear(self.D, w),
            ))

    # ------------------------------------------------------------------
    def encode(self, X):
        """X: complex (B, T, F) -> band features (B, T, NB, D)"""
        re, im = X.real, X.imag
        mag = torch.sqrt(re ** 2 + im ** 2 + 1e-9)
        logmag = torch.log(mag + 1e-5)
        feats = []
        for i, (s, e) in enumerate(self.edges):
            # explicit concatenation along the feature axis; the channel
            # layout is built here rather than reshaped from a flat vector,
            # so bins cannot be interleaved across channels.
            b = torch.cat([re[..., s:e], im[..., s:e], logmag[..., s:e]], dim=-1)
            b = self.band_norm[i](b)
            feats.append(self.band_in[i](b))
        return torch.stack(feats, dim=2)          # (B, T, NB, D)

    def deep_filter(self, X, coef, x_tail=None):
        """Apply causal complex FIR across time, per bin.

        X:     complex (B, T, F)
        coef:  (B, T, F, K, 2)   coefficients for taps t, t-1, ... t-K+1
        x_tail: complex (B, K-1, F) -- the frames immediately preceding X.

        `x_tail` is what makes this correct in streaming. Zero-padding the
        first K-1 frames is only valid at the true start of a signal; doing it
        at every chunk boundary silently discards the filter's tap history and
        makes chunked output differ from single-pass output.
        """
        B, T, Fr = X.shape
        K = self.K
        if K > 1:
            if x_tail is None:
                x_tail = X.new_zeros(B, K - 1, Fr)
            elif x_tail.shape[1] < K - 1:
                # a short first chunk can leave fewer than K-1 frames of
                # history; left-pad so the tap window is always well formed
                pad = X.new_zeros(B, K - 1 - x_tail.shape[1], Fr)
                x_tail = torch.cat([pad, x_tail], dim=1)
            Xp = torch.cat([x_tail[:, -(K - 1):], X], dim=1)      # (B, T+K-1, F)
        else:
            Xp = X
        # stack the K causal taps
        taps = torch.stack([Xp[:, K - 1 - k: K - 1 - k + T] for k in range(K)],
                           dim=3)                                  # (B,T,F,K)
        cr, ci = coef[..., 0], coef[..., 1]
        out_r = (taps.real * cr - taps.imag * ci).sum(dim=3)
        out_i = (taps.real * ci + taps.imag * cr).sum(dim=3)
        return torch.complex(out_r, out_i)

    def forward(self, X, spk_emb=None, states=None):
        """X: complex (B, T, F). Returns dict with estimate and heads.

        `states` is a dict carrying every piece of time-axis history:
        per-block GRU states, the detector GRU state, and the deep-filter tap
        tail. All three must be threaded, or chunked inference diverges from
        single-pass inference.
        """
        B, T, Fr = X.shape
        if states is not None and not isinstance(states, dict):
            states = {"blocks": states, "det": None, "x_tail": None}
        st_blocks = states.get("blocks") if states else None
        st_det = states.get("det") if states else None
        x_tail = states.get("x_tail") if states else None

        feats = self.encode(X)                                   # (B,T,NB,D)

        # detector on band-pooled features
        pooled = feats.mean(dim=2)                               # (B,T,D)
        det_h, st_det_new = self.det_rnn(pooled, st_det)
        det_logits = self.det_out(det_h)                         # (B,T,2)

        if spk_emb is None:
            spk_emb = torch.zeros(B, self.cfg.model.spk_dim, device=X.device)
        spk = self.spk_proj(spk_emb).unsqueeze(1).expand(-1, T, -1)
        cond = self.cond_mix(torch.cat([torch.sigmoid(det_logits), spk], dim=-1))

        h = feats
        new_states = []
        for i, blk in enumerate(self.blocks):
            st = None if st_blocks is None else st_blocks[i]
            h, st = blk(h, cond, st)
            new_states.append(st)

        # ---- deep filter coefficients ----
        coef = X.new_zeros(B, T, Fr, self.K, 2, dtype=torch.float32)
        inp = X.new_zeros(B, T, Fr, dtype=torch.float32)
        rec_logit = X.new_zeros(B, T, Fr, dtype=torch.float32)
        for i, (s, e) in enumerate(self.edges):
            w = e - s
            c = self.band_out[i](h[:, :, i])                     # (B,T,w*2*K)
            c = c.reshape(B, T, w, self.K, 2)
            # bound the filter so a single frame cannot blow up
            c = torch.tanh(c) * 2.0
            coef[:, :, s:e] = c
            inp[:, :, s:e] = self.inpaint[i](h[:, :, i])
            rec_logit[:, :, s:e] = self.rec_head[i](h[:, :, i])

        Y_df = self.deep_filter(X, coef, x_tail)
        rec_hat = torch.sigmoid(rec_logit)
        # Next call's tap history: the last K-1 frames of (history + this
        # chunk), so a chunk shorter than K-1 frames still leaves a full window.
        if self.K > 1:
            if x_tail is not None:
                hist = torch.cat([x_tail, X], dim=1)
            else:
                hist = X
            new_tail = hist[:, -(self.K - 1):].detach()
        else:
            new_tail = None

        # Inpainting path: predicted magnitude carried on the mixture's phase.
        # Phase is not recoverable in destroyed bins, and pretending otherwise
        # is what drives the hallucinated broadband energy seen in earlier runs.
        mag_inp = torch.exp(torch.clamp(inp, max=3.0))
        phase = torch.angle(X)
        Y_inp = torch.complex(mag_inp * torch.cos(phase),
                              mag_inp * torch.sin(phase))

        # Trust the filter where information survived, the synthesiser where it
        # did not. This blend is what keeps output energy bounded inside the
        # blast, where the reconstruction loss is necessarily weak.
        r = rec_hat
        Y = torch.complex(r * Y_df.real + (1 - r) * Y_inp.real,
                          r * Y_df.imag + (1 - r) * Y_inp.imag)

        return {
            "est": Y,
            "est_df": Y_df,
            "det_logits": det_logits,
            "inpaint_logmag": inp,
            "rec_logit": rec_logit,
            "rec_hat": rec_hat,
            "states": {"blocks": new_states, "det": st_det_new,
                       "x_tail": new_tail},
            "coef": coef,
        }


# --------------------------------------------------------------------------
# verification
# --------------------------------------------------------------------------

@torch.no_grad()
def assert_causal(model: Mark6, T: int = 40, Fr: int = None, tol: float = 1e-6):
    """Numerically verify that output frame t does not depend on inputs > t.

    Perturbs the second half of the input and checks the first half of the
    output is bit-identical. This catches any accidental bidirectional or
    lookahead path on the time axis, which inspection alone has missed before.
    """
    model.eval()
    Fr = Fr or model.n_freq
    X = torch.randn(1, T, Fr) + 1j * torch.randn(1, T, Fr)
    y1 = model(X)["est"]
    X2 = X.clone()
    half = T // 2
    X2[:, half:] = torch.randn(1, T - half, Fr) + 1j * torch.randn(1, T - half, Fr)
    y2 = model(X2)["est"]
    diff = (y1[:, :half] - y2[:, :half]).abs().max().item()
    ok = diff < tol
    return ok, diff


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
