"""Dictionary-guided residual removal (supervised NMF), for use after MARK6.5.

Two fixed dictionaries, learned on TRAINING clips only:
  W_v  voice atoms   -- from clean voice targets
  W_r  noise atoms   -- from what is left to remove: the raw noise (mix - voice)
                        AND what MARK6.5 leaves behind (output - voice)

For a signal y, per frame (so it can run causally in real time), solve
|Y| ~= W_v H_v + W_r H_r with KL multiplicative updates, then apply the
soft mask  M = V^p / (V^p + R^p), floored at `floor` to avoid musical noise.
The noise activations W_r H_r also say WHERE noise remains (time and
frequency), which drives the stop rule of the iterative loop.
"""
import numpy as np
import torch

EPS = 1e-9


def kl_nmf(V, k, iters=200, seed=0):
    """Learn a dictionary W (F x k) on magnitudes V (F x N) with KL-NMF."""
    g = torch.Generator().manual_seed(seed)
    F, N = V.shape
    W = torch.rand(F, k, generator=g) + 0.1
    H = torch.rand(k, N, generator=g) + 0.1
    ones = torch.ones_like(V)
    for _ in range(iters):
        WH = W @ H + EPS
        H *= (W.T @ (V / WH)) / (W.T @ ones + EPS)
        WH = W @ H + EPS
        W *= ((V / WH) @ H.T) / (ones @ H.T + EPS)
        s = W.sum(0, keepdim=True) + EPS
        W /= s; H *= s.T
    return W


class DictRefiner:
    def __init__(self, W_v, W_r, iters=60, floor=0.1, power=2.0):
        self.W = torch.cat([W_v, W_r], 1)
        self.kv = W_v.shape[1]
        self.iters, self.floor, self.power = iters, floor, power

    def decompose(self, mag):
        """mag: (F, T). Returns voice estimate V and residual-noise estimate R."""
        W = self.W
        H = torch.full((W.shape[1], mag.shape[1]), 0.1)
        ones = torch.ones_like(mag)
        denom = W.T @ ones + EPS
        for _ in range(self.iters):              # each frame solved independently
            H *= (W.T @ (mag / (W @ H + EPS))) / denom
        V = W[:, :self.kv] @ H[:self.kv]
        R = W[:, self.kv:] @ H[self.kv:]
        return V, R

    def __call__(self, Y):
        """Y: complex STFT (T, F) torch. Returns (refined Y, noise-to-voice dB)."""
        mag = Y.abs().T                           # (F, T)
        V, R = self.decompose(mag)
        p = self.power
        M = V ** p / (V ** p + R ** p + EPS)
        M = torch.clamp(M, min=self.floor)
        nvr = 10 * torch.log10((R ** 2).sum() / ((V ** 2).sum() + EPS) + EPS).item()
        return Y * M.T, nvr, (R ** 2).sum(0)       # last: noise energy per frame
