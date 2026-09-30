"""Streaming inference with persistent state.

What this fixes
---------------
Earlier "streaming" modes re-ran the whole offline pipeline on each chunk from
scratch, with no state carried between chunks. That demonstrates chunking, not
streaming: every chunk boundary resets the recurrent layers, and the result
tells you nothing about how the system behaves on a continuous input.

Here the recurrent state of every block is threaded across chunks, together
with the STFT analysis tail and the overlap-add tail, so processing a signal in
N chunks produces the same output as processing it in one pass. That equality
is asserted by `verify_streaming_equivalence()` -- if streaming and offline
disagree, the streaming path is wrong and the number it produces is worthless.

Latency accounting (defaults: 16 kHz, 512-sample window, 128-sample hop):
    algorithmic  = window - hop  = 384 samples = 24.0 ms
    chunk        = hop * frames_per_chunk
    total        = algorithmic + chunk
There is no lookahead: the model is causal, verified numerically in model.py.
"""
import numpy as np
import torch

from .config import Cfg, DEFAULT


class StreamingEnhancer:
    """Frame-synchronous wrapper around Mark6.

    Usage:
        st = StreamingEnhancer(model, cfg)
        for chunk in chunks:                 # chunk: 1-D float32
            out = st.process(chunk)          # same length as chunk
        tail = st.flush()
    """

    def __init__(self, model, cfg: Cfg = DEFAULT, spk_emb=None):
        self.model = model.eval()
        self.cfg = cfg
        self.n_fft = cfg.sig.n_fft
        self.hop = cfg.sig.hop
        self.win = torch.hann_window(cfg.sig.win)
        self.spk = spk_emb
        self.reset()

    def reset(self):
        self.states = None
        # samples not yet consumed by a full analysis frame
        self._in_buf = np.zeros(0, dtype=np.float32)
        # overlap-add accumulator and its window-sum normaliser
        self._ola = np.zeros(0, dtype=np.float32)
        self._wsum = np.zeros(0, dtype=np.float32)
        self._emitted = 0
        self._frames_seen = 0

    # ------------------------------------------------------------------
    @property
    def algorithmic_latency_samples(self) -> int:
        return self.cfg.sig.win - self.hop

    def latency_ms(self, chunk_samples: int) -> dict:
        sr = self.cfg.sig.sr
        alg = self.algorithmic_latency_samples / sr * 1000
        chk = chunk_samples / sr * 1000
        return {"algorithmic_ms": round(alg, 2),
                "chunk_ms": round(chk, 2),
                "total_ms": round(alg + chk, 2)}

    # ------------------------------------------------------------------
    def _frames_from_buffer(self):
        """Pop as many complete analysis frames as the buffer allows."""
        frames = []
        while len(self._in_buf) >= self.n_fft:
            frames.append(self._in_buf[:self.n_fft].copy())
            self._in_buf = self._in_buf[self.hop:]
        return frames

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """Feed `chunk`, return the samples that became available."""
        chunk = np.asarray(chunk, dtype=np.float32)
        self._in_buf = np.concatenate([self._in_buf, chunk])
        frames = self._frames_from_buffer()
        if not frames:
            return np.zeros(0, dtype=np.float32)

        w = self.win.numpy()
        F = np.stack(frames) * w                       # (nf, n_fft)
        spec = np.fft.rfft(F, self.n_fft, axis=-1)
        X = torch.from_numpy(spec.astype(np.complex64)).unsqueeze(0)

        with torch.no_grad():
            out = self.model(X, self.spk, states=self.states)
        self.states = out["states"]
        Y = out["est"][0].numpy()

        y_frames = np.fft.irfft(Y, self.n_fft, axis=-1).astype(np.float32) * w

        # overlap-add into the accumulator
        need = self._frames_seen * self.hop + len(frames) * self.hop + self.n_fft
        if len(self._ola) < need:
            pad = need - len(self._ola)
            self._ola = np.concatenate([self._ola, np.zeros(pad, np.float32)])
            self._wsum = np.concatenate([self._wsum, np.zeros(pad, np.float32)])
        for i, fr in enumerate(y_frames):
            s = (self._frames_seen + i) * self.hop
            self._ola[s:s + self.n_fft] += fr
            self._wsum[s:s + self.n_fft] += w ** 2
        self._frames_seen += len(frames)

        # emit everything that is fully overlapped (all contributing frames in)
        ready_to = max(0, (self._frames_seen - 1) * self.hop)
        if ready_to <= self._emitted:
            return np.zeros(0, dtype=np.float32)
        seg = self._ola[self._emitted:ready_to]
        wn = self._wsum[self._emitted:ready_to]
        out_samples = (seg / np.maximum(wn, 1e-8)).astype(np.float32)
        self._emitted = ready_to
        return out_samples

    def flush(self) -> np.ndarray:
        """Emit the remaining overlap-add tail."""
        end = min(len(self._ola), self._frames_seen * self.hop + self.n_fft)
        if end <= self._emitted:
            return np.zeros(0, dtype=np.float32)
        seg = self._ola[self._emitted:end]
        wn = self._wsum[self._emitted:end]
        out = (seg / np.maximum(wn, 1e-8)).astype(np.float32)
        self._emitted = end
        return out


# ----------------------------------------------------------------------
def offline_reference(model, x: np.ndarray, cfg: Cfg = DEFAULT, spk=None):
    """Process the whole signal in one pass, matching the streaming framing.

    Uses centre=False framing so the two paths are directly comparable; the
    training path uses centre=True, which differs only by edge padding.
    """
    n_fft, hop = cfg.sig.n_fft, cfg.sig.hop
    w = torch.hann_window(cfg.sig.win).numpy()
    nf = 1 + max(0, (len(x) - n_fft) // hop)
    F = np.stack([x[i * hop:i * hop + n_fft] for i in range(nf)]) * w
    spec = np.fft.rfft(F, n_fft, axis=-1)
    X = torch.from_numpy(spec.astype(np.complex64)).unsqueeze(0)
    with torch.no_grad():
        out = model(X, spk)
    Y = out["est"][0].numpy()
    y_frames = np.fft.irfft(Y, n_fft, axis=-1).astype(np.float32) * w
    total = (nf - 1) * hop + n_fft
    ola = np.zeros(total, np.float32)
    wsum = np.zeros(total, np.float32)
    for i, fr in enumerate(y_frames):
        ola[i * hop:i * hop + n_fft] += fr
        wsum[i * hop:i * hop + n_fft] += w ** 2
    return (ola / np.maximum(wsum, 1e-8)).astype(np.float32)


def verify_streaming_equivalence(model, cfg: Cfg = DEFAULT,
                                 n: int = 32000, chunk: int = 1024,
                                 tol: float = 1e-4):
    """Assert chunked streaming reproduces the single-pass result.

    Returns (ok, max_abs_diff, n_compared). A failure here means the streaming
    path is not the same system that was evaluated offline.
    """
    rng = np.random.RandomState(0)
    x = (rng.randn(n) * 0.05).astype(np.float32)

    ref = offline_reference(model, x, cfg)

    st = StreamingEnhancer(model, cfg)
    pieces = []
    for i in range(0, len(x), chunk):
        pieces.append(st.process(x[i:i + chunk]))
    pieces.append(st.flush())
    stream = np.concatenate([p for p in pieces if len(p)])

    m = min(len(ref), len(stream))
    # ignore the final partial-overlap tail, which differs by construction
    m = max(0, m - cfg.sig.n_fft)
    if m <= 0:
        return False, float("inf"), 0
    diff = float(np.max(np.abs(ref[:m] - stream[:m])))
    return diff < tol, diff, m
