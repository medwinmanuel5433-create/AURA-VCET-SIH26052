"""Causal residual gate driven by the model's own speech detector.

The network already predicts, every 8 ms frame, whether target speech is
present (detector channel 0, trained on the clean target). In frames where it
is confident there is no speech, what remains in the output can only be
residual noise -- mostly the blast's reverberant tail spreading into speech
gaps, which is what "mild noise spread across the whole signal" is.

The gate turns those frames down by up to `floor_db`, with a fast attack (so
speech onsets are not clipped) and a slow release (so word endings and soft
consonants are not chopped). It never boosts, never touches frames the
detector believes contain speech, and is strictly causal: frame t depends only
on detector outputs up to t. Parameters are chosen on the validation split
only, never on test.
"""
import numpy as np
import torch


def gate_curve(p_speech: np.ndarray, floor_db: float = -15.0,
               thresh: float = 0.5, attack_frames: int = 1,
               release_frames: int = 12) -> np.ndarray:
    """p_speech: (T,) in [0,1]. Returns per-frame linear gain (T,)."""
    floor = 10 ** (floor_db / 20.0)
    # soft map: below `thresh` -> floor, above -> 1, smooth in between
    target = np.clip((p_speech - (thresh - 0.25)) / 0.5, 0.0, 1.0)
    target = floor + (1 - floor) * target
    a_up = 1.0 / max(attack_frames, 1)
    a_dn = 1.0 / max(release_frames, 1)
    g = np.empty_like(target)
    cur = 1.0
    for t, v in enumerate(target):
        cur = cur + (a_up if v > cur else a_dn) * (v - cur)
        g[t] = cur
    return g


def apply_gate(est_spec: torch.Tensor, det_logits: torch.Tensor,
               floor_db: float = -15.0, **kw) -> torch.Tensor:
    """est_spec: complex (B,T,F); det_logits: (B,T,2). Returns gated spec."""
    p = torch.sigmoid(det_logits[..., 0]).cpu().numpy()
    g = np.stack([gate_curve(pi, floor_db, **kw) for pi in p])      # (B,T)
    return est_spec * torch.from_numpy(g).to(est_spec.real.dtype).unsqueeze(-1)
