"""Reference-mic residual post-filter (stage 6).

Standard subband NLMS adaptive noise canceller run on the network output Y: per frequency bin
a short complex FIR filter maps the outer reference mic onto Y and the filtered reference is
subtracted. Adaptation is slowed where the network says voice is present (p_voice).
The canceller output is used only as an attenuation gain (never boosts), softened by `soft`,
so the post-filter removes residual blast energy without adding artefacts.
Y, R: complex STFT (T, F) of network output and reference mic. Returns (Z, E)."""
import numpy as np
from .canceller import subband_nlms


def cancel_reference(Y, R, p_voice=None, K=4, mu0=0.2, soft=0.5):
    E, _ = subband_nlms(Y, R, K=K, mu0=mu0, steer=p_voice)
    g = np.clip(np.abs(E) / (np.abs(Y) + 1e-9), 0.0, 1.0) ** soft
    return Y * g, E
