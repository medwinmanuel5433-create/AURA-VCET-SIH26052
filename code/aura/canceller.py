"""AI-steered sub-band adaptive noise canceller (the MARK7-DM front end).

Classic two-mic adaptive noise cancellation (Widrow 1975) predicts the noise in
the primary (voice) mic from a reference mic that hears the noise but little
voice, and subtracts it. Here it runs per STFT bin (sub-band NLMS, K taps of
8 ms each) so it is cheap, converges fast and covers long reverberant paths.

Plain NLMS fails on blasts in two ways:
  1. it keeps adapting while the wearer talks -> learns the voice leak in the
     reference mic and cancels part of the voice ("double talk");
  2. a blast is a sudden change of the acoustic path, far faster than NLMS
     converges.
The steering fixes both with the MARK network itself (novel part):
  * mu(t,f) = mu0 * (1 - p_voice(t,f))^gamma where p_voice is the network's
    own voice mask on the primary mic -> adapts only in noise-dominated bins,
    frozen under voice;
  * onset kick: when the reference mic jumps >= kick_db above its running
    floor (a blast starts), mu is raised for `kick_ms` so the filter locks on
    within a few frames;
  * update clipping so a single clipped/saturated frame cannot throw W away.
Everything is causal, frame by frame.
"""
import numpy as np


def subband_nlms(X, R, K=4, mu0=0.5, steer=None, gamma=1.0, kick_db=12.0, kick_mu=1.0,
                 kick_ms=60.0, hop_s=0.008, delta=1e-8, max_step=1.0, leak=1.0):
    """X, R: complex STFT (T, F) of primary and reference mic.
    steer: None (plain NLMS) or p_voice (T, F) in [0,1] from the network.
    Returns E = X - N_hat (T, F) and N_hat."""
    T, F = X.shape
    W = np.zeros((F, K), np.complex128)
    buf = np.zeros((F, K), np.complex128)
    E = np.empty_like(X); Nh = np.empty_like(X)
    fr_e = np.log10((np.abs(R) ** 2).sum(1) + 1e-12) * 10
    floor = fr_e[0]; kick_left = 0; kick_n = max(1, int(kick_ms / 1000 / hop_s))
    a_floor = 0.02
    for t in range(T):
        buf = np.roll(buf, 1, axis=1); buf[:, 0] = R[t]
        nh = (W * buf).sum(1)
        e = X[t] - nh
        E[t] = e; Nh[t] = nh
        # onset detection on the reference mic (it hears the blast first and loudest)
        if steer is not None:
            if fr_e[t] > floor + kick_db:
                kick_left = kick_n
            floor = min(fr_e[t], floor + 0.05) if fr_e[t] < floor else (1 - a_floor) * floor + a_floor * fr_e[t]
            mu = mu0 * (1.0 - steer[t]) ** gamma
            if kick_left > 0:
                mu = np.maximum(mu, kick_mu * (1.0 - steer[t]))
                kick_left -= 1
        else:
            mu = mu0
        p = (np.abs(buf) ** 2).sum(1) + delta
        step = (mu / p)[:, None] * e[:, None] * np.conj(buf)
        mag = np.abs(step).max(1, keepdims=True) / (np.abs(W).max(1, keepdims=True) + max_step)
        step = step / np.maximum(1.0, mag)
        W = leak * W + step
    return E, Nh
