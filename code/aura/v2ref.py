"""v2 pipeline, copied unchanged from the user's v2 notebook (Stages 0-4 + run_pipeline).
Used only as the comparison baseline and for its event gate / reference cancellation."""
import numpy as np
from scipy.signal import stft, istft, windows, correlate
from scipy.ndimage import median_filter

def nmf_learn(mag, n_components, n_iter=200, seed=0):
    """Basic multiplicative-update KL-NMF: mag (F,T) ~= W (F,K) @ H (K,T)."""
    rng = np.random.default_rng(seed)
    F, T = mag.shape
    W = rng.random((F, n_components)) + 1e-3
    H = rng.random((n_components, T)) + 1e-3
    eps = 1e-9
    for _ in range(n_iter):
        WH = W @ H + eps
        H *= (W.T @ (mag / WH)) / (W.T @ np.ones_like(mag) + eps)
        WH = W @ H + eps
        W *= ((mag / WH) @ H.T) / (np.ones_like(mag) @ H.T + eps)
        norm = W.sum(axis=0, keepdims=True) + eps
        W /= norm
        H *= norm.T
    return W, H


def learn_dictionary(wav, sr, n_fft, hop, n_components, n_iter=200, seed=0):
    """Learns a fixed spectral dictionary from an isolated reference
    recording (e.g. an enrollment clip of the wearer's own voice, or a
    library noise recording)."""
    _, _, Z = stft(wav, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    W, _ = nmf_learn(mag, n_components, n_iter=n_iter, seed=seed)
    return W


def locate_event_xcorr(mixture, noise_ref, sr):
    # (fixed) scipy's mode='valid' silently swaps roles when noise_ref is
    # longer than mixture: the output is still non-empty, but the index
    # it returns no longer means "reference starts here in the mixture" --
    # it comes out as a bogus, sometimes-huge sample offset (this crashed
    # on real Drive data: 851 gunshot files of varying length meant some
    # references were longer than the mixture being tested). A reference
    # that doesn't even fit inside the mixture can't be meaningfully
    # localized by this correlation, so treat it as no match instead of
    # guessing.
    if len(noise_ref) >= len(mixture):
        return 0, min(len(noise_ref), len(mixture)), 0.0
    corr = correlate(mixture, noise_ref, mode='valid')
    onset = int(np.argmax(np.abs(corr)))
    offset = onset + len(noise_ref)
    peak = np.abs(corr[onset])
    norm = np.sqrt(np.sum(noise_ref ** 2) * np.sum(mixture[onset:onset + len(noise_ref)] ** 2)) + 1e-9
    confidence = peak / norm
    return onset, offset, confidence


def spectral_flatness_envelope(mixture, sr, n_fft=512, hop=128):
    """Per-sample spectral flatness ("Wiener entropy"): near 1 for
    broadband/impulsive content (gunshots, explosions), much lower for
    voiced speech even when loud."""
    _, _, Z = stft(mixture, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z) + 1e-9
    gmean = np.exp(np.mean(np.log(mag), axis=0))
    amean = np.mean(mag, axis=0)
    flatness = gmean / amean
    n = len(mixture)
    frame_centers = np.arange(len(flatness)) * hop
    return np.interp(np.arange(n), frame_centers, flatness)


def locate_event_energy(mixture, sr, expected_len=None, env_ms=6,
                         enter_db=20.0, exit_db=10.0, floor_percentile=10,
                         hold_ms=150, flatness_thresh=0.30, min_dwell_ms=120):
    """Template-independent event boundary detector: energy excess over a
    global low-percentile noise floor, gated by spectral flatness, AND
    required to hold continuously for `min_dwell_ms` before triggering.
    The dwell requirement is what actually separates a loud voice moment
    (tens of ms) from a real blast (hundreds of ms+) -- energy or flatness
    alone were each fooled by real speech during testing."""
    env_win = max(1, int(sr * env_ms / 1000))
    kernel = np.ones(env_win) / env_win
    env = np.sqrt(np.convolve(mixture ** 2, kernel, mode='same') + 1e-12)
    n = len(env)
    flatness = spectral_flatness_envelope(mixture, sr)

    bg = max(np.percentile(env, floor_percentile), 1e-6)
    excess_db = 20 * np.log10((env + 1e-9) / bg)

    qualifies = (excess_db > enter_db) & (flatness > flatness_thresh)
    dwell = max(1, int(sr * min_dwell_ms / 1000))
    hold = max(1, int(sr * hold_ms / 1000))

    active = np.zeros(n, dtype=bool)
    is_active = False
    qualify_run = 0
    hold_ctr = 0
    for i in range(n):
        if qualifies[i]:
            qualify_run += 1
        else:
            qualify_run = 0
        if not is_active and qualify_run >= dwell:
            is_active = True
            hold_ctr = hold
        elif is_active and excess_db[i] < exit_db:
            if hold_ctr <= 0:
                is_active = False
            else:
                hold_ctr -= 1
        elif is_active:
            hold_ctr = hold
        active[i] = is_active

    idxs = np.where(active)[0]
    if len(idxs) == 0:
        return None, None, 0.0
    gaps = np.where(np.diff(idxs) > hold)[0]
    runs = np.split(idxs, gaps + 1)
    best_run = max(runs, key=lambda r: r[-1] - r[0])
    onset = max(0, int(best_run[0]) - dwell + 1)
    offset = int(best_run[-1])
    if expected_len is not None:
        offset = min(n, onset + expected_len)
    strength = float(np.mean(excess_db[onset:offset]))
    return onset, offset, strength


def best_matching_reference(mixture, ref_library, sr):
    """Searches every recording in a library of known threat signatures
    and returns the best match. Two rounds fired from the same weapon are
    not sample-identical, so a single reference is a poor stand-in --
    searching a broad library is what actually makes cross-correlation
    cancellation generalize."""
    best = None
    for ref in ref_library:
        onset, offset, conf = locate_event_xcorr(mixture, ref, sr)
        if best is None or conf > best[2]:
            best = (onset, offset, conf, ref)
    return best


def locate_event(mixture, noise_ref, sr, ref_library=None):
    """Returns (onset, offset, lib_ref, confidence). lib_ref is the
    library recording used for stage 1, or None if nothing was found
    confidently (energy detector supplies the boundaries in that case).
    confidence is the raw cross-correlation score (0-1ish) even when it fell
    below the hard accept threshold -- callers that want a graded fallback
    (e.g. the AI/ML hybrid blend) use this instead of just the boolean.
    """
    library = [noise_ref] + (list(ref_library) if ref_library else [])
    onset, offset, conf, lib_ref = best_matching_reference(mixture, library, sr)
    en_onset, en_offset, _ = locate_event_energy(mixture, sr, expected_len=len(noise_ref))

    if conf > 0.5:
        return onset, offset, lib_ref, conf
    if en_onset is None:
        return onset, offset, lib_ref, conf
    return en_onset, en_offset, None, conf


def reference_cancel(mixture, noise_ref, onset, offset, sr, block_ms=20, overlap=0.75,
                      min_confidence=0.3):
    """Short-time least-squares cross-correlation subtraction, block by
    block, overlap-added back with a Hann taper. Skips subtraction in any
    block where the aligned reference doesn't actually explain much of
    that block's energy (confidence gate) -- forcing it there would just
    inject a wrong-shaped copy of the reference as new distortion."""
    n = len(mixture)
    out = mixture.copy()
    block = max(1, int(sr * block_ms / 1000))
    hop = max(1, int(block * (1 - overlap)))
    win = windows.hann(block, sym=False)
    acc = np.zeros(n)
    wsum = np.zeros(n)

    ref_len = len(noise_ref)
    span_lo, span_hi = max(0, onset), min(n, offset)

    pos = span_lo
    while pos < span_hi:
        b_end = min(n, pos + block)
        b = mixture[pos:b_end]
        ref_idx = pos - onset
        if ref_idx < 0 or ref_idx >= ref_len:
            r = np.zeros_like(b)
        else:
            r_end = min(ref_len, ref_idx + (b_end - pos))
            r = noise_ref[ref_idx:r_end]
            if len(r) < len(b):
                r = np.pad(r, (0, len(b) - len(r)))
        denom = np.dot(r, r)
        alpha = (np.dot(b, r) / denom) if denom > 1e-12 else 0.0
        alpha = np.clip(alpha, 0.0, 20.0)
        b_energy = np.dot(b, b) + 1e-12
        explained = alpha * np.dot(b, r)
        confidence = max(0.0, explained) / b_energy
        est = alpha * r if confidence >= min_confidence else np.zeros_like(b)
        residual = b - est
        w = win[:len(b)]
        acc[pos:b_end] += residual * w
        wsum[pos:b_end] += w
        pos += hop

    wsum[wsum < 1e-9] = 1.0
    out[span_lo:span_hi] = acc[span_lo:span_hi] / wsum[span_lo:span_hi]
    return out


def solve_activation(mag_block, W, n_iter=150, H_init=None):
    F, K = W.shape
    T = mag_block.shape[1]
    if H_init is not None and H_init.shape[1] >= T:
        H = np.maximum(H_init[:, -T:].copy(), 1e-3)
    else:
        rng = np.random.default_rng(0)
        H = rng.random((K, T)) + 1e-3
    eps = 1e-9
    for _ in range(n_iter):
        WH = W @ H + eps
        H *= (W.T @ (mag_block / WH)) / (W.T @ np.ones_like(mag_block) + eps)
    return H


def nmf_separate_block(mag_block, Wv, Wn, mask_power=1.0, H_init=None, n_iter=150):
    W = np.concatenate([Wv, Wn], axis=1)
    H = solve_activation(mag_block, W, n_iter=n_iter, H_init=H_init)
    Kv = Wv.shape[1]
    Hv, Hn = H[:Kv], H[Kv:]
    voice_est = Wv @ Hv
    noise_est = Wn @ Hn
    mask = (voice_est ** mask_power) / (voice_est ** mask_power + noise_est ** mask_power + 1e-9)
    return mask, H


def nmf_residual_cleanup(signal, Wv, Wn, proc_start, proc_end, sr, n_fft=512, hop=128,
                          block_ms=16, mask_power=1.5, extra_context_blocks=1,
                          smooth_frames=5, mask_floor=0.25, activation_iters=150):
    n = len(signal)
    _, _, Z = stft(signal, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    mag = np.abs(Z)
    phase = np.angle(Z)
    n_frames = mag.shape[1]

    frame_proc_start = max(0, int(proc_start / hop) - extra_context_blocks)
    frame_proc_end = min(n_frames, int(proc_end / hop) + 1)
    frames_per_block = max(1, int(block_ms / (1000 * hop / sr)))
    mask_full = np.ones_like(mag)

    ctx = frames_per_block * extra_context_blocks
    H_prev = None
    for start in range(frame_proc_start, frame_proc_end, frames_per_block):
        end = min(frame_proc_end, start + frames_per_block)
        ctx_start = max(frame_proc_start, start - ctx)
        block = mag[:, ctx_start:end]
        m, H_prev = nmf_separate_block(block, Wv, Wn, mask_power=mask_power,
                                        H_init=H_prev, n_iter=activation_iters)
        mask_full[:, start:end] = m[:, (start - ctx_start):]

    mask_full = mask_floor + (1 - mask_floor) * mask_full

    if smooth_frames > 1 and frame_proc_end > frame_proc_start:
        pad = smooth_frames // 2
        lo_p, hi_p = max(0, frame_proc_start - pad), min(n_frames, frame_proc_end + pad)
        mask_full[:, lo_p:hi_p] = median_filter(mask_full[:, lo_p:hi_p], size=(1, smooth_frames))

    Z_masked = mag * mask_full * np.exp(1j * phase)
    _, cleaned = istft(Z_masked, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
    cleaned = cleaned[:n] if len(cleaned) >= n else np.pad(cleaned, (0, n - len(cleaned)))

    out = signal.copy()
    out[proc_start:proc_end] = cleaned[proc_start:proc_end]
    return out, mask_full


def apply_edge_refinement(signal, onset, offset, sr, edge_ms=8):
    # (fixed) onset/offset can land outside the signal when a reference
    # recording is longer than the local mixture -- clamp `center` before
    # computing the crossfade window, otherwise `hi - lo` can go negative
    # and np.ones() raises "negative dimensions are not allowed".
    n = len(signal)
    g = np.ones(n)
    edge_len = max(1, int(sr * edge_ms / 1000))

    def crossfade_down(center, length):
        center = int(np.clip(center, 0, n))
        lo = max(0, center - length)
        hi = min(n, center + length)
        if hi <= lo:
            return
        seg = np.ones(hi - lo)
        left_len = center - lo
        right_len = hi - center
        if left_len > 0:
            seg[:left_len] = np.linspace(1.0, 0.05, left_len)
        if right_len > 0:
            seg[left_len:] = np.linspace(0.05, 1.0, right_len)
        g[lo:hi] = np.minimum(g[lo:hi], seg)

    crossfade_down(onset, edge_len)
    crossfade_down(offset, edge_len)

    event_span = np.zeros(n, dtype=bool)
    event_span[max(0, min(n, onset)):max(0, min(n, offset))] = True
    return g, event_span


def adaptive_gain(signal, voice_conf, event_span, boost_db=5.0):
    boost = 10 ** (boost_db / 20)
    gain = np.ones_like(signal)
    apply_here = voice_conf & (~event_span)
    gain[apply_here] = boost
    return signal * gain


def run_pipeline(mixture, sr, Wv, Wn, noise_ref, n_fft=512, hop=128,
                  block_ms=16, mask_power=1.5, edge_ms=8, boost_db=5.0,
                  voice_conf_thresh=0.5, extra_context_blocks=1, smooth_frames=5,
                  event_guard_ms=80, activation_iters=150,
                  cancel_block_ms=20, cancel_overlap=0.75, mask_floor=0.25,
                  cancel_min_confidence=0.3, ref_library=None):
    """Defaults here are the best config found by sweeping "pacing" (block
    sizes, context, smoothing) on real audio -- see the module docstring
    for the measured numbers. `noise_ref` is required (also tried as part
    of the library); `ref_library` is the rest of your known recordings
    for this noise class."""
    n = len(mixture)

    onset, offset, lib_ref, match_confidence = locate_event(mixture, noise_ref, sr, ref_library=ref_library)
    guard = int(sr * event_guard_ms / 1000)
    proc_start = max(0, onset - guard)
    proc_end = min(n, offset + guard)

    if lib_ref is not None:
        stage1 = reference_cancel(mixture, lib_ref, onset, offset, sr,
                                   block_ms=cancel_block_ms, overlap=cancel_overlap,
                                   min_confidence=cancel_min_confidence)
    else:
        stage1 = mixture.copy()

    stage2, mask_full = nmf_residual_cleanup(
        stage1, Wv, Wn, proc_start, proc_end, sr, n_fft=n_fft, hop=hop,
        block_ms=block_ms, mask_power=mask_power, extra_context_blocks=extra_context_blocks,
        smooth_frames=smooth_frames, mask_floor=mask_floor, activation_iters=activation_iters)

    edge_gain, event_span = apply_edge_refinement(stage2, onset, offset, sr, edge_ms=edge_ms)
    refined = stage2 * edge_gain

    mask_energy = mask_full.mean(axis=0)
    sample_conf = np.interp(np.arange(n), np.arange(len(mask_energy)) * hop, mask_energy)
    voice_conf_full = np.zeros(n, dtype=bool)
    voice_conf_full[proc_start:proc_end] = sample_conf[proc_start:proc_end] > voice_conf_thresh

    final = adaptive_gain(refined, voice_conf_full, event_span, boost_db=boost_db)

    info = dict(onset=onset, offset=offset, lib_ref_used=lib_ref is not None,
                match_confidence=match_confidence,
                event_span=event_span, voice_conf=voice_conf_full, mask_full=mask_full,
                proc_start=proc_start, proc_end=proc_end)
    return final, info
