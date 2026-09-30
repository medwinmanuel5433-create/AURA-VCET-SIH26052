"""Audio I/O and resampling.

Handles the real dataset formats: AAC/.m4a and Opus/.ogg voice notes,
16/24-bit and 8-bit PCM wavs, mono and stereo, 16k/44.1k/48k.
Everything is decoded through ffmpeg to float32 mono at the target rate so
downstream code never has to think about source format again.
"""
import os
import subprocess
import numpy as np


def load_audio(path: str, sr: int = 16000, mono: bool = True) -> np.ndarray:
    """Decode any ffmpeg-readable file to float32 mono at `sr`."""
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1" if mono else "2",
        "-ar", str(sr), "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {path}: {proc.stderr.decode()[:300]}")
    x = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    if x.size == 0:
        raise RuntimeError(f"empty decode: {path}")
    return x


def save_wav(path: str, x: np.ndarray, sr: int = 16000):
    """Write float32 [-1,1] as 16-bit PCM wav."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16)
    import wave
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def read_wav(path: str) -> tuple:
    import wave
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32767.0
    return x, sr


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2) + 1e-12))


def set_rms(x: np.ndarray, target: float) -> np.ndarray:
    cur = rms(x)
    if cur < 1e-9:
        return x
    return x * (target / cur)


def fit_length(x: np.ndarray, n: int, mode: str = "loop") -> np.ndarray:
    """Trim or extend x to exactly n samples."""
    if len(x) == n:
        return x
    if len(x) > n:
        start = np.random.randint(0, len(x) - n + 1)
        return x[start:start + n]
    if mode == "loop":
        reps = int(np.ceil(n / len(x)))
        return np.tile(x, reps)[:n]
    out = np.zeros(n, dtype=np.float32)
    out[:len(x)] = x
    return out


def active_speech_mask(x: np.ndarray, sr: int, frame_ms: float = 20.0,
                       thresh_db: float = -45.0) -> np.ndarray:
    """Cheap energy VAD returning a per-sample boolean mask."""
    fl = int(sr * frame_ms / 1000)
    n_fr = len(x) // fl
    if n_fr == 0:
        return np.ones(len(x), dtype=bool)
    frames = x[:n_fr * fl].reshape(n_fr, fl)
    e = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-12)
    peak = np.max(e)
    act = e > (peak + thresh_db)
    mask = np.repeat(act, fl)
    out = np.zeros(len(x), dtype=bool)
    out[:len(mask)] = mask
    return out
