# 🧠 Architecture — MARK7-DM

![pipeline](pipeline_overview.png)

## 1. Data: voice and noise pools
* **Voice pool** — the team's own recordings + LibriSpeech speakers (train / val / test split by speaker).
* **Noise pool** — gunshots (one folder per weapon) and explosions, split so test events are never seen in training.
* **Simulator** (`code/aura/physmix.py`, `code/aura/dualmic.py`) mixes them at random distances, rooms (reverb),
  signal-to-noise ratios and overlap positions, and renders **both** microphones: the reference mic hears the voice 10–18 dB
  weaker and its own copy of the blast.

## 2. Dictionaries and library
* **Voice / noise dictionaries** (`weights/dicts_m65.pt`, `code/learn_dicts.py`) — spectral patterns learned from
  the pools; the single-mic MARK7 stage uses them to separate what is voice from what is noise.
* **Blast library** (`mark7_bundle/library/`) — recorded events used to recognise a known blast as it starts.

## 3. Features
* **STFT** — 512-point window (32 ms) every 128 samples (8 ms), 257 bins, both mics.
* **DSRA — Dynamic Spectral Residual Analysis** (`code/aura/model_dm3.py: dsra_vec`) — 8 descriptors per frame
  per mic: spectral centroid, flatness, roll-off, low / speech / high band shares, loudness, loudness jump.
* **Speaker voiceprint** (`code/aura/model_dm2.py: voiceprint`) — 80-d log-mel mean / std from frames that are
  blast-free (gate) and voiced (DSRA).

## 4. Network — MARK7-DM v3 (`code/aura/model_dm3.py`)
* MARK6.5 core (causal, deep filter + recoverability head), 1.7 M parameters.
* Two inputs: voice mic `X` and reference mic `R`; 2 frames (16 ms) of lookahead.
* FiLM conditioning from DSRA and the voiceprint, so the network knows whose voice to keep.
* Neural reference canceller: predicts the blast from `R` every frame and subtracts it.
* Output: voice estimate `Y`.

## 5. Post-processing (`code/aura/ref_cancel.py`, `mark7_bundle/run_mark7.py: Mark7V4`)
* **Reference-mic post-filter** — subband NLMS (4 complex taps per bin) on the network output `Y`, adaptation slowed where the network hears voice → residual estimate `E`.
* **Attenuation-only gain** — `Z = Y · min(1, |E| / |Y|)^0.5`: removes residual blast energy, never boosts.
* **iSTFT + loudness** — overlap-add, −18 LUFS, peak limiter (`code/aura/loudness.py`).

## Model lineage
MARK6.5 (1-mic) → MARK7 (gate, library, limiter, dictionary) → MARK7-DM (+ reference mic)
→ v2 (+ lookahead, voiceprint) → v3 (+ DSRA) → v4 (+ reference-mic post-filter).
