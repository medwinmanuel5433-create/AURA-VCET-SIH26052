# 📊 Evaluation

**Targets (fixed):** SNR > 15 dB · STOI > 0.85 · PESQ > 2.5

| Metric | How it is measured |
|---|---|
| SNR | Level-aligned SNR inside the noise-event window (output gain-aligned to the clean voice first) |
| STOI | Short-Time Objective Intelligibility vs the clean voice (`pystoi`) |
| PESQ | Wide-band PESQ vs the clean voice (`pesq`) |
| SI-SDR | Scale-invariant signal-to-distortion ratio (supporting metric) |
| Noise heard | Share of the input noise energy left in the output |

## Test conditions
* **Current headset** (`corpus_dual_eval`): boom mic with automatic gain, outer reference mic.
* **Hardware scenario** (`corpus_hw_eval`): raw boom mic (gain applied after cleaning), close reference mic.
* Splits: `test` (LibriSpeech dev-clean speakers) and `field` (the team's own voice recordings).
  Test gunshots / explosions are never used in training.

## Ablation against state-of-the-art frameworks
See [`ABLATION_STUDY.md`](ABLATION_STUDY.md) and [`Ablation_Study_AURA_VCET.docx`](Ablation_Study_AURA_VCET.docx).

| Framework | Output SNR (dB) | Output STOI | Output PESQ | All 3 targets |
|---|---|---|---|---|
| BSRNN Large (non-causal) | 17.495 | 0.9857 | 3.1139 | PASS |
| TF-GridNet | 4.993 | 0.8643 | 1.3011 | FAIL |
| μNet / U-Net (dataset-adapted) | 0.806 | 0.5156 | 1.1123 | FAIL |
| GTCRN | −4.159 | 0.3515 | 1.0632 | FAIL |
| DeepFilterNet3 | 0.272 | 0.7742 | 1.2265 | FAIL |
| **Proposed AURA** | **18.106** | **0.9888** | **2.8670** | **PASS** |

## Component ablation (reproducible with this repository)
Mean over 60 field clips. Cells: SNR (dB) / STOI / PESQ and the share of clips passing all three targets.

| Configuration | SNR / STOI / PESQ | Pass all 3 |
|---|---|---|
| Noisy input (current headset) | 8.0 / 0.908 / 1.83 | 8 % |
| MARK7 (single mic) | 10.5 / 0.916 / 2.10 | 10 % |
| MARK7-DM v2 (+ reference mic, lookahead, voiceprint) | 12.8 / 0.941 / 2.48 | 25 % |
| MARK7-DM v3 (+ DSRA cues) | 13.0 / 0.944 / 2.53 | 30 % |
| Noisy input (hardware scenario) | 10.7 / 0.926 / 1.86 | 13 % |
| MARK7-DM v3 (hardware scenario) | 16.6 / 0.956 / 2.58 | 42 % |
| **MARK7-DM v4 (hardware scenario)** | **15.8 / 0.956 / 2.64** | **47 %** |

## Where results appear
* Stage C → `runs/lv3_test.json`, `runs/lv3_field.json`
* Stage D → `runs/hw_test.json`, `runs/hw_field.json`
* Stage E → demo audio sets + `/content/AURA_results.zip`
* Stage F → 5 PASS + 5 FAIL cases, each a different signal → `/content/AURA_PASS_FAIL_CASES.zip`
