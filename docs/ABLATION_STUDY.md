# Ablation Study — AURA@VCET (Team ID 152043)

# 1. Problem Statement and Team Details

| **Problem Statement ID**    | SIH26052                                                                                                                                                                                                                                                |
|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Problem Statement Title** | To develop an AI/ML-enabled adaptive noise cancellation (ANC) system that effectively suppresses stationary, non-stationary, and impulsive defence noises while maintaining high speech intelligibility and real-time performance on embedded hardware. |
| **Theme**                   | Miscellaneous                                                                                                                                                                                                                                           |
| **PS Category**             | Hardware                                                                                                                                                                                                                                                |
| **Team ID**                 | 152043                                                                                                                                                                                                                                                  |
| **Team Name**               | AURA@VCET                                                                                                                                                                                                                                               |

# 2. Evaluation Metrics

| **Metric** | **Full Form**                              | **What It Measures**                                                                                                                                                                                                    | **Why It Matters to AURA**                                                                                                                                   |
|------------|--------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SNR**    | Signal-to-Noise Ratio                      | Measures the ratio between the desired speech signal power and the remaining noise/interference power. It is expressed in decibels (dB). A higher SNR indicates that the recovered speech is cleaner relative to noise. | For AURA, SNR indicates how effectively stationary, non-stationary, and impulsive noise components are suppressed while retaining the speech signal.         |
| **STOI**   | Short-Time Objective Intelligibility       | An objective measure of speech intelligibility, generally reported on a scale from 0 to 1. Higher values indicate that speech content is more understandable.                                                           | STOI is especially important because aggressive noise suppression can make audio quieter while also removing speech information. The AURA target is \> 0.85. |
| **PESQ**   | Perceptual Evaluation of Speech Quality    | An objective perceptual measure used to estimate the perceived quality of processed speech. Higher values indicate better perceived speech quality within the applicable PESQ scale.                                    | PESQ helps verify that noise suppression does not produce excessive distortion, musical noise, muffling, or other artifacts. The AURA target is \> 2.5.      |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio | Measures the separation between the target speech and distortion/error in a scale-invariant manner. Higher values generally indicate better recovery of the target speech.                                              | SI-SDR is used as a supporting metric to examine source recovery quality alongside SNR, STOI, and PESQ.                                                      |

# 

# 3. Fixed Acceptance Constraints

The following thresholds are fixed for this project and are not changed
between frameworks:

| **Metric**  | **Fixed Constraint** | **Primary Purpose**                    |
|-------------|----------------------|----------------------------------------|
| Output SNR  | \> 15 dB             | Noise suppression / signal cleanliness |
| Output STOI | \> 0.85              | Speech intelligibility preservation    |
| Output PESQ | \> 2.5               | Perceptual speech quality              |

# 

# 4. Ablation Study Results

The Ablation Study is done using State of the Art Model (SOTA) for our
dataset. The reported values below are taken directly from the supplied
evaluation outputs. A metric is marked PASS only when the output value
is strictly above the fixed threshold.

| **Framework**                    | **Input SNR** | **Output SNR** | **Δ SNR** | **Input STOI** | **Output STOI** | **Input PESQ** | **Output PESQ** | **Noise** | **SNR**  | **STOI** | **PESQ** | **All 3** |
|----------------------------------|---------------|----------------|-----------|----------------|-----------------|----------------|-----------------|-----------|----------|----------|----------|-----------|
| **BSRNN Large Non-Causal**       | 4.278         | 17.495         | 13.217    | 0.886          | 0.9857          | 1.4889         | 3.1139          | 13.877    | **PASS** | **PASS** | **PASS** | **PASS**  |
| **TF-GridNet**                   | 0.000         | 4.993          | 4.993     | 0.781          | 0.8643          | 1.2354         | 1.3011          | 4.993     | **FAIL** | **PASS** | **FAIL** | **FAIL**  |
| **μNet / U-Net Dataset-Adapted** | -3.336        | 0.806          | 4.142     | 0.518          | 0.5156          | 1.1290         | 1.1123          | 5.740     | **FAIL** | **FAIL** | **FAIL** | **FAIL**  |
| **GTCRN**                        | -3.336        | -4.159         | -0.823    | 0.518          | 0.3515          | 1.1290         | 1.0632          | -0.545    | **FAIL** | **FAIL** | **FAIL** | **FAIL**  |
| **DeepFilterNet3**               | -5.000        | 0.272          | 5.272     | 0.688          | 0.7742          | 1.0864         | 1.2265          | 5.272     | **FAIL** | **FAIL** | **FAIL** | **FAIL**  |
| **Proposed AURA Framework**      | 3.560         | 18.106         | 14.546    | 0.945          | 0.9888          | 1.2482         | 2.8670          | 14.543    | **PASS** | **PASS** | **PASS** | **PASS**  |

# 5. Supporting SI-SDR Results

| **Framework**                | **Input SI-SDR (dB)** | **Output SI-SDR (dB)** | **SI-SDR Improvement (dB)** |
|------------------------------|-----------------------|------------------------|-----------------------------|
| BSRNN Large Non-Causal       | 4.297                 | 17.779                 | 13.482                      |
| TF-GridNet                   | -0.046                | 4.511                  | 4.557                       |
| μNet / U-Net Dataset-Adapted | -3.382                | -3.548                 | -0.166                      |
| GTCRN                        | -3.382                | -16.516                | -13.134                     |
| DeepFilterNet3               | -5.082                | -0.227                 | 4.855                       |
| Proposed AURA Framework      | 1.038                 | 18.038                 | 17.001                      |

# 6. Framework-Wise Ablation Interpretation

**BSRNN Large Non-Causal:** The recovered output reaches 17.495 dB SNR,
0.9857 STOI, and 3.1139 PESQ. It therefore exceeds all three fixed
acceptance constraints. The supplied result also reports 13.217 dB SNR
improvement, 13.482 dB SI-SDR improvement, and 13.877 dB gunshot
reduction. The non-causal configuration is evaluated here as a reference
enhancement framework; its non-causal nature is relevant when
considering deployment latency for real-time embedded operation.

**TF-GridNet:** The output improves SNR from 0.000 dB to 4.993 dB and
STOI from 0.7816 to 0.8643. However, the output SNR remains below 15 dB
and PESQ remains below 2.5, so it does not satisfy the complete fixed
acceptance set. The supplied result reports 4.993 dB gunshot reduction.

**μNet / U-Net Dataset-Adapted:** The output SNR improves to 0.806 dB,
but STOI changes from 0.5181 to 0.5156 and PESQ changes from 1.1290 to
1.1123. It therefore does not satisfy the fixed SNR, STOI, or PESQ
constraints. The supplied result reports 5.740 dB gunshot reduction.

**GTCRN:** The supplied output shows a decrease in SNR from -3.336 dB to
-4.159 dB, a decrease in STOI from 0.5181 to 0.3515, and a decrease in
PESQ from 1.1290 to 1.0632. It does not satisfy the fixed constraints.
The reported gunshot reduction value is -0.545 dB, indicating that this
evaluation did not show positive gunshot attenuation under the supplied
test condition.

**DeepFilterNet3:** The output SNR improves from -5.000 dB to 0.272 dB,
while STOI improves from 0.6881 to 0.7742 and PESQ improves from 1.0864
to 1.2265. These improvements are measurable, but the output remains
below all three fixed acceptance constraints. The supplied result
reports 5.272 dB gunshot reduction.

**Proposed AURA Framework:** The proposed AURA framework produces 18.106
dB output SNR, 0.9888 output STOI, and 2.8670 output PESQ. These values
exceed all three fixed acceptance constraints. The supplied evaluation
also reports 14.546 dB SNR improvement, 17.001 dB SI-SDR improvement,
and 14.543 dB gunshot reduction. The result demonstrates the intended
balance between noise suppression, speech intelligibility, and
perceptual quality rather than optimizing a single metric in isolation.

# 7. Overall Evaluation Summary

The ablation study uses three fixed project constraints: SNR \> 15 dB,
STOI \> 0.85, and PESQ \> 2.5. Under the supplied evaluation results,
the BSRNN Large Non-Causal reference and the proposed AURA framework
both exceed all three thresholds. The other evaluated frameworks do not
satisfy the complete three-metric acceptance set under their supplied
test results. The AURA result reaches 18.106 dB SNR, 0.9888 STOI, and
2.8670 PESQ, together with 17.001 dB SI-SDR improvement and 14.543 dB
reported gunshot reduction.

**Importantly, the ablation is interpreted as a multi-objective
evaluation. The purpose is not simply to maximize noise attenuation; the
processed audio must also preserve speech intelligibility and perceptual
quality.**

# 8. Evaluation Notes and Reproducibility

• All numerical values in this document are transcribed from the
evaluation outputs supplied for the respective frameworks.

• Input and output metrics are reported as provided; no recalculation or
normalization of the supplied values has been applied.

• The three acceptance constraints are fixed throughout this document:
SNR \> 15 dB, STOI \> 0.85, and PESQ \> 2.5.

• A framework can show improvement over its input while still failing
the project acceptance threshold. Improvement and threshold compliance
are therefore reported separately.

• Real-time embedded suitability requires additional measurements such
as inference latency, memory footprint, computational load, and power
consumption. These measurements are not included in the supplied metric
logs and should be reported separately when available.
