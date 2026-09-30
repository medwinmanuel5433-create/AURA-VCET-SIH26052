"""AURA-MARK6 configuration.

Single source of truth for signal, corpus, model and training parameters.
Everything downstream imports from here so a config change cannot silently
desynchronise the corpus generator from the model.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Tuple
import json


@dataclass
class SignalCfg:
    sr: int = 16000
    clip_seconds: float = 7.0
    n_fft: int = 512
    hop: int = 128          # 8 ms hop -> low algorithmic latency
    win: int = 512          # 32 ms window
    # Causal operation: the model never sees a frame later than the current one.
    # Algorithmic latency = win/sr = 32 ms (analysis window only, no lookahead).

    @property
    def n_samples(self) -> int:
        return int(self.sr * self.clip_seconds)

    @property
    def n_freq(self) -> int:
        return self.n_fft // 2 + 1


@dataclass
class ChainCfg:
    """Physical capture-chain parameters (capsule -> preamp/AGC -> ADC)."""
    # soft-clip headroom (linear). Lower = earlier capsule saturation.
    capsule_headroom: Tuple[float, float] = (0.6, 1.4)
    agc_attack_ms: float = 5.0
    agc_release_ms: float = 250.0
    agc_target_rms: float = 0.06
    agc_max_gain_db: float = 24.0
    agc_min_gain_db: float = -30.0
    # Air absorption low-pass corner as a function of distance
    air_lp_hz_near: float = 7800.0
    air_lp_hz_far: float = 2600.0


@dataclass
class CorpusCfg:
    """The situation grid. Every axis here is varied per generated example."""
    event_types: List[str] = field(
        default_factory=lambda: ["none", "gunshot", "explosion", "both"])
    # event-window SNR in dB (speech vs event, measured inside the event span)
    snr_db: List[float] = field(
        default_factory=lambda: [-15, -10, -5, 0, 5, 10, 15, 20])
    overlap: List[str] = field(
        default_factory=lambda: ["none", "onset", "mid", "full"])
    distance: List[str] = field(
        default_factory=lambda: ["near", "mid", "far"])
    room: List[str] = field(
        default_factory=lambda: ["anechoic", "small", "medium", "large"])
    chain: List[str] = field(
        default_factory=lambda: ["clean", "moderate", "severe"])

    n_train: int = 1200
    n_val: int = 300
    n_test: int = 300

    # Held-out-by-construction splits. Weapons never shared across splits.
    test_weapons: List[str] = field(
        default_factory=lambda: ["Zastava M92", "MG-42"])
    val_weapons: List[str] = field(default_factory=lambda: ["MP5"])

    seed: int = 1234


@dataclass
class ModelCfg:
    # band-split definition: (width_in_bins, count)
    band_plan: List[Tuple[int, int]] = field(
        default_factory=lambda: [(4, 16), (8, 8), (16, 4), (32, 2), (1, 1)])
    width: int = 64            # per-band feature dim D
    n_blocks: int = 4          # dual-path blocks
    rnn_hidden: int = 64
    df_order: int = 5          # deep-filter taps (causal, includes current frame)
    cond_dim: int = 32         # FiLM conditioning vector width
    spk_dim: int = 32          # speaker reference embedding width

    def total_bins(self) -> int:
        return sum(w * c for w, c in self.band_plan)

    def n_bands(self) -> int:
        return sum(c for _, c in self.band_plan)


@dataclass
class TrainCfg:
    batch_size: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 30
    grad_clip: float = 5.0
    # loss weights
    w_ri: float = 1.0          # complex real/imag on recoverable bins
    w_mag: float = 1.0         # log-magnitude
    w_wav: float = 0.5         # time-domain
    w_sisdr: float = 0.3
    w_mrstft: float = 0.5
    w_det: float = 0.2         # event detector BCE
    # Raised from 2.0 after run 3: clean-clip passthrough SNR measured 21.4 dB,
    # i.e. the model was audibly reworking audio that contained no event.
    w_pass: float = 6.0        # clean passthrough preservation
    w_inpaint: float = 0.3     # magnitude inpainting on unrecoverable bins
    w_rec: float = 0.5         # recoverability prediction (drives the blend)
    # recoverability curve: w = sigmoid((snr_db + center)/scale)
    rec_center_db: float = 12.0
    rec_scale_db: float = 6.0


@dataclass
class Cfg:
    sig: SignalCfg = field(default_factory=SignalCfg)
    chain: ChainCfg = field(default_factory=ChainCfg)
    corpus: CorpusCfg = field(default_factory=CorpusCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    def dump(self, path):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


DEFAULT = Cfg()
