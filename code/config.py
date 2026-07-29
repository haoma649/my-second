"""Configuration for the LEO multibeam uplink reproduction scaffold.

Values marked PAPER are explicitly reported in Hu et al. (2024),
"Multi-dimensional resource allocation strategy for LEO satellite
communication uplinks based on deep reinforcement learning".

Values marked ASSUMPTION are not clearly specified in the paper and should be
treated as tunable hypotheses for sensitivity analysis.
"""

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


@dataclass
class SimulationConfig:
    # PAPER: Table 2.
    num_beams: int = 37
    num_channels: int = 16
    default_num_users: int = 200
    satellite_altitude_km: float = 780.0
    max_tx_power_dbw: float = 20.0
    min_sinr_db: float = 3.0
    channel_bandwidth_hz: float = 1e6
    free_space_loss_db: float = 212.0
    tx_antenna_gain_db: float = 40.0
    rx_antenna_gain_db: float = 50.0

    # PAPER: two objective-weight settings used in simulation.
    weights_equal: Tuple[float, float, float] = (1.0 / 3, 1.0 / 3, 1.0 / 3)
    weights_blocking: Tuple[float, float, float] = (1.0 / 4, 1.0 / 4, 1.0 / 2)

    # ASSUMPTION: the paper says power is divided into multiple domains but
    # does not specify the number or exact levels.
    num_power_levels: int = 5
    min_tx_power_dbw: float = 0.0

    # ASSUMPTION: receiver noise model is not specified. This uses a standard
    # thermal-noise density plus a receiver noise figure.
    thermal_noise_dbm_per_hz: float = -174.0
    noise_figure_db: float = 3.0

    # ASSUMPTION: the paper gives antenna formulas but not user angles or beam
    # layout coordinates. We approximate a 37-beam hexagonal layout and use
    # distance-dependent co-channel coupling in dB.
    interference_coupling_db: Dict[int, float] = field(
        default_factory=lambda: {1: -12.0, 2: -18.0, 3: -25.0, 4: -32.0}
    )
    max_interference_ring: int = 4

    # ASSUMPTION: traffic generation details beyond "Poisson distribution" are
    # incomplete. These settings make distributions configurable.
    traffic_distribution: str = "poisson"  # poisson | uniform | hotspot
    hotspot_fraction: float = 0.5
    hotspot_beams: Tuple[int, ...] = (0, 1, 2, 3)

    # ASSUMPTION: optional heterogeneous services for upgrade experiments.
    service_sinr_thresholds_db: Dict[str, float] = field(
        default_factory=lambda: {"voice": 1.0, "iot": 0.0, "video": 5.0, "default": 3.0}
    )
    service_rate_min_mbps: Dict[str, float] = field(
        default_factory=lambda: {"voice": 0.064, "iot": 0.01, "video": 2.0, "default": 0.5}
    )

    # ASSUMPTION: normalization reference values for reward. Keep fixed across
    # experiments to avoid leakage from future/evaluation data.
    se_norm_ref: float = 400.0
    ee_norm_ref: float = 120.0

    strict_qos_admission: bool = True


@dataclass
class TrainingConfig:
    # PAPER: Table 2.
    replay_capacity: int = 10000
    gamma: float = 0.9
    learning_rate: float = 0.001
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01

    # ASSUMPTION: not explicitly specified in the paper.
    batch_size: int = 64
    train_every: int = 4
    target_update_interval: int = 100
    train_episodes: int = 250
    max_steps_per_episode: int = 200
    epsilon_decay_steps: int = 5000
    seed: int = 42
    device: str = "cpu"


@dataclass
class ExperimentConfig:
    users: List[int] = field(default_factory=lambda: [25, 50, 75, 100, 125, 150, 175, 200])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    traffic: List[str] = field(default_factory=lambda: ["uniform", "poisson", "hotspot"])
    eval_episodes: int = 50


USER_COUNTS = [25, 50, 75, 100, 125, 150, 175, 200]
WEIGHT_PRESETS = {
    "equal": SimulationConfig().weights_equal,
    "blocking": SimulationConfig().weights_blocking,
}


def _filter_dataclass_kwargs(cls, values):
    allowed = {f.name for f in fields(cls)}
    return {k: v for k, v in values.items() if k in allowed}


def _load_mapping(path: str):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_configs(path: str | None):
    """Load YAML config while preserving dataclass defaults.

    The YAML schema intentionally mirrors paper/report terminology. A few
    aliases are accepted so config files can use concise names such as
    `bandwidth_mhz` and `fspl_db`.
    """
    sim = SimulationConfig()
    train = TrainingConfig()
    exp = ExperimentConfig()
    if not path:
        return sim, train, exp

    raw = _load_mapping(path)
    env_raw = dict(raw.get("env", {}))
    if "bandwidth_mhz" in env_raw:
        env_raw["channel_bandwidth_hz"] = float(env_raw.pop("bandwidth_mhz")) * 1e6
    if "fspl_db" in env_raw:
        env_raw["free_space_loss_db"] = env_raw.pop("fspl_db")
    sim = SimulationConfig(**{**sim.__dict__, **_filter_dataclass_kwargs(SimulationConfig, env_raw)})

    train_raw = dict(raw.get("training", {}))
    aliases = {
        "episodes": "train_episodes",
        "lr": "learning_rate",
        "replay_size": "replay_capacity",
        "target_update": "target_update_interval",
    }
    for old, new in aliases.items():
        if old in train_raw:
            train_raw[new] = train_raw.pop(old)
    train = TrainingConfig(**{**train.__dict__, **_filter_dataclass_kwargs(TrainingConfig, train_raw)})

    exp_raw = dict(raw.get("experiment", {}))
    exp = ExperimentConfig(**{**exp.__dict__, **_filter_dataclass_kwargs(ExperimentConfig, exp_raw)})
    return sim, train, exp
