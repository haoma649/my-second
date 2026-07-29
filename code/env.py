"""LEO multibeam uplink environment for joint channel-power allocation."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import SimulationConfig


@dataclass
class User:
    beam: int
    service: str = "default"
    sinr_threshold_db: float = 3.0
    rate_min_mbps: float = 0.5


class LEOEnv:
    """Sequential admission/channel/power environment.

    The paper treats the satellite as the agent, beams/users as environment
    state, and available channel plus terminal transmit power as actions. This
    class follows that abstraction while making unspecified details explicit.
    """

    def __init__(
        self,
        sim_cfg: Optional[SimulationConfig] = None,
        num_users: Optional[int] = None,
        weights: Tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        seed: int = 42,
        traffic_distribution: Optional[str] = None,
        reconstruction_rings: int = 4,
    ) -> None:
        self.cfg = sim_cfg or SimulationConfig()
        self.num_users = num_users or self.cfg.default_num_users
        self.weights = weights
        self.reconstruction_rings = reconstruction_rings
        self.rng = np.random.default_rng(seed)
        self.py_rng = random.Random(seed)
        self.traffic_distribution = traffic_distribution or self.cfg.traffic_distribution

        self.coords = self._make_hex_coords(radius=3)
        if len(self.coords) != self.cfg.num_beams:
            raise ValueError("37-beam layout generation failed.")
        self.dist_matrix = self._beam_distances()
        self.interference_coupling_linear = self._interference_coupling_matrix()
        self.power_levels_dbw = np.linspace(
            self.cfg.min_tx_power_dbw,
            self.cfg.max_tx_power_dbw,
            self.cfg.num_power_levels,
        )
        self.num_actions = 1 + self.cfg.num_channels * self.cfg.num_power_levels
        self.state_shape = (4, self._max_reconstructed_beams(), self.cfg.num_channels)
        self.reset(seed=seed)

    def reset(self, num_users: Optional[int] = None, seed: Optional[int] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.py_rng = random.Random(seed)
        if num_users is not None:
            self.num_users = num_users
        self.W = np.zeros((self.cfg.num_beams, self.cfg.num_channels), dtype=np.float32)
        self.power_dbw = np.full((self.cfg.num_beams, self.cfg.num_channels), np.nan, dtype=np.float32)
        self.user_threshold_db = np.full((self.cfg.num_beams, self.cfg.num_channels), self.cfg.min_sinr_db, dtype=np.float32)
        self.user_rate_min_mbps = np.zeros((self.cfg.num_beams, self.cfg.num_channels), dtype=np.float32)
        self.blocked_users = 0
        self.blocked_reject_action = 0
        self.blocked_occupied_channel = 0
        self.blocked_qos = 0
        self.accepted_by_beam = np.zeros(self.cfg.num_beams, dtype=np.float32)
        self.blocked_by_beam = np.zeros(self.cfg.num_beams, dtype=np.float32)
        self.current_index = 0
        self.users = self._generate_users(self.num_users)
        self.U = self._traffic_distribution_matrix()
        self.prev_Z = self.compute_objective()
        return self.get_state()

    def action_to_tuple(self, action: int) -> Tuple[int, int]:
        """Return (channel, power_index); (-1, -1) means reject allocation."""
        if action == 0:
            return -1, -1
        idx = action - 1
        return idx // self.cfg.num_power_levels, idx % self.cfg.num_power_levels

    def tuple_to_action(self, channel: int, power_index: int) -> int:
        return 1 + channel * self.cfg.num_power_levels + power_index

    def step(self, action: int):
        if self.current_index >= len(self.users):
            return self.get_state(), 0.0, True, self.metrics()

        user = self.users[self.current_index]
        before_Z = self.prev_Z
        blocked_this_step = False
        channel, power_idx = self.action_to_tuple(action)

        if action == 0:
            self.blocked_users += 1
            self.blocked_reject_action += 1
            self.blocked_by_beam[user.beam] += 1.0
            blocked_this_step = True
        elif self.W[user.beam, channel] > 0:
            self.blocked_users += 1
            self.blocked_occupied_channel += 1
            self.blocked_by_beam[user.beam] += 1.0
            blocked_this_step = True
        else:
            snapshot = self._snapshot()
            self.W[user.beam, channel] = 1.0
            self.power_dbw[user.beam, channel] = self.power_levels_dbw[power_idx]
            self.user_threshold_db[user.beam, channel] = user.sinr_threshold_db
            self.user_rate_min_mbps[user.beam, channel] = user.rate_min_mbps
            if self.cfg.strict_qos_admission and self._active_qos_violations() > 0:
                self._restore(snapshot)
                self.blocked_users += 1
                self.blocked_qos += 1
                self.blocked_by_beam[user.beam] += 1.0
                blocked_this_step = True
            else:
                self.accepted_by_beam[user.beam] += 1.0

        self.current_index += 1
        self.U = self._traffic_distribution_matrix()
        new_Z = self.compute_objective()
        reward = 1.0 if (new_Z - before_Z) > 0 else 0.0
        if blocked_this_step and action == 0:
            reward = 1.0 if (new_Z - before_Z) > 0 else 0.0
        self.prev_Z = new_Z
        done = self.current_index >= len(self.users)
        return self.get_state(), reward, done, self.metrics()

    def simulate_action(self, action: int) -> Tuple[float, bool, Dict[str, float]]:
        """Evaluate one real `step` and restore the environment afterward.

        This is used by model-based baselines and selectors that need a
        one-step objective score without consuming the current user. It calls
        the same `step` method as normal execution, so QoS admission, blocking
        counters, traffic updates, and objective bookkeeping stay consistent.
        """
        snapshot = self._full_snapshot()
        try:
            _, reward, done, info = self.step(action)
            return float(reward), bool(done), dict(info)
        finally:
            self._full_restore(snapshot)

    def get_action_mask(self, include_qos: bool = False) -> np.ndarray:
        """Return a boolean mask for valid actions.

        `include_qos=False` masks only structurally impossible actions, mainly
        channels already occupied by the current beam. `include_qos=True` also
        simulates candidate allocations and masks actions that immediately
        violate active SINR/rate constraints. The latter is useful for ablation
        but costs more per decision.
        """
        mask = np.ones(self.num_actions, dtype=bool)
        if self.current_index >= len(self.users):
            return mask
        beam = self.users[self.current_index].beam
        for ch in range(self.cfg.num_channels):
            if self.W[beam, ch] > 0:
                for p in range(self.cfg.num_power_levels):
                    mask[self.tuple_to_action(ch, p)] = False
        if include_qos:
            user = self.users[self.current_index]
            snapshot = self._snapshot()
            for action in np.flatnonzero(mask):
                if action == 0:
                    continue
                ch, pidx = self.action_to_tuple(int(action))
                self.W[user.beam, ch] = 1.0
                self.power_dbw[user.beam, ch] = self.power_levels_dbw[pidx]
                self.user_threshold_db[user.beam, ch] = user.sinr_threshold_db
                self.user_rate_min_mbps[user.beam, ch] = user.rate_min_mbps
                if self._active_qos_violations() > 0:
                    mask[action] = False
                self._restore(tuple(x.copy() for x in snapshot))
        return mask

    def valid_action_mask(self) -> np.ndarray:
        """Backward-compatible structural action mask."""
        return self.get_action_mask(include_qos=False)

    def get_state(self) -> np.ndarray:
        if self.current_index >= len(self.users):
            beam = 0
            threshold = self.cfg.min_sinr_db
        else:
            user = self.users[self.current_index]
            beam = user.beam
            threshold = user.sinr_threshold_db
        selected = self._reconstructed_beams(beam)
        channels = self.cfg.num_channels
        state = np.zeros(self.state_shape, dtype=np.float32)
        sinr = self.sinr_db_matrix(fill_value=0.0)
        for row, b in enumerate(selected[: self.state_shape[1]]):
            state[0, row, :] = self.W[b]
            state[1, row, :] = self.U[b] / max(1, self.num_users)
            state[2, row, :] = 1.0 if b == beam else 0.0
            state[3, row, :] = np.nan_to_num(sinr[b], nan=0.0) / 30.0
        state[2, 0, 0] = threshold / 30.0
        return state[:, :, :channels]

    def graph_feature_names(self) -> List[str]:
        """Return node-feature names used by `get_graph_observation`.

        This is a GNN-ready diagnostic interface. It does not change the paper
        reproduction DQN state. Per-service features are generated from the
        configured service dictionaries, so experiments can audit exactly which
        traffic heterogeneity signals are exposed to a future graph encoder.
        """
        return [
            "remaining_user_fraction",
            "accepted_user_fraction",
            "blocked_user_fraction",
            "available_channel_fraction",
            "occupied_channel_fraction",
            "avg_sinr_db_norm",
            "min_sinr_margin_db_norm",
            "current_beam_indicator",
            "beam_power_fraction",
            *[f"channel_{ch}_occupied" for ch in range(self.cfg.num_channels)],
            *[f"channel_{ch}_power_fraction" for ch in range(self.cfg.num_channels)],
            *[f"channel_{ch}_sinr_db_norm" for ch in range(self.cfg.num_channels)],
            *[f"remaining_service_fraction_{name}" for name in self._service_feature_names()],
        ]

    def get_graph_observation(self, include_self_loops: bool = True) -> Dict[str, np.ndarray]:
        """Return graph-structured beam state for future GNN experiments.

        Nodes are beams. Directed edges represent potential co-channel
        interference according to the configured coupling matrix. The returned
        arrays are NumPy-only to avoid requiring PyTorch Geometric at the
        reproduction stage.
        """
        current_beam = self.users[self.current_index].beam if self.current_index < len(self.users) else -1
        active = self.W > 0
        sinr = self.sinr_db_matrix(fill_value=np.nan)
        margins = sinr - self.user_threshold_db
        features = []
        service_counts = self._remaining_service_counts_by_beam()
        max_beam_power_w = self.cfg.num_channels * dbw_to_w(self.cfg.max_tx_power_dbw)
        power_by_beam = np.nansum(self._power_w_matrix() * active, axis=1)
        max_channel_power_w = max(float(dbw_to_w(self.cfg.max_tx_power_dbw)), 1e-12)
        power_fraction = np.nan_to_num(self._power_w_matrix(), nan=0.0) / max_channel_power_w
        channel_sinr_norm = np.nan_to_num(sinr, nan=0.0) / 30.0
        for beam in range(self.cfg.num_beams):
            beam_active = active[beam]
            avg_sinr = float(np.nanmean(sinr[beam, beam_active])) if np.any(beam_active) else 0.0
            min_margin = float(np.nanmin(margins[beam, beam_active])) if np.any(beam_active) else 0.0
            row = [
                self.U[beam] / max(1, self.num_users),
                self.accepted_by_beam[beam] / max(1, self.num_users),
                self.blocked_by_beam[beam] / max(1, self.num_users),
                float(np.sum(~beam_active)) / self.cfg.num_channels,
                float(np.sum(beam_active)) / self.cfg.num_channels,
                avg_sinr / 30.0,
                min_margin / 30.0,
                1.0 if beam == current_beam else 0.0,
                float(power_by_beam[beam]) / max(float(max_beam_power_w), 1e-12),
            ]
            row.extend(active[beam].astype(np.float32).tolist())
            row.extend(power_fraction[beam].astype(np.float32).tolist())
            row.extend(channel_sinr_norm[beam].astype(np.float32).tolist())
            row.extend(service_counts[beam] / max(1, self.num_users))
            features.append(row)

        edge_index, edge_weight = self.graph_edges(include_self_loops=include_self_loops)
        return {
            "node_features": np.asarray(features, dtype=np.float32),
            "edge_index": edge_index,
            "edge_weight": edge_weight,
            "feature_names": np.asarray(self.graph_feature_names(), dtype=object),
        }

    def graph_edges(self, include_self_loops: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return directed beam-interference edges as `(edge_index, weight)`.

        `edge_index` has shape `[2, num_edges]` with source and destination
        beam indices. Edge weights are linear interference-coupling values.
        """
        sources = []
        targets = []
        weights = []
        for src in range(self.cfg.num_beams):
            for dst in range(self.cfg.num_beams):
                if src == dst:
                    if include_self_loops:
                        sources.append(src)
                        targets.append(dst)
                        weights.append(1.0)
                    continue
                weight = float(self.interference_coupling_linear[dst, src])
                if weight > 0.0:
                    sources.append(src)
                    targets.append(dst)
                    weights.append(weight)
        return (
            np.asarray([sources, targets], dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
        )

    def compute_objective(self) -> float:
        m = self.metrics()
        se = min(m["spectral_efficiency"], self.cfg.se_norm_ref) / self.cfg.se_norm_ref
        ee = min(m["energy_efficiency"], self.cfg.ee_norm_ref) / self.cfg.ee_norm_ref
        one_minus_ve = 1.0 - m["blocking_rate"]
        a1, a2, a3 = self.weights
        return a1 * se + a2 * ee + a3 * one_minus_ve

    def metrics(self) -> Dict[str, float]:
        rates_bps = self.rate_matrix_bps()
        active = self.W > 0
        total_rate_bps = float(np.nansum(rates_bps[active]))
        total_power_w = float(np.nansum(self._power_w_matrix()[active]))
        served = int(np.sum(active))
        qos_violations = self._active_qos_violations()
        denominator = max(1, self.current_index)
        blocking_rate = (self.blocked_users + qos_violations) / denominator
        se = total_rate_bps / self.cfg.channel_bandwidth_hz
        ee = (total_rate_bps / 1e6) / max(total_power_w, 1e-12)
        return {
            "served_users": served,
            "blocked_users": int(self.blocked_users),
            "blocked_reject_action": int(self.blocked_reject_action),
            "blocked_occupied_channel": int(self.blocked_occupied_channel),
            "blocked_qos": int(self.blocked_qos),
            "qos_violations": int(qos_violations),
            "blocking_rate": float(min(1.0, blocking_rate)),
            "spectral_efficiency": float(se),
            "energy_efficiency": float(ee),
            "total_power_w": total_power_w,
            "objective": float(getattr(self, "prev_Z", 0.0)),
        }

    def received_power_dbw_matrix(self) -> np.ndarray:
        p = self.power_dbw.copy()
        return p + self.cfg.tx_antenna_gain_db + self.cfg.rx_antenna_gain_db - self.cfg.free_space_loss_db

    def sinr_db_matrix(self, fill_value: float = np.nan) -> np.ndarray:
        rx_dbw = self.received_power_dbw_matrix()
        rx_w = dbw_to_w(rx_dbw)
        active = self.W > 0
        noise_w = self.noise_power_w()
        tx_by_channel = np.nan_to_num(rx_w, nan=0.0) * active
        interference = self.interference_coupling_linear @ tx_by_channel
        sinr = np.nan_to_num(rx_w, nan=0.0) / np.maximum(noise_w + interference, 1e-30)
        out = linear_to_db(sinr)
        out[~active] = fill_value
        return out

    def rate_matrix_bps(self) -> np.ndarray:
        sinr_db = self.sinr_db_matrix(fill_value=np.nan)
        return self._rate_from_sinr_db(sinr_db)

    def noise_power_w(self) -> float:
        noise_dbm = (
            self.cfg.thermal_noise_dbm_per_hz
            + 10.0 * math.log10(self.cfg.channel_bandwidth_hz)
            + self.cfg.noise_figure_db
        )
        return 10 ** ((noise_dbm - 30.0) / 10.0)

    def _active_qos_violations(self) -> int:
        active = self.W > 0
        sinr = self.sinr_db_matrix(fill_value=-1e9)
        rates_mbps = self._rate_from_sinr_db(sinr) / 1e6
        sinr_bad = sinr < self.user_threshold_db
        rate_bad = rates_mbps < self.user_rate_min_mbps
        return int(np.sum(active & (sinr_bad | rate_bad)))

    def _rate_from_sinr_db(self, sinr_db: np.ndarray) -> np.ndarray:
        sinr_linear = db_to_linear(sinr_db)
        return self.cfg.channel_bandwidth_hz * np.log2(1.0 + sinr_linear)

    def _no_available_resources(self) -> bool:
        if self.current_index >= len(self.users):
            return True
        beam = self.users[self.current_index].beam
        return bool(np.all(self.W[beam] > 0))

    def _power_w_matrix(self) -> np.ndarray:
        return dbw_to_w(self.power_dbw)

    def _snapshot(self):
        return (
            self.W.copy(),
            self.power_dbw.copy(),
            self.user_threshold_db.copy(),
            self.user_rate_min_mbps.copy(),
        )

    def _restore(self, snapshot) -> None:
        self.W, self.power_dbw, self.user_threshold_db, self.user_rate_min_mbps = snapshot

    def _full_snapshot(self):
        return {
            "arrays": self._snapshot(),
            "blocked_users": self.blocked_users,
            "blocked_reject_action": self.blocked_reject_action,
            "blocked_occupied_channel": self.blocked_occupied_channel,
            "blocked_qos": self.blocked_qos,
            "accepted_by_beam": self.accepted_by_beam.copy(),
            "blocked_by_beam": self.blocked_by_beam.copy(),
            "current_index": self.current_index,
            "U": self.U.copy(),
            "prev_Z": self.prev_Z,
        }

    def _full_restore(self, snapshot) -> None:
        self._restore(snapshot["arrays"])
        self.blocked_users = snapshot["blocked_users"]
        self.blocked_reject_action = snapshot["blocked_reject_action"]
        self.blocked_occupied_channel = snapshot["blocked_occupied_channel"]
        self.blocked_qos = snapshot["blocked_qos"]
        self.accepted_by_beam = snapshot["accepted_by_beam"]
        self.blocked_by_beam = snapshot["blocked_by_beam"]
        self.current_index = snapshot["current_index"]
        self.U = snapshot["U"]
        self.prev_Z = snapshot["prev_Z"]

    def _generate_users(self, n: int) -> List[User]:
        if self.traffic_distribution == "uniform":
            beams = self.rng.integers(0, self.cfg.num_beams, size=n)
        elif self.traffic_distribution == "hotspot":
            beams = []
            for _ in range(n):
                if self.rng.random() < self.cfg.hotspot_fraction:
                    beams.append(int(self.rng.choice(self.cfg.hotspot_beams)))
                else:
                    beams.append(int(self.rng.integers(0, self.cfg.num_beams)))
            beams = np.asarray(beams)
        else:
            probs = self.rng.poisson(lam=4.0, size=self.cfg.num_beams).astype(float) + 1.0
            probs /= probs.sum()
            beams = self.rng.choice(self.cfg.num_beams, size=n, p=probs)

        services = list(self.cfg.service_sinr_thresholds_db.keys())
        if "default" in services:
            services.remove("default")
        users = []
        for b in beams:
            service = self.py_rng.choice(services or ["default"])
            users.append(
                User(
                    beam=int(b),
                    service=service,
                    sinr_threshold_db=self.cfg.service_sinr_thresholds_db.get(service, self.cfg.min_sinr_db),
                    rate_min_mbps=self.cfg.service_rate_min_mbps.get(service, 0.5),
                )
            )
        return users

    def _traffic_distribution_matrix(self) -> np.ndarray:
        counts = np.zeros(self.cfg.num_beams, dtype=np.float32)
        for u in self.users[self.current_index :]:
            counts[u.beam] += 1.0
        return counts

    def _service_feature_names(self) -> List[str]:
        names = sorted(
            set(self.cfg.service_sinr_thresholds_db)
            | set(self.cfg.service_rate_min_mbps)
            | {"default"}
        )
        return names

    def _remaining_service_counts_by_beam(self) -> np.ndarray:
        names = self._service_feature_names()
        name_to_col = {name: idx for idx, name in enumerate(names)}
        counts = np.zeros((self.cfg.num_beams, len(names)), dtype=np.float32)
        for user in self.users[self.current_index :]:
            counts[user.beam, name_to_col.get(user.service, name_to_col["default"])] += 1.0
        return counts

    def _make_hex_coords(self, radius: int) -> List[Tuple[int, int]]:
        coords = []
        for q in range(-radius, radius + 1):
            r1 = max(-radius, -q - radius)
            r2 = min(radius, -q + radius)
            for r in range(r1, r2 + 1):
                coords.append((q, r))
        coords.sort(key=lambda x: (abs(x[0]) + abs(x[1]) + abs(-x[0] - x[1]), x[0], x[1]))
        return coords

    def _beam_distances(self) -> np.ndarray:
        n = len(self.coords)
        out = np.zeros((n, n), dtype=int)
        for i, a in enumerate(self.coords):
            for j, b in enumerate(self.coords):
                out[i, j] = hex_distance(a, b)
        return out

    def _interference_coupling_matrix(self) -> np.ndarray:
        mat = np.zeros((self.cfg.num_beams, self.cfg.num_beams), dtype=np.float64)
        for i in range(self.cfg.num_beams):
            for j in range(self.cfg.num_beams):
                if i == j:
                    continue
                ring = int(self.dist_matrix[i, j])
                mat[i, j] = db_to_linear(self.cfg.interference_coupling_db.get(ring, -45.0))
        return mat

    def _reconstructed_beams(self, center: int) -> List[int]:
        rings = [
            b
            for b in range(self.cfg.num_beams)
            if self.dist_matrix[center, b] <= self.reconstruction_rings
        ]
        high_traffic = np.argsort(-self.U)[: max(1, min(6, self.cfg.num_beams // 6))].tolist()
        merged = []
        for b in [center] + rings + high_traffic:
            if b not in merged:
                merged.append(int(b))
        return merged

    def _max_reconstructed_beams(self) -> int:
        # Four-ring state reconstruction in a 37-beam layout can cover all
        # beams; a fixed row count keeps CNN input dimensions stable.
        return self.cfg.num_beams


def hex_distance(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    aq, ar = a
    bq, br = b
    return int((abs(aq - bq) + abs(ar - br) + abs((-aq - ar) - (-bq - br))) / 2)


def db_to_linear(x):
    return np.power(10.0, np.asarray(x, dtype=np.float64) / 10.0)


def dbw_to_w(x):
    return db_to_linear(x)


def linear_to_db(x):
    return 10.0 * np.log10(np.maximum(x, 1e-30))
