"""Sanity checks for DB-GNN final benchmark CSV files.

This script checks simulator/benchmark health. It does not claim algorithmic
superiority or paper-level conclusions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = ["blocking_rate", "spectral_efficiency", "energy_efficiency", "total_power"]


def status(ok: bool, message: str, level: str = "FAIL") -> bool:
    tag = "OK" if ok else level
    print(f"[{tag}] {message}")
    return ok or level == "WARN"


def check_required_columns(df: pd.DataFrame) -> bool:
    required = [
        "algo",
        "users",
        "seed",
        "traffic",
        "weight_se",
        "weight_ee",
        "weight_blocking",
        "blocking_rate",
        "spectral_efficiency",
        "energy_efficiency",
        "total_power",
        "qos_violations",
        "episode_reward",
        "epsilon",
        "train_steps",
    ]
    missing = [c for c in required if c not in df.columns]
    return status(not missing, f"required columns present" if not missing else f"missing columns: {missing}")


def check_numeric_health(df: pd.DataFrame) -> bool:
    ok = True
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    nan_counts = df[numeric_cols].isna().sum()
    optional = {
        "episode_reward",
        "epsilon",
        "loss",
        "mean_q_value",
        "invalid_action_rate",
        "replay_buffer_size",
        "candidate_steps",
        "sage_steps",
        "hotspot_threshold",
        "candidate_min_users",
        "candidate_min_remaining",
        "candidate_min_peak_load",
        "selector_reject_actions",
        "selector_margin",
        "selector_tolerance",
        "selector_evaluated_actions",
        "selector_top_k",
        "eval_episodes",
    }
    optional.update(c for c in nan_counts.index if c.endswith("_std"))
    optional.update(c for c in nan_counts.index if c.startswith("blocked_"))
    non_optional_nan = nan_counts.drop(labels=[c for c in optional if c in nan_counts.index])
    ok &= status((non_optional_nan == 0).all(), "no NaN in required numeric columns" if (non_optional_nan == 0).all() else f"NaN counts:\n{non_optional_nan[non_optional_nan > 0]}")
    for col in numeric_cols:
        has_inf = np.isinf(df[col].to_numpy(dtype=float, na_value=np.nan)).any()
        ok &= status(not has_inf, f"{col} has no inf" if not has_inf else f"{col} has inf")
    return ok


def check_metric_ranges(df: pd.DataFrame) -> bool:
    ok = True
    ok &= status(((df["blocking_rate"] >= 0) & (df["blocking_rate"] <= 1)).all(), "blocking_rate in [0, 1]")
    for col in ["spectral_efficiency", "energy_efficiency", "total_power", "qos_violations"]:
        ok &= status((df[col] >= 0).all(), f"{col} >= 0")
    return ok


def check_completeness(df: pd.DataFrame) -> bool:
    combos = df.groupby(["algo", "users", "seed", "traffic", "weight_se", "weight_ee", "weight_blocking"]).size()
    duplicates = combos[combos > 1]
    ok = status(duplicates.empty, "no duplicate algo/users/seed/traffic/weight rows" if duplicates.empty else f"duplicate rows:\n{duplicates}")
    print(f"Rows: {len(df)}")
    print("Rows by algo:")
    print(df.groupby("algo").size().to_string())
    return ok


def check_trends(df: pd.DataFrame) -> bool:
    ok = True
    grouped = df.groupby(["algo", "traffic", "weight_se", "weight_ee", "weight_blocking", "users"])["blocking_rate"].mean().reset_index()
    for key, g in grouped.groupby(["algo", "traffic", "weight_se", "weight_ee", "weight_blocking"]):
        g = g.sort_values("users")
        if g["users"].nunique() >= 2:
            low = g.iloc[0]["blocking_rate"]
            high = g.iloc[-1]["blocking_rate"]
            ok &= status(high + 1e-9 >= low, f"blocking nondecreasing endpoint for {key}: {low:.3f} -> {high:.3f}", level="WARN")

    if {"uniform", "hotspot"}.issubset(set(df["traffic"].unique())):
        traffic_mean = df.groupby(["algo", "users", "weight_se", "weight_ee", "weight_blocking", "traffic"])["blocking_rate"].mean().unstack()
        if {"uniform", "hotspot"}.issubset(traffic_mean.columns):
            bad = traffic_mean[traffic_mean["hotspot"] + 1e-9 < traffic_mean["uniform"]]
            ok &= status(bad.empty, "hotspot blocking >= uniform blocking for matched groups" if bad.empty else f"hotspot lower than uniform in {len(bad)} matched groups", level="WARN")

    if {"random", "greedy_z"}.issubset(set(df["algo"].unique())):
        pivot = df.groupby(["algo", "users", "traffic", "weight_se", "weight_ee", "weight_blocking"])["blocking_rate"].mean().unstack("algo")
        bad = pivot[pivot["greedy_z"] > pivot["random"] + 1e-9]
        ok &= status(bad.empty, "greedy_z blocking <= random blocking" if bad.empty else f"greedy_z worse than random in {len(bad)} matched groups", level="WARN")
    return ok


def check_dqn_logs(df: pd.DataFrame, log_dir: str | None) -> bool:
    dqn = df[df["algo"] == "dqn"]
    if dqn.empty:
        return status(True, "no DQN rows to inspect", level="WARN")
    ok = True
    if "loss" in dqn.columns:
        finite_loss = dqn["loss"].dropna().map(np.isfinite).all()
        ok &= status(finite_loss, "DQN final loss finite where present", level="WARN")
    if "invalid_action_rate" in dqn.columns:
        invalid = dqn["invalid_action_rate"].dropna()
        if not invalid.empty:
            ok &= status(((invalid >= 0) & (invalid <= 1)).all(), "DQN invalid_action_rate in [0, 1]", level="WARN")
            high = invalid.mean() > 0.5
            ok &= status(not high, f"DQN mean invalid_action_rate = {invalid.mean():.3f}", level="WARN")

    if log_dir:
        paths = list(Path(log_dir).glob("dqn_*.csv"))
        ok &= status(bool(paths), f"DQN episode logs found in {log_dir}" if paths else f"no DQN logs found in {log_dir}", level="WARN")
        for path in paths[:5]:
            log = pd.read_csv(path)
            if "loss" in log.columns:
                vals = log["loss"].dropna()
                if not vals.empty:
                    ok &= status(np.isfinite(vals).all(), f"{path.name}: loss finite", level="WARN")
            if "mean_q_value" in log.columns:
                vals = log["mean_q_value"].dropna()
                if not vals.empty:
                    ok &= status(np.isfinite(vals).all(), f"{path.name}: mean_q_value finite", level="WARN")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Raw benchmark CSV, e.g. data/raw/final_db_gnn_main_results.csv")
    parser.add_argument("--log-dir", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f"File: {args.csv}")
    ok = True
    ok &= check_required_columns(df)
    ok &= check_completeness(df)
    ok &= check_numeric_health(df)
    ok &= check_metric_ranges(df)
    ok &= check_trends(df)
    ok &= check_dqn_logs(df, args.log_dir)

    print("\nMetric means by algo/users:")
    print(df.groupby(["algo", "users"])[METRICS].mean().to_string())
    print("\nOverall:", "PASS_WITH_WARNINGS_OR_OK" if ok else "CHECK_FAILURES_FOUND")


if __name__ == "__main__":
    main()
