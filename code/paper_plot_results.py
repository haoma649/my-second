"""Create paper-style figures for the LEO reproduction experiments.

The source paper uses a MATLAB-like style: boxed axes, ticks on all sides,
serif labels, no grid, white legend box, and distinctive red/blue/black
marker-line curves. This script keeps that visual style while still computing
curves from raw CSV rows. Standard-deviation bars are optional because the
paper figures themselves do not show them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "blocking_rate": ("Blocking rate", "blocking_rate"),
    "spectral_efficiency": ("Spectral efficiency(Mbps/MHz)", "spectral_efficiency"),
    "energy_efficiency": ("Energy efficiency(Mbps/W)", "energy_efficiency"),
    "total_power": ("Consumed power(W)", "total_power"),
}

BASE_LABELS = {
    "dqn": "DQN",
    "qlearning": "Q-Learning",
    "random": "Random",
    "greedy_z": "Greedy-Z",
    "gnn_topk10_selector_sage_candidate": "DB-GNN",
    "db_gnn": "DB-GNN",
}

ALGO_ORDER = {
    "dqn": 0,
    "qlearning": 1,
    "random": 2,
    "greedy_z": 3,
    "gnn_topk10_selector_sage_candidate": 4,
    "db_gnn": 4,
}

PAPER_SERIES_STYLE = [
    {"color": "red", "marker": "*", "linestyle": "-", "markersize": 9, "markerfacecolor": "red"},
    {"color": "blue", "marker": "o", "linestyle": "-", "markersize": 6, "markerfacecolor": "none"},
    {"color": "black", "marker": "D", "linestyle": "-", "markersize": 5.5, "markerfacecolor": "none"},
    {"color": "magenta", "marker": "s", "linestyle": "-", "markersize": 5.5, "markerfacecolor": "none"},
    {"color": "green", "marker": "^", "linestyle": "-", "markersize": 6, "markerfacecolor": "none"},
    {"color": "darkorange", "marker": "v", "linestyle": "-", "markersize": 6, "markerfacecolor": "none"},
    {"color": "purple", "marker": "P", "linestyle": "-", "markersize": 6, "markerfacecolor": "none"},
    {"color": "dimgray", "marker": "x", "linestyle": "-", "markersize": 6, "markerfacecolor": "none"},
]


def paper_style() -> None:
    plt.rcParams.update(
        {
            # DejaVu Serif is bundled with Matplotlib and avoids noisy font
            # fallback warnings on Windows while preserving the paper's serif look.
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.labelweight": "bold",
            "axes.titlesize": 13,
            "legend.fontsize": 9,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.linewidth": 1.2,
            "lines.linewidth": 1.7,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "mathtext.fontset": "dejavuserif",
        }
    )


def filter_df(df: pd.DataFrame, args) -> pd.DataFrame:
    out = df.copy()
    if args.traffic:
        out = out[out["traffic"].isin(args.traffic)]
    if args.weights and "weight_name" in out.columns:
        out = out[out["weight_name"].isin(args.weights)]
    if args.algos:
        out = out[out["algo"].isin(args.algos)]
    return out


def series_label(algo: str, group: pd.DataFrame, include_weight: bool) -> str:
    label = BASE_LABELS.get(algo, algo)
    if include_weight and {"weight_se", "weight_ee", "weight_blocking"}.issubset(group.columns):
        weights = group[["weight_se", "weight_ee", "weight_blocking"]].dropna().drop_duplicates()
        if len(weights) == 1:
            w = weights.iloc[0]
            label += f"({w['weight_se']:.3g},{w['weight_ee']:.3g},{w['weight_blocking']:.3g})"
    return label


def group_columns(df: pd.DataFrame, split_weights: bool) -> list[str]:
    cols = ["algo"]
    if split_weights:
        if "weight_name" in df.columns:
            cols.append("weight_name")
        elif {"weight_se", "weight_ee", "weight_blocking"}.issubset(df.columns):
            cols.extend(["weight_se", "weight_ee", "weight_blocking"])
    return cols


def ordered_groups(df: pd.DataFrame, split_weights: bool):
    cols = group_columns(df, split_weights)
    groups = list(df.groupby(cols, sort=False))

    def rank(item) -> tuple[int, str]:
        key, _ = item
        algo = key if not isinstance(key, tuple) else key[0]
        return (ALGO_ORDER.get(str(algo), 99), str(key))

    return sorted(groups, key=rank)


def add_origin_point(agg: pd.DataFrame) -> pd.DataFrame:
    if 0 in set(agg["users"]):
        return agg
    origin = pd.DataFrame({"users": [0], "mean": [0.0], "std": [0.0]})
    return pd.concat([origin, agg], ignore_index=True).sort_values("users")


def style_axes(ax, ylabel: str, title: str | None, y_min_zero: bool, grid: bool) -> None:
    ax.set_xlabel("Number of users", labelpad=8)
    ax.set_ylabel(ylabel, labelpad=10)
    if title:
        ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 200)
    ax.set_xticks(list(range(0, 201, 25)))
    if y_min_zero:
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom=0, top=top)
    ax.tick_params(direction="in", top=True, right=True, length=5.5, width=1.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)
        spine.set_color("black")
    if grid:
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    else:
        ax.grid(False)


def draw_metric(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    out_dir: Path,
    split_traffic: bool,
    file_prefix: str,
    split_weights: bool,
    error_bars: bool,
    add_origin: bool,
    grid: bool,
    legend_loc: str,
    show_title: bool,
) -> None:
    traffic_values = sorted(df["traffic"].dropna().unique()) if split_traffic and "traffic" in df.columns else [None]
    for traffic in traffic_values:
        gdf = df[df["traffic"] == traffic] if traffic is not None else df
        fig, ax = plt.subplots(figsize=(5.2, 4.1))
        for idx, (key, group) in enumerate(ordered_groups(gdf, split_weights)):
            if not isinstance(key, tuple):
                algo = key
            else:
                algo = key[0]
            agg = group.groupby("users")[metric].agg(["mean", "std"]).reset_index().sort_values("users")
            agg["std"] = agg["std"].fillna(0.0)
            if add_origin:
                agg = add_origin_point(agg)
            style = PAPER_SERIES_STYLE[idx % len(PAPER_SERIES_STYLE)]
            label = series_label(algo, group, include_weight=split_weights)
            if error_bars:
                ax.errorbar(
                    agg["users"],
                    agg["mean"],
                    yerr=agg["std"],
                    capsize=3.0,
                    markeredgewidth=1.2,
                    label=label,
                    **style,
                )
            else:
                ax.plot(
                    agg["users"],
                    agg["mean"],
                    markeredgewidth=1.2,
                    label=label,
                    **style,
                )
        title = traffic.capitalize() if traffic is not None and show_title else None
        style_axes(ax, ylabel, title, y_min_zero=True, grid=grid)
        ax.legend(loc=legend_loc, frameon=True, edgecolor="black", facecolor="white", framealpha=1.0, fancybox=False)
        fig.subplots_adjust(left=0.23, bottom=0.18, right=0.98, top=0.97)
        suffix = f"_{traffic}" if traffic is not None else ""
        png = out_dir / f"{file_prefix}{metric}{suffix}.png"
        pdf = out_dir / f"{file_prefix}{metric}{suffix}.pdf"
        fig.savefig(png, bbox_inches="tight")
        fig.savefig(pdf, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {png}")
        print(f"wrote {pdf}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--out-dir", default="figures/paper_style")
    parser.add_argument("--traffic", nargs="*", default=None)
    parser.add_argument("--weights", nargs="*", default=None)
    parser.add_argument("--algos", nargs="*", default=None)
    parser.add_argument("--split-traffic", action="store_true")
    parser.add_argument("--split-weights", action="store_true", help="Draw separate curves for each algorithm/weight preset.")
    parser.add_argument("--error-bars", action="store_true", help="Show mean +/- std. Disabled by default to match the source paper.")
    parser.add_argument("--add-origin", action="store_true", help="Add a synthetic (0 users, 0 metric) point, matching the paper x-axis origin.")
    parser.add_argument("--grid", action="store_true", help="Show a light dashed grid. Disabled by default to match the source paper.")
    parser.add_argument("--show-title", action="store_true", help="Show traffic title above split-traffic figures.")
    parser.add_argument("--legend-loc", default="lower right")
    parser.add_argument("--file-prefix", default="")
    args = parser.parse_args()

    paper_style()
    df = pd.concat([pd.read_csv(p) for p in args.csv], ignore_index=True, sort=False)
    if "algo" not in df.columns and "method" in df.columns:
        df["algo"] = df["method"]
    if "total_power" not in df.columns and "total_power_w" in df.columns:
        df["total_power"] = df["total_power_w"]
    df = filter_df(df, args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric, (ylabel, col) in METRICS.items():
        if col in df.columns:
            draw_metric(
                df,
                col,
                ylabel,
                out_dir,
                args.split_traffic,
                args.file_prefix,
                args.split_weights,
                args.error_bars,
                args.add_origin,
                args.grid,
                args.legend_loc,
                args.show_title,
            )


if __name__ == "__main__":
    main()
