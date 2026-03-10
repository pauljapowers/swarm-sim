"""
analysis.py – Statistical analysis, uncertainty quantification, and plotting.

Updated to handle split violation metrics:
  t_red_first_viol / t_red_total_viol  (and amber, req2 equivalents)
  task_completion_rate
"""

from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(data: np.ndarray, stat_fn=np.mean,
                 n_boot: int = 2000, alpha: float = 0.05,
                 rng_seed: int = 42) -> Tuple[float, float, float]:
    rng = np.random.default_rng(rng_seed)
    n = len(data)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    boot_stats = [stat_fn(rng.choice(data, size=n, replace=True))
                  for _ in range(n_boot)]
    lo = float(np.percentile(boot_stats, 100 * alpha / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    return float(stat_fn(data)), lo, hi


# ---------------------------------------------------------------------------
# Per-mode summary
# ---------------------------------------------------------------------------

def summarise_mode(df: pd.DataFrame) -> pd.Series:
    summary = {}

    # Violation probabilities
    for col, label in [
        ("any_req1_red",   "p_req1_red"),
        ("any_req1_amber", "p_req1_amber"),
        ("any_req2",       "p_req2"),
        ("any_req_functional", "p_functional"),
    ]:
        if col in df.columns:
            mean_, lo, hi = bootstrap_ci(df[col].values.astype(float))
            summary[f"{label}_mean"]  = mean_
            summary[f"{label}_ci_lo"] = lo
            summary[f"{label}_ci_hi"] = hi

    # Time-to-first and total violation durations
    time_cols = [
        ("t_red_first_viol",   "t_red_first"),
        ("t_amber_first_viol", "t_amber_first"),
        ("t_req2_first_viol",  "t_req2_first"),
        ("t_red_total_viol",   "t_red_total"),
        ("t_amber_total_viol", "t_amber_total"),
        ("t_req2_total_viol",  "t_req2_total"),
        # backward-compat
        ("t_red_viol",   "t_red_viol"),
        ("t_amber_viol", "t_amber_viol"),
        ("t_req2_viol",  "t_req2_viol"),
    ]
    for col, label in time_cols:
        if col in df.columns:
            arr = df[col].values.astype(float)
            summary[f"{label}_mean"] = float(np.mean(arr))
            summary[f"{label}_std"]  = float(np.std(arr))
            summary[f"{label}_p95"]  = float(np.percentile(arr, 95))

    # Functional metrics
    for col in ["task_completion_rate", "n_carriers_deposited"]:
        if col in df.columns:
            arr = df[col].values.astype(float)
            summary[f"{col}_mean"] = float(np.mean(arr))
            summary[f"{col}_std"]  = float(np.std(arr))

    summary["n_trials"] = len(df)
    return pd.Series(summary)


def compare_modes(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode_name, df in dfs.items():
        row = summarise_mode(df)
        row["mode"] = mode_name
        rows.append(row)
    return pd.DataFrame(rows).set_index("mode")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_aggregated(directory: str) -> Dict[str, pd.DataFrame]:
    dfs = {}
    for p in Path(directory).glob("metrics_*.csv"):
        mode = p.stem.replace("metrics_", "")
        dfs[mode] = pd.read_csv(p)
    return dfs


def load_raw_trial(directory: str, trial_id: int) -> pd.DataFrame:
    path = os.path.join(directory, f"trial_{trial_id:05d}.csv")
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plot: violation probabilities
# ---------------------------------------------------------------------------

def plot_violation_probs(comparison_df: pd.DataFrame,
                         output_path: Optional[str] = None,
                         title: str = "Violation Probabilities by Mode") -> plt.Figure:
    modes = list(comparison_df.index)
    metrics = [
        ("p_req1_red_mean",   "p_req1_red_ci_lo",   "p_req1_red_ci_hi",   "REQ1 Red"),
        ("p_req1_amber_mean", "p_req1_amber_ci_lo", "p_req1_amber_ci_hi", "REQ1 Amber"),
        ("p_req2_mean",       "p_req2_ci_lo",        "p_req2_ci_hi",       "REQ2 Density"),
        ("p_functional_mean", "p_functional_ci_lo",  "p_functional_ci_hi", "REQ-F Task Complete"),
    ]
    # Only include metrics present in the dataframe
    metrics = [(m, l, h, lbl) for m, l, h, lbl in metrics if m in comparison_df.columns]

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(4 * n + 2, 5), sharey=False)
    if n == 1:
        axes = [axes]
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(modes)))

    for ax, (mean_col, lo_col, hi_col, label) in zip(axes, metrics):
        means = comparison_df[mean_col].values
        lows  = comparison_df[lo_col].values  if lo_col  in comparison_df.columns else means
        highs = comparison_df[hi_col].values  if hi_col  in comparison_df.columns else means
        yerr_lo = means - lows
        yerr_hi = highs - means

        bars = ax.bar(modes, means, color=colors,
                      yerr=[yerr_lo, yerr_hi], capsize=6,
                      error_kw={"elinewidth": 1.5})
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Probability")
        ax.set_ylim(0, min(1.05, max(means) * 1.4 + 0.05))
        ax.tick_params(axis="x", rotation=30)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot: time in unsafe states (boxplot) — updated for split columns
# ---------------------------------------------------------------------------

def plot_time_in_unsafe(dfs: Dict[str, pd.DataFrame],
                        output_path: Optional[str] = None,
                        title: str = "Time in Unsafe / Functional States") -> plt.Figure:
    # Show both first-violation time and total violation time side by side
    cols = [
        ("t_red_first_viol",   "REQ1 Red\nTime-to-First (s)"),
        ("t_red_total_viol",   "REQ1 Red\nTotal Duration (s)"),
        ("t_amber_first_viol", "REQ1 Amber\nTime-to-First (s)"),
        ("t_amber_total_viol", "REQ1 Amber\nTotal Duration (s)"),
        ("task_completion_rate", "Task Completion\nRate"),
    ]
    # Filter to columns that exist in at least one df
    available = [c for c in cols
                 if any(c[0] in df.columns for df in dfs.values())]

    n_metrics = len(available)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    mode_names = list(dfs.keys())
    colors = plt.cm.Set2(np.linspace(0, 0.8, len(mode_names)))

    for ax, (col, label) in zip(axes, available):
        data_to_plot = []
        valid_modes = []
        for mode in mode_names:
            if col in dfs[mode].columns:
                data_to_plot.append(dfs[mode][col].dropna().values)
                valid_modes.append(mode)
        if not data_to_plot:
            ax.set_title(label + "\n(no data)")
            continue
        bp = ax.boxplot(data_to_plot, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
        ax.set_xticks(range(1, len(valid_modes) + 1))
        ax.set_xticklabels(valid_modes, rotation=30, ha="right")
        ax.set_title(label, fontweight="bold")
        ax.set_ylabel("Value")

    fig.suptitle(title, fontweight="bold", fontsize=13)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot: single trial trajectory
# ---------------------------------------------------------------------------

def plot_trial_trajectory(trial_df: pd.DataFrame,
                          arena_half: float = 1.85,
                          output_path: Optional[str] = None,
                          title: str = "Robot Trajectories") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-arena_half, arena_half)
    ax.set_ylim(-arena_half, arena_half)
    ax.set_aspect("equal")

    red_rect    = mpatches.Rectangle((-1.85, -1.85), 0.85, 1.85,
                    linewidth=1, edgecolor="red", facecolor="red", alpha=0.18)
    amber_rect  = mpatches.Rectangle((-2.35, -2.35), 1.85, 2.35,
                    linewidth=1, edgecolor="orange", facecolor="orange", alpha=0.12)
    deposit_rect = mpatches.Rectangle((1.00, -1.85), 0.85, 3.70,
                    linewidth=1, edgecolor="blue", facecolor="blue", alpha=0.12)
    for p in [amber_rect, red_rect, deposit_rect]:
        ax.add_patch(p)

    # Collection point marker
    ax.plot(-0.5, 0.0, "D", color="#cc6600", markersize=10, zorder=8,
            label="Collection point")
    # Drop-off point marker
    ax.plot(1.425, 0.0, "^", color="#0055cc", markersize=10, zorder=8,
            label="Drop-off point")

    colors = plt.cm.tab10(np.linspace(0, 0.9, trial_df["robot_id"].nunique()))
    for rid, color in zip(sorted(trial_df["robot_id"].unique()), colors):
        rdf = trial_df[trial_df["robot_id"] == rid]
        ax.plot(rdf["x"], rdf["y"], "-", color=color, alpha=0.5, linewidth=0.7)
        ax.plot(rdf["x"].iloc[0], rdf["y"].iloc[0], "o", color=color,
                markersize=5, label=f"Robot {rid}")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot: state proportions over time
# ---------------------------------------------------------------------------

def plot_state_proportions(trial_df: pd.DataFrame,
                           output_path: Optional[str] = None,
                           title: str = "Robot State Proportions Over Time") -> plt.Figure:
    states = ["SEARCHING", "PICKUP", "DROPOFF",
              "AVOIDANCE_S", "AVOIDANCE_P", "AVOIDANCE_D"]
    colors_map = {
        "SEARCHING":   "#4c72b0", "PICKUP":      "#55a868",
        "DROPOFF":     "#c44e52", "AVOIDANCE_S": "#8172b2",
        "AVOIDANCE_P": "#ccb974", "AVOIDANCE_D": "#64b5cd",
    }

    pivot = (trial_df.groupby(["time", "state"]).size()
             .unstack(fill_value=0)
             .reindex(columns=states, fill_value=0))
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    bottom = np.zeros(len(pivot_pct))
    for state in states:
        if state in pivot_pct.columns:
            vals = pivot_pct[state].values
            ax.fill_between(pivot_pct.index.values, bottom, bottom + vals,
                            label=state, alpha=0.85,
                            color=colors_map.get(state, "grey"))
            bottom += vals

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Fraction of robots")
    ax.set_ylim(0, 1)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Plot: violations timeline
# ---------------------------------------------------------------------------

def plot_violations_timeline(trial_df: pd.DataFrame,
                              output_path: Optional[str] = None,
                              title: str = "Safety Violations Timeline") -> plt.Figure:
    ts_df = trial_df.groupby("time").agg(
        req1_red=("req1_red_violation", "max"),
        req1_amber=("req1_amber_violation", "max"),
        req2=("req2_violation", "max"),
    ).reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(12, 5), sharex=True)
    for ax, col, label, color in [
        (axes[0], "req1_red",   "REQ1 Red",   "red"),
        (axes[1], "req1_amber", "REQ1 Amber", "orange"),
        (axes[2], "req2",       "REQ2",       "purple"),
    ]:
        ax.fill_between(ts_df["time"], ts_df[col].values,
                        color=color, alpha=0.6, step="pre")
        ax.set_ylabel(label, fontsize=9)
        ax.set_ylim(-0.05, 1.2)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title, fontweight="bold", fontsize=12)
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Patch attribution table
# ---------------------------------------------------------------------------

def patch_attribution_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for mode, df in dfs.items():
        row = {"mode": mode}
        for col in ["t_red_first_viol", "t_red_total_viol",
                    "t_amber_first_viol", "t_amber_total_viol",
                    "task_completion_rate"]:
            if col in df.columns:
                row[f"{col}_mean"] = df[col].mean()
                row[f"{col}_std"]  = df[col].std()
        for col in ["any_req1_red", "any_req1_amber", "any_req2", "any_req_functional"]:
            row[f"p_{col}"] = df[col].mean() if col in df else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("mode")


def ewd(values, n_bins=5):
    import numpy as np
    bins = np.linspace(min(values), max(values), n_bins + 1)
    return np.digitize(values, bins, right=True)
