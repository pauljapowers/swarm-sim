"""
app.py – Streamlit interactive dashboard for the swarm cloakroom simulation.

Run with:
    streamlit run app.py
"""

import os
import sys
import math
import random
import copy
import time

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.arena import Arena
from src.robot import Robot, RobotState, STATE_NAMES
from src.simulation import run_trial, run_batch
from src.analysis import (
    load_aggregated, compare_modes,
    plot_violation_probs, plot_time_in_unsafe,
    plot_trial_trajectory, plot_state_proportions,
    plot_violations_timeline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

CONFIG_PATH = os.path.join(ROOT, "config.yaml")


@st.cache_data(show_spinner=False)
def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def apply_sidebar_overrides(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)

    st.sidebar.title("⚙️ Configuration")

    # Mode
    mode = st.sidebar.selectbox(
        "Mode",
        ["baseline", "patches_no_adapt", "patches_adapt", "adapt_only"],
        index=0,
    )
    cfg["mode"] = mode
    cfg["patches_enabled"] = mode in ("patches_no_adapt", "patches_adapt")
    cfg["robot_patch_adaptation_enabled"] = mode in ("patches_adapt", "adapt_only")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation")
    cfg["robots"]["count"] = st.sidebar.slider("# Robots", 1, 10, cfg["robots"]["count"])
    cfg["simulation"]["duration"] = st.sidebar.slider(
        "Trial duration (s)", 20, 200, int(cfg["simulation"]["duration"]))
    cfg["simulation"]["dt"] = 0.04

    st.sidebar.markdown("---")
    st.sidebar.subheader("PFSM probabilities")
    cfg["pfsm"]["Ps"] = st.sidebar.slider("Ps (find carrier)",   0.01, 0.30, float(cfg["pfsm"]["Ps"]), 0.01)
    cfg["pfsm"]["Pp"] = st.sidebar.slider("Pp (pickup)",          0.01, 0.30, float(cfg["pfsm"]["Pp"]), 0.01)
    cfg["pfsm"]["Pd"] = st.sidebar.slider("Pd (dropoff)",         0.01, 0.30, float(cfg["pfsm"]["Pd"]), 0.01)
    cfg["pfsm"]["Pa"] = st.sidebar.slider("Pa (avoidance)",       0.01, 0.50, float(cfg["pfsm"]["Pa"]), 0.01)

    if cfg["patches_enabled"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Friction patches")
        cfg["friction"]["num_patches"] = st.sidebar.slider(
            "# Patches", 0, 8, int(cfg["friction"].get("num_patches", 3)))
        cfg["friction"]["placement_strategy"] = st.sidebar.selectbox(
            "Placement", ["random", "near_red", "near_deposit"])
        cfg["friction"]["patch_size"] = st.sidebar.slider(
            "Patch size (m)", 0.20, 1.50, float(cfg["friction"].get("patch_size", 0.5)), 0.05)

    return cfg


# ── Arena visualisation ───────────────────────────────────────────────────────

STATE_COLORS = {
    "SEARCHING":   "#4c72b0",
    "PICKUP":      "#55a868",
    "DROPOFF":     "#c44e52",
    "AVOIDANCE_S": "#8172b2",
    "AVOIDANCE_P": "#ccb974",
    "AVOIDANCE_D": "#64b5cd",
}

PATCH_COLORS = {
    "WATER": "#3399ff",
    "OIL":   "#cc6600",
    "ICE":   "#aaddff",
}


def draw_arena_frame(arena: Arena, robots: list, t: float = 0.0,
                     ax: plt.Axes = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    half = 1.85
    ax.set_xlim(-half - 0.1, half + 0.1)
    ax.set_ylim(-half - 0.1, half + 0.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f5ee")

    # Arena boundary
    arena_rect = mpatches.Rectangle(
        (-half, -half), 2 * half, 2 * half,
        linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(arena_rect)

    # Friction patches (draw first, under zones)
    for p in arena.patches:
        if p.active:
            color = PATCH_COLORS.get(p.spill_type, "grey")
            patch_rect = mpatches.Rectangle(
                (p.x, p.y), p.width, p.height,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.35,
                zorder=1)
            ax.add_patch(patch_rect)
            ax.text(p.x + p.width / 2, p.y + p.height / 2,
                    p.spill_type[0], ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold", zorder=2)

    # Amber zone
    amber = arena.amber_zone
    ax.add_patch(mpatches.Rectangle(
        (amber.x_min, amber.y_min),
        amber.x_max - amber.x_min,
        amber.y_max - amber.y_min,
        facecolor="orange", alpha=0.18, zorder=2))

    # Red zone
    red = arena.red_zone
    ax.add_patch(mpatches.Rectangle(
        (red.x_min, red.y_min),
        red.x_max - red.x_min,
        red.y_max - red.y_min,
        facecolor="red", alpha=0.28, zorder=3))

    # Fire exit marker
    ax.plot(-1.85, -1.0, "Xg", markersize=14, zorder=10, label="Fire exit")

    # Deposit zone
    dep = arena.deposit_zone
    ax.add_patch(mpatches.Rectangle(
        (dep.x_min, dep.y_min),
        dep.x_max - dep.x_min,
        dep.y_max - dep.y_min,
        facecolor="#3399ff", alpha=0.14, zorder=2))

    # Carriers
    for c in arena.get_active_carriers():
        circ = mpatches.Circle(
            (c.x, c.y), c.radius,
            facecolor="#336699", edgecolor="#003366",
            linewidth=1.2, alpha=0.8, zorder=5)
        ax.add_patch(circ)

    # Robots
    for r in robots:
        color = STATE_COLORS.get(r.state_name, "grey")
        circ = mpatches.Circle(
            (r.x, r.y), r.radius,
            facecolor=color, edgecolor="black",
            linewidth=0.8, alpha=0.9, zorder=6)
        ax.add_patch(circ)
        # Velocity arrow
        spd = math.hypot(r.vx, r.vy)
        if spd > 0.05:
            ax.annotate("",
                        xy=(r.x + r.vx * 0.12, r.y + r.vy * 0.12),
                        xytext=(r.x, r.y),
                        arrowprops=dict(arrowstyle="->", color=color,
                                        lw=1.2),
                        zorder=7)

    # Violation indicators
    n_red   = sum(1 for r in robots if r.in_red)
    n_amber = sum(1 for r in robots if r.in_amber)
    if n_red > 0:
        ax.set_title(f"t={t:.1f}s  ⚠️ REQ1 RED VIOLATION", color="red",
                     fontweight="bold")
    elif n_amber > 1:
        ax.set_title(f"t={t:.1f}s  ⚠️ REQ1 AMBER CRITICAL",
                     color="orange", fontweight="bold")
    else:
        ax.set_title(f"t={t:.1f}s  ✓ OK", color="green")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2, zorder=0)
    return fig


# ── Pages ─────────────────────────────────────────────────────────────────────

def page_live_sim(cfg):
    st.header("🔴 Live Simulation")
    st.markdown(
        "Run a single trial in real time. Watch robots move around the cloakroom "
        "arena and observe safety violations."
    )

    seed = st.number_input("Random seed", value=42, step=1)
    speed = st.slider("Playback speed (steps per frame)", 1, 50, 10)
    run_btn = st.button("▶ Run Trial")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()

    if run_btn:
        rng = random.Random(int(seed))
        arena = Arena(cfg)
        arena.reset_patches(rng)
        for i, pos in enumerate(cfg["carriers"]["initial_positions"]):
            arena.carriers[i].x = pos[0]
            arena.carriers[i].y = pos[1]
            arena.carriers[i].active = True

        from src.simulation import _random_deposit_position
        robots = []
        placed = []
        r_radius = cfg["robots"]["diameter"] / 2.0
        for rid in range(cfg["robots"]["count"]):
            x, y = _random_deposit_position(arena, r_radius, rng, placed)
            placed.append((x, y))
            robots.append(Robot(rid, x, y, cfg,
                                rng=random.Random(rng.randint(0, 2**31))))

        dt = cfg["simulation"]["dt"]
        n_steps = int(cfg["simulation"]["duration"] / dt)

        for step in range(n_steps):
            t = step * dt
            arena.update_patches(t)
            for robot in robots:
                robot.step(arena, robots, dt)

            if step % speed == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                draw_arena_frame(arena, robots, t=t, ax=ax)
                frame_placeholder.pyplot(fig, use_container_width=True)
                plt.close(fig)

                state_counts = {
                    s: sum(1 for r in robots if r.state_name == s)
                    for s in STATE_NAMES.values()
                }
                stats_placeholder.dataframe(
                    pd.DataFrame([state_counts], index=["# robots"]),
                    use_container_width=True)

        st.success("Trial complete!")


def page_batch_run(cfg):
    st.header("📊 Batch Simulation")
    st.markdown("Run multiple trials and collect statistics.")

    n_trials = st.number_input("Number of trials", 1, 1000, 20)
    seed = st.number_input("Random seed", value=42, step=1)

    if st.button("▶ Run Batch"):
        with st.spinner(f"Running {n_trials} trials…"):
            df = run_batch(
                cfg, n_trials=int(n_trials), seed=int(seed),
                output_dir="data/raw",
                aggregated_dir="data/aggregated",
                save_raw=False,
                verbose=False,
            )
        st.success(f"Done! {len(df)} trials completed.")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Summary Statistics")
        from src.analysis import summarise_mode
        summary = summarise_mode(df)
        st.dataframe(pd.DataFrame(summary).T, use_container_width=True)

        # Plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ax, col, label, color in zip(
            axes,
            ["t_red_viol", "t_amber_viol", "t_req2_viol"],
            ["REQ1 Red (s)", "REQ1 Amber (s)", "REQ2 Density (s)"],
            ["red", "orange", "purple"]
        ):
            if col in df.columns:
                ax.hist(df[col], bins=20, color=color, alpha=0.7, edgecolor="white")
                ax.set_title(label)
                ax.set_xlabel("Time (s)")
        fig.suptitle(f"Mode: {cfg['mode']}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


def page_compare(cfg):
    st.header("📈 Mode Comparison")
    st.markdown(
        "Load pre-computed aggregated metrics and compare across modes. "
        "Run the batch simulation for each mode first via the sidebar."
    )

    agg_dir = "data/aggregated"
    dfs = load_aggregated(agg_dir)
    if not dfs:
        st.warning(f"No aggregated data found in `{agg_dir}`. "
                   "Please run batch simulations first.")
        return

    st.write(f"Found {len(dfs)} mode(s): {', '.join(dfs.keys())}")
    comp = compare_modes(dfs)
    st.dataframe(comp, use_container_width=True)

    fig = plot_violation_probs(comp, title="Violation Probabilities by Mode")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    fig2 = plot_time_in_unsafe(dfs, title="Time in Unsafe States")
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


def page_trial_analysis(cfg):
    st.header("🔬 Trial Analysis")
    st.markdown("Analyse a saved raw trial CSV.")

    raw_dir = "data/raw"
    if not os.path.isdir(raw_dir):
        st.warning("No raw data directory found. Run a batch first.")
        return

    # Look for trial files across subdirs
    all_files = list(Path(raw_dir).rglob("trial_*.csv"))
    if not all_files:
        st.warning(f"No trial CSV files found under {raw_dir}.")
        return

    chosen = st.selectbox("Select trial file", [str(f) for f in all_files[:50]])
    if st.button("Load & Analyse"):
        df = pd.read_csv(chosen)
        st.write(f"Loaded {len(df)} rows, "
                 f"{df['robot_id'].nunique()} robots, "
                 f"{df['timestep'].nunique()} timesteps.")

        tab1, tab2, tab3 = st.tabs(["Trajectories", "State Proportions",
                                     "Violations Timeline"])
        with tab1:
            fig = plot_trial_trajectory(df, title=f"Trajectories – {os.path.basename(chosen)}")
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with tab2:
            fig = plot_state_proportions(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        with tab3:
            fig = plot_violations_timeline(df)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Swarm Cloakroom Simulation",
        page_icon="🤖",
        layout="wide",
    )

    st.title("🤖 Swarm Cloakroom Simulation Dashboard")
    st.caption(
        "Replication of Abeywickrama et al. IEEE ERAS 2025 + "
        "dynamic friction patch extensions for PhD safety assurance research."
    )

    cfg = load_cfg()
    cfg = apply_sidebar_overrides(cfg)

    page = st.sidebar.radio(
        "Page",
        ["Live Simulation", "Batch Run", "Mode Comparison", "Trial Analysis"],
    )

    from pathlib import Path

    if page == "Live Simulation":
        page_live_sim(cfg)
    elif page == "Batch Run":
        page_batch_run(cfg)
    elif page == "Mode Comparison":
        page_compare(cfg)
    elif page == "Trial Analysis":
        page_trial_analysis(cfg)


if __name__ == "__main__":
    main()
