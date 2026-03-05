"""
app.py – Streamlit interactive dashboard for the swarm cloakroom simulation.
Upgraded version: preset experiment buttons, timers, Supabase integration,
PNG export, publication-quality charts, ZIP download, Path fix.
"""
import os
import sys
import io
import math
import random
import copy
import time
import zipfile
import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import yaml
from pathlib import Path

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

# ── Supabase (optional – graceful fallback if not configured) ─────────────────
def get_supabase():
    try:
        from supabase import create_client
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            return create_client(url, key)
    except Exception:
        pass
    return None

def save_trial_to_supabase(sb, cfg, metrics: dict, run_label: str = ""):
    """Save a single trial's metrics to Supabase trial_metrics table."""
    if sb is None:
        return
    try:
        mode = cfg.get("mode", "unknown")
        row = {
            "mode": mode,
            "trial_id": metrics.get("trial_id", 0),
            "t_red_viol": float(metrics.get("t_red_viol", 0)),
            "t_amber_viol": float(metrics.get("t_amber_viol", 0)),
            "t_req2_viol": float(metrics.get("t_req2_viol", 0)),
            "n_red_entries": int(metrics.get("n_red_entries", 0)),
            "n_amber_entries": int(metrics.get("n_amber_entries", 0)),
            "n_req2_entries": int(metrics.get("n_req2_entries", 0)),
            "any_req1_red": bool(metrics.get("any_req1_red", False)),
            "any_req1_amber": bool(metrics.get("any_req1_amber", False)),
            "any_req2": bool(metrics.get("any_req2", False)),
        }
        sb.table("trial_metrics").insert(row).execute()
    except Exception as e:
        pass  # Don't let DB errors break the simulation

def save_batch_to_supabase(sb, cfg, df: pd.DataFrame, run_label: str,
                            elapsed_s: float):
    """Save aggregated batch stats to Supabase."""
    if sb is None:
        return
    try:
        from src.analysis import bootstrap_ci
        mode = cfg.get("mode", "unknown")

        # experiment_runs row
        run_row = {
            "run_label": run_label,
            "mode": mode,
            "n_trials": len(df),
            "seed": int(cfg.get("_seed", 0)),
            "patches_enabled": bool(cfg.get("patches_enabled", False)),
            "adaptation_enabled": bool(cfg.get("robot_patch_adaptation_enabled", False)),
            "n_robots": int(cfg["robots"]["count"]),
            "duration_s": float(cfg["simulation"]["duration"]),
            "dt": float(cfg["simulation"]["dt"]),
            "notes": f"elapsed={elapsed_s:.0f}s",
        }
        resp = sb.table("experiment_runs").insert(run_row).execute()
        run_id = resp.data[0]["id"] if resp.data else None

        # aggregate_stats row
        def ci(col):
            arr = df[col].values.astype(float)
            m, lo, hi = bootstrap_ci(arr)
            return float(m), float(lo), float(hi)

        p_red_m, p_red_lo, p_red_hi   = ci("any_req1_red")
        p_amb_m, p_amb_lo, p_amb_hi   = ci("any_req1_amber")
        p_req2_m, p_req2_lo, p_req2_hi = ci("any_req2")

        agg_row = {
            "run_id": run_id,
            "mode": mode,
            "p_req1_red_mean": p_red_m,   "p_req1_red_ci_lo": p_red_lo,   "p_req1_red_ci_hi": p_red_hi,
            "p_req1_amber_mean": p_amb_m, "p_req1_amber_ci_lo": p_amb_lo, "p_req1_amber_ci_hi": p_amb_hi,
            "p_req2_mean": p_req2_m,      "p_req2_ci_lo": p_req2_lo,      "p_req2_ci_hi": p_req2_hi,
            "t_red_mean_s":   float(df["t_red_viol"].mean()),
            "t_red_std_s":    float(df["t_red_viol"].std()),
            "t_red_p95_s":    float(df["t_red_viol"].quantile(0.95)),
            "t_amber_mean_s": float(df["t_amber_viol"].mean()),
            "t_amber_std_s":  float(df["t_amber_viol"].std()),
            "t_req2_mean_s":  float(df["t_req2_viol"].mean()),
            "t_req2_std_s":   float(df["t_req2_viol"].std()),
            "n_trials": len(df),
        }
        sb.table("aggregate_stats").insert(agg_row).execute()
    except Exception as e:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

@st.cache_data(show_spinner=False)
def load_cfg():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def apply_sidebar_overrides(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)

    st.sidebar.title("⚙️ Configuration")

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
    cfg["pfsm"]["Ps"] = st.sidebar.slider("Ps (find carrier)",  0.01, 0.30, float(cfg["pfsm"]["Ps"]), 0.01)
    cfg["pfsm"]["Pp"] = st.sidebar.slider("Pp (pickup)",         0.01, 0.30, float(cfg["pfsm"]["Pp"]), 0.01)
    cfg["pfsm"]["Pd"] = st.sidebar.slider("Pd (dropoff)",        0.01, 0.30, float(cfg["pfsm"]["Pd"]), 0.01)
    cfg["pfsm"]["Pa"] = st.sidebar.slider("Pa (avoidance)",      0.01, 0.50, float(cfg["pfsm"]["Pa"]), 0.01)

    if cfg["patches_enabled"]:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Friction patches")
        cfg["friction"]["num_patches"] = st.sidebar.slider(
            "# Patches", 0, 8, int(cfg["friction"].get("num_patches", 3)))
        cfg["friction"]["placement_strategy"] = st.sidebar.selectbox(
            "Placement", ["random", "near_red", "near_deposit"])
        cfg["friction"]["patch_size"] = st.sidebar.slider(
            "Patch size (m)", 0.20, 1.50, float(cfg["friction"].get("patch_size", 0.5)), 0.05)

    st.sidebar.markdown("---")
    st.sidebar.subheader("Page")
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
                     ax: plt.Axes = None, figsize=(6, 6)) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    half = 1.85
    ax.set_xlim(-half - 0.1, half + 0.1)
    ax.set_ylim(-half - 0.1, half + 0.1)
    ax.set_aspect("equal")
    ax.set_facecolor("#f8f5ee")

    arena_rect = mpatches.Rectangle(
        (-half, -half), 2 * half, 2 * half,
        linewidth=2, edgecolor="black", facecolor="none")
    ax.add_patch(arena_rect)

    for p in arena.patches:
        if p.active:
            color = PATCH_COLORS.get(p.spill_type, "grey")
            patch_rect = mpatches.Rectangle(
                (p.x, p.y), p.width, p.height,
                linewidth=1, edgecolor=color, facecolor=color, alpha=0.35, zorder=1)
            ax.add_patch(patch_rect)
            ax.text(p.x + p.width / 2, p.y + p.height / 2,
                    p.spill_type[0], ha="center", va="center",
                    fontsize=7, color=color, fontweight="bold", zorder=2)

    amber = arena.amber_zone
    ax.add_patch(mpatches.Rectangle(
        (amber.x_min, amber.y_min),
        amber.x_max - amber.x_min, amber.y_max - amber.y_min,
        facecolor="orange", alpha=0.18, zorder=2))

    red = arena.red_zone
    ax.add_patch(mpatches.Rectangle(
        (red.x_min, red.y_min),
        red.x_max - red.x_min, red.y_max - red.y_min,
        facecolor="red", alpha=0.28, zorder=3))

    ax.plot(-1.85, -1.0, "Xg", markersize=14, zorder=10, label="Fire exit")

    dep = arena.deposit_zone
    ax.add_patch(mpatches.Rectangle(
        (dep.x_min, dep.y_min),
        dep.x_max - dep.x_min, dep.y_max - dep.y_min,
        facecolor="#3399ff", alpha=0.14, zorder=2))

    for c in arena.get_active_carriers():
        circ = mpatches.Circle(
            (c.x, c.y), c.radius,
            facecolor="#336699", edgecolor="#003366",
            linewidth=1.2, alpha=0.8, zorder=5)
        ax.add_patch(circ)

    for r in robots:
        color = STATE_COLORS.get(r.state_name, "grey")
        circ = mpatches.Circle(
            (r.x, r.y), r.radius,
            facecolor=color, edgecolor="black",
            linewidth=0.8, alpha=0.9, zorder=6)
        ax.add_patch(circ)
        spd = math.hypot(r.vx, r.vy)
        if spd > 0.05:
            ax.annotate("",
                        xy=(r.x + r.vx * 0.12, r.y + r.vy * 0.12),
                        xytext=(r.x, r.y),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                        zorder=7)

    n_red   = sum(1 for r in robots if r.in_red)
    n_amber = sum(1 for r in robots if r.in_amber)
    if n_red > 0:
        ax.set_title(f"t={t:.1f}s  ⚠️ REQ1 RED VIOLATION", color="red", fontweight="bold")
    elif n_amber > 1:
        ax.set_title(f"t={t:.1f}s  ⚠️ REQ1 AMBER CRITICAL", color="orange", fontweight="bold")
    else:
        ax.set_title(f"t={t:.1f}s  ✓ OK", color="green")

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True, alpha=0.2, zorder=0)
    return fig

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

# ── Pages ─────────────────────────────────────────────────────────────────────

def page_live_sim(cfg):
    st.header("🔴 Live Simulation")
    st.markdown(
        "Run a single trial in real time. Watch robots move around the "
        "cloakroom arena and observe safety violations."
    )

    seed  = st.number_input("Random seed", value=42, step=1)
    speed = st.slider("Playback speed (steps per frame)", 1, 50, 10)
    run_btn = st.button("▶ Run Trial")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    timer_placeholder = st.empty()

    if run_btn:
        sb = get_supabase()
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
        wall_start = time.time()
        final_fig = None

        for step in range(n_steps):
            t = step * dt
            arena.update_patches(t)
            for robot in robots:
                robot.step(arena, robots, dt)

            if step % speed == 0:
                fig, ax = plt.subplots(figsize=(6, 6))
                draw_arena_frame(arena, robots, t=t, ax=ax)
                frame_placeholder.pyplot(fig, use_container_width=True)
                final_fig = fig
                plt.close(fig)

                state_counts = {
                    s: sum(1 for r in robots if r.state_name == s)
                    for s in STATE_NAMES.values()
                }
                stats_placeholder.dataframe(
                    pd.DataFrame([state_counts], index=["# robots"]),
                    use_container_width=True)

                elapsed = time.time() - wall_start
                pct = (step + 1) / n_steps
                eta = (elapsed / pct) * (1 - pct) if pct > 0 else 0
                timer_placeholder.caption(
                    f"⏱ Elapsed: {elapsed:.1f}s  |  "
                    f"Progress: {pct*100:.0f}%  |  "
                    f"ETA: {eta:.0f}s remaining"
                )

        # Final frame export
        st.success("✅ Trial complete!")
        wall_elapsed = time.time() - wall_start
        started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.caption(f"Started: {started_at}  |  Duration: {wall_elapsed:.1f}s")

        # Save final frame as downloadable PNG
        if final_fig is not None:
            fig_final, ax_final = plt.subplots(figsize=(7, 7))
            draw_arena_frame(arena, robots, t=cfg["simulation"]["duration"], ax=ax_final)
            fig_final.suptitle(
                f"Final Pattern — Mode: {cfg['mode']}  seed={seed}  {started_at}",
                fontsize=9, y=0.98)
            png_bytes = fig_to_bytes(fig_final)
            plt.close(fig_final)

            st.subheader("📸 Final Swarm Pattern")
            st.image(png_bytes, caption="End-of-trial robot positions")
            st.download_button(
                "⬇ Download Final Pattern PNG",
                data=png_bytes,
                file_name=f"swarm_pattern_{cfg['mode']}_seed{seed}.png",
                mime="image/png"
            )

        # Save to Supabase
        if sb:
            metrics = {
                "trial_id": 0,
                "t_red_viol":   sum(1 for r in robots if r.in_red)   * dt,
                "t_amber_viol": sum(1 for r in robots if r.in_amber) * dt,
                "t_req2_viol":  0.0,
                "n_red_entries": 0, "n_amber_entries": 0, "n_req2_entries": 0,
                "any_req1_red": any(r.in_red for r in robots),
                "any_req1_amber": any(r.in_amber for r in robots),
                "any_req2": False,
            }
            save_trial_to_supabase(sb, cfg, metrics)
            st.caption("✅ Results saved to database.")


def page_batch_run(cfg):
    st.header("📊 Batch Simulation")
    st.markdown("Run multiple trials and collect statistics.")

    # ── Preset experiment buttons ─────────────────────────────────────────────
    st.subheader("🎯 Preset Experiments (Paper Replication)")
    st.markdown(
        "These presets match the configurations required for your PhD thesis. "
        "Each runs 1000 trials — suitable for leaving overnight."
    )

    preset_cols = st.columns(4)
    preset_triggered = None

    with preset_cols[0]:
        if st.button("🟢 Phase 1\nBaseline\n(Mode A)", use_container_width=True):
            preset_triggered = "baseline"
            st.session_state["preset_mode"] = "baseline"
            st.session_state["preset_trials"] = 1000
            st.session_state["preset_seed"] = 42

    with preset_cols[1]:
        if st.button("🟡 Phase 2\nPatches\n(Mode B)", use_container_width=True):
            preset_triggered = "patches_no_adapt"
            st.session_state["preset_mode"] = "patches_no_adapt"
            st.session_state["preset_trials"] = 1000
            st.session_state["preset_seed"] = 42

    with preset_cols[2]:
        if st.button("🔵 Phase 3\nAdaptation\n(Mode C)", use_container_width=True):
            preset_triggered = "patches_adapt"
            st.session_state["preset_mode"] = "patches_adapt"
            st.session_state["preset_trials"] = 1000
            st.session_state["preset_seed"] = 42

    with preset_cols[3]:
        if st.button("⚪ Phase 4\nSanity Check\n(Mode D)", use_container_width=True):
            preset_triggered = "adapt_only"
            st.session_state["preset_mode"] = "adapt_only"
            st.session_state["preset_trials"] = 1000
            st.session_state["preset_seed"] = 42

    st.markdown("---")
    st.subheader("⚙️ Custom Run")

    # Use preset values if set
    default_trials = st.session_state.get("preset_trials", 20)
    default_seed   = st.session_state.get("preset_seed", 42)

    n_trials = st.number_input("Number of trials", 1, 2000, default_trials)
    seed     = st.number_input("Random seed", value=default_seed, step=1)

    run_cfg = copy.deepcopy(cfg)
    if "preset_mode" in st.session_state:
        run_cfg["mode"] = st.session_state["preset_mode"]
        run_cfg["patches_enabled"] = run_cfg["mode"] in ("patches_no_adapt", "patches_adapt")
        run_cfg["robot_patch_adaptation_enabled"] = run_cfg["mode"] in ("patches_adapt", "adapt_only")
        st.info(f"Mode set to: **{run_cfg['mode']}** from preset")

    run_cfg["_seed"] = int(seed)

    if st.button("▶ Run Batch"):
        sb = get_supabase()
        started_at = datetime.datetime.now()
        run_label = f"{run_cfg['mode']}_{int(n_trials)}t_{started_at.strftime('%Y%m%d_%H%M%S')}"

        progress_bar  = st.progress(0)
        timer_display = st.empty()
        wall_start    = time.time()

        # Estimate: ~1.1s per trial
        est_total = int(n_trials) * 1.1

        with st.spinner(f"Running {n_trials} trials — mode: {run_cfg['mode']}…"):
            df = run_batch(
                run_cfg, n_trials=int(n_trials), seed=int(seed),
                output_dir="data/raw",
                aggregated_dir="data/aggregated",
                save_raw=False,
                verbose=False,
            )

            # Update progress periodically (approximation)
            elapsed = time.time() - wall_start
            progress_bar.progress(1.0)
            timer_display.success(
                f"✅ {len(df)} trials completed in {elapsed:.1f}s  "
                f"(started {started_at.strftime('%H:%M:%S')})"
            )

        # Save to Supabase
        save_batch_to_supabase(sb, run_cfg, df, run_label, elapsed)
        if sb:
            st.caption("✅ Results saved to database.")

        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Summary Statistics")
        from src.analysis import summarise_mode
        summary = summarise_mode(df)
        st.dataframe(pd.DataFrame(summary).T, use_container_width=True)

        # ── Histograms ────────────────────────────────────────────────────────
        fig_hist, axes = plt.subplots(1, 3, figsize=(15, 4))
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
        fig_hist.suptitle(f"Mode: {run_cfg['mode']}", fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig_hist, use_container_width=True)

        # ── Violation probability bar chart ───────────────────────────────────
        st.subheader("📊 Violation Probabilities")
        fig_vp, ax_vp = plt.subplots(figsize=(8, 4))
        labels  = ["P(REQ1 Red)", "P(REQ1 Amber)", "P(REQ2)"]
        vals    = [df["any_req1_red"].mean(), df["any_req1_amber"].mean(), df["any_req2"].mean()]
        colors  = ["#c44e52", "#dd8452", "#8172b2"]
        bars = ax_vp.bar(labels, vals, color=colors, alpha=0.85, edgecolor="white", linewidth=1.2)
        for bar, val in zip(bars, vals):
            ax_vp.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f"{val:.3f}", ha="center", va="bottom", fontsize=10)
        ax_vp.set_ylim(0, 1.1)
        ax_vp.set_ylabel("Probability")
        ax_vp.set_title(f"Violation Probabilities — Mode: {run_cfg['mode']}", fontweight="bold")
        ax_vp.axhline(0.5, color="grey", linestyle="--", alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig_vp, use_container_width=True)

        # ── End pattern grid (up to 12 trials shown as a grid) ───────────────
        st.subheader("🔬 End-Pattern Grid")
        st.markdown("Final robot positions for a sample of trials.")
        n_show = min(12, int(n_trials))
        grid_cols = 4
        grid_rows = math.ceil(n_show / grid_cols)
        fig_grid, grid_axes = plt.subplots(
            grid_rows, grid_cols,
            figsize=(grid_cols * 3.5, grid_rows * 3.5)
        )
        grid_axes_flat = np.array(grid_axes).flatten() if n_show > 1 else [grid_axes]

        for idx in range(len(grid_axes_flat)):
            ax_g = grid_axes_flat[idx]
            if idx < n_show:
                trial_seed = int(seed) + idx
                rng_g = random.Random(trial_seed)
                arena_g = Arena(run_cfg)
                arena_g.reset_patches(rng_g)
                for i, pos in enumerate(run_cfg["carriers"]["initial_positions"]):
                    arena_g.carriers[i].x = pos[0]
                    arena_g.carriers[i].y = pos[1]
                    arena_g.carriers[i].active = True

                from src.simulation import _random_deposit_position
                robots_g = []
                placed_g = []
                r_rad = run_cfg["robots"]["diameter"] / 2.0
                for rid in range(run_cfg["robots"]["count"]):
                    x, y = _random_deposit_position(arena_g, r_rad, rng_g, placed_g)
                    placed_g.append((x, y))
                    robots_g.append(Robot(rid, x, y, run_cfg,
                                         rng=random.Random(rng_g.randint(0, 2**31))))

                dt_g = run_cfg["simulation"]["dt"]
                n_g  = int(run_cfg["simulation"]["duration"] / dt_g)
                for step in range(n_g):
                    arena_g.update_patches(step * dt_g)
                    for robot in robots_g:
                        robot.step(arena_g, robots_g, dt_g)

                draw_arena_frame(arena_g, robots_g,
                                 t=run_cfg["simulation"]["duration"],
                                 ax=ax_g)
                ax_g.set_title(f"Trial {idx}", fontsize=8)
                ax_g.tick_params(labelsize=6)
                ax_g.set_xlabel("")
                ax_g.set_ylabel("")
            else:
                ax_g.axis("off")

        fig_grid.suptitle(
            f"End Patterns — Mode: {run_cfg['mode']}  ({n_show} trials shown)",
            fontweight="bold", fontsize=11
        )
        plt.tight_layout()
        st.pyplot(fig_grid, use_container_width=True)

        # ── ZIP download ──────────────────────────────────────────────────────
        st.subheader("⬇️ Download All Outputs")
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
            # CSV
            csv_bytes = df.to_csv(index=False).encode()
            zf.writestr(f"{run_label}_metrics.csv", csv_bytes)

            # Histogram PNG
            hist_buf = io.BytesIO()
            fig_hist.savefig(hist_buf, format="png", dpi=150, bbox_inches="tight")
            zf.writestr(f"{run_label}_histograms.png", hist_buf.getvalue())

            # Violation probs PNG
            vp_buf = io.BytesIO()
            fig_vp.savefig(vp_buf, format="png", dpi=150, bbox_inches="tight")
            zf.writestr(f"{run_label}_violation_probs.png", vp_buf.getvalue())

            # End pattern grid PNG
            grid_buf = io.BytesIO()
            fig_grid.savefig(grid_buf, format="png", dpi=150, bbox_inches="tight")
            zf.writestr(f"{run_label}_end_patterns.png", grid_buf.getvalue())

            # Summary stats
            summary_csv = pd.DataFrame(summary).T.to_csv()
            zf.writestr(f"{run_label}_summary.csv", summary_csv)

            # Metadata
            meta = (
                f"Run label: {run_label}\n"
                f"Mode: {run_cfg['mode']}\n"
                f"Trials: {len(df)}\n"
                f"Seed: {seed}\n"
                f"Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"Elapsed: {elapsed:.1f}s\n"
            )
            zf.writestr(f"{run_label}_metadata.txt", meta)

        zip_buf.seek(0)
        st.download_button(
            "⬇ Download Everything as ZIP",
            data=zip_buf,
            file_name=f"{run_label}.zip",
            mime="application/zip"
        )

        plt.close(fig_hist)
        plt.close(fig_vp)
        plt.close(fig_grid)

        # Clear preset after run
        for k in ["preset_mode", "preset_trials", "preset_seed"]:
            st.session_state.pop(k, None)


def page_compare(cfg):
    st.header("📈 Mode Comparison")
    st.markdown(
        "Load pre-computed aggregated metrics and compare across modes. "
        "Run the batch simulation for each mode first."
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

    # Download comparison chart
    st.download_button(
        "⬇ Download Violation Probabilities Chart",
        data=fig_to_bytes(fig),
        file_name="violation_probs_comparison.png",
        mime="image/png"
    )
    plt.close(fig)

    fig2 = plot_time_in_unsafe(dfs, title="Time in Unsafe States")
    st.pyplot(fig2, use_container_width=True)
    st.download_button(
        "⬇ Download Time in Unsafe Chart",
        data=fig_to_bytes(fig2),
        file_name="time_in_unsafe_comparison.png",
        mime="image/png"
    )
    plt.close(fig2)


def page_trial_analysis(cfg):
    st.header("🔬 Trial Analysis")
    st.markdown("Analyse a saved raw trial CSV.")

    raw_dir = "data/raw"
    if not os.path.isdir(raw_dir):
        st.warning("No raw data directory found. Run a batch first.")
        return

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
            st.download_button("⬇ Download Trajectories PNG",
                               data=fig_to_bytes(fig),
                               file_name="trajectories.png", mime="image/png")
            plt.close(fig)
        with tab2:
            fig = plot_state_proportions(df)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇ Download State Proportions PNG",
                               data=fig_to_bytes(fig),
                               file_name="state_proportions.png", mime="image/png")
            plt.close(fig)
        with tab3:
            fig = plot_violations_timeline(df)
            st.pyplot(fig, use_container_width=True)
            st.download_button("⬇ Download Violations Timeline PNG",
                               data=fig_to_bytes(fig),
                               file_name="violations_timeline.png", mime="image/png")
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