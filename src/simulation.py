"""
simulation.py – Trial runner, batch runner, and per-timestep data logger.

Data logged per timestep (per robot):
  trial_id, timestep, time, robot_id, state,
  x, y, vx, vy, speed,
  in_red, in_amber, in_deposit, is_stationary,
  on_patch, mu_eff, patch_type,
  patch_near_red, patch_near_amber,
  req1_red_violation, req1_amber_violation,
  req2_violation,
  violation_on_patch, violation_recent_patch,
  patch_sensed (adaptation flag)
"""

from __future__ import annotations

import math
import os
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .arena import Arena
from .robot import Robot, RobotState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_deposit_position(arena, robot_radius: float,
                              rng: random.Random,
                              existing: List[Tuple[float, float]],
                              max_attempts: int = 200) -> Tuple[float, float]:
    """
    Sample a random position inside the deposit zone with minimal separation
    from other robots (anti-overlap initialisation).
    """
    dz = arena.deposit_zone
    margin = robot_radius + 0.02
    for _ in range(max_attempts):
        x = rng.uniform(dz.x_min + margin, dz.x_max - margin)
        y = rng.uniform(dz.y_min + margin, dz.y_max - margin)
        ok = True
        for ex, ey in existing:
            if math.hypot(x - ex, y - ey) < robot_radius * 2.5:
                ok = False
                break
        if ok:
            return x, y
    # Fallback – place in a grid within deposit zone
    return (dz.x_min + margin + rng.uniform(0, 0.3),
            rng.uniform(dz.y_min + margin, dz.y_max - margin))


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

class TrialResult:
    """Holds per-timestep records and per-trial aggregate metrics."""

    __slots__ = ("trial_id", "records", "metrics")

    def __init__(self, trial_id: int):
        self.trial_id = trial_id
        self.records: List[dict] = []
        self.metrics: dict = {}


def run_trial(trial_id: int, cfg: dict,
              rng: Optional[random.Random] = None,
              progress: bool = False) -> TrialResult:
    """
    Run a single trial and return a TrialResult.

    Parameters
    ----------
    trial_id   : unique integer identifier
    cfg        : parsed config dict (from config.yaml)
    rng        : optional seeded Random instance for reproducibility
    progress   : if True, print basic progress
    """
    rng = rng or random.Random()
    result = TrialResult(trial_id)

    dt: float       = cfg["simulation"]["dt"]
    duration: float = cfg["simulation"]["duration"]
    n_steps: int    = int(round(duration / dt))
    log_every: int  = cfg.get("logging", {}).get("log_every_n_steps", 1)
    patch_window: float = cfg.get("logging", {}).get("patch_contact_window", 5.0)

    req2_threshold: float = cfg["req2"]["stationary_fraction"]
    n_robots: int         = cfg["robots"]["count"]

    # ── Build arena & reset carriers and patches ─────────────────────────
    arena = Arena(cfg)
    arena.reset_patches(rng)

    # Restore carrier positions from config each trial
    for i, pos in enumerate(cfg["carriers"]["initial_positions"]):
        arena.carriers[i].x = pos[0]
        arena.carriers[i].y = pos[1]
        arena.carriers[i].active = True

    # ── Initialise robots in deposit zone ────────────────────────────────
    robots: List[Robot] = []
    placed: List[Tuple[float, float]] = []
    r_radius = cfg["robots"]["diameter"] / 2.0
    for rid in range(n_robots):
        x, y = _random_deposit_position(arena, r_radius, rng, placed)
        placed.append((x, y))
        robots.append(Robot(rid, x, y, cfg, rng=random.Random(rng.randint(0, 2**31))))

    # Per-robot recent-patch tracker (list of booleans per step)
    recent_patch: List[List[bool]] = [[] for _ in range(n_robots)]
    patch_window_steps = int(math.ceil(patch_window / dt))

    # ── Main simulation loop ─────────────────────────────────────────────
    for step in range(n_steps):
        t = step * dt

        # Update friction patches
        arena.update_patches(t)

        # Step all robots
        for robot in robots:
            robot.step(arena, robots, dt)

        # ── Collect metrics ───────────────────────────────────────────────

        # REQ1 check
        n_in_red   = sum(1 for r in robots if r.in_red)
        n_in_amber = sum(1 for r in robots if r.in_amber)
        # REQ1(i): any robot in red
        req1_red_viol = n_in_red > 0
        # REQ1(ii): >1 robot in amber zone (counting red as amber per paper)
        req1_amber_viol = (n_in_red + n_in_amber) > 1

        # REQ2 check
        n_stationary_outside = sum(
            1 for r in robots
            if r.is_stationary and not r.in_deposit
        )
        req2_viol = n_stationary_outside / n_robots >= req2_threshold

        # Patch overlap flags
        patch_near_red   = len(arena.patches_overlapping_red()) > 0
        patch_near_amber = len(arena.patches_overlapping_amber()) > 0

        # Log if required
        if step % log_every == 0:
            for robot in robots:
                # Update recent-patch history
                recent_patch[robot.id].append(robot.on_patch)
                if len(recent_patch[robot.id]) > patch_window_steps:
                    recent_patch[robot.id].pop(0)
                recently_on_patch = any(recent_patch[robot.id])

                # Per-robot violation attribution
                r_req1_red   = req1_red_viol   and robot.in_red
                r_req1_amber = req1_amber_viol and (robot.in_red or robot.in_amber)
                r_req2       = req2_viol        and robot.is_stationary and not robot.in_deposit

                result.records.append({
                    "trial_id":      trial_id,
                    "timestep":      step,
                    "time":          round(t, 4),
                    "robot_id":      robot.id,
                    "state":         robot.state_name,
                    "x":             round(robot.x, 4),
                    "y":             round(robot.y, 4),
                    "vx":            round(robot.vx, 4),
                    "vy":            round(robot.vy, 4),
                    "speed":         round(robot.speed, 4),
                    "in_red":        int(robot.in_red),
                    "in_amber":      int(robot.in_amber),
                    "in_deposit":    int(robot.in_deposit),
                    "is_stationary": int(robot.is_stationary),
                    "on_patch":      int(robot.on_patch),
                    "mu_eff":        round(robot.mu_eff, 3),
                    "patch_type":    robot.patch_type,
                    "patch_near_red":   int(patch_near_red),
                    "patch_near_amber": int(patch_near_amber),
                    # REQ1 / REQ2 event flags (swarm-level)
                    "req1_red_violation":   int(req1_red_viol),
                    "req1_amber_violation": int(req1_amber_viol),
                    "req2_violation":       int(req2_viol),
                    # Violation × patch attribution
                    "violation_on_patch":     int((r_req1_red or r_req1_amber or r_req2) and robot.on_patch),
                    "violation_recent_patch": int((r_req1_red or r_req1_amber or r_req2) and recently_on_patch),
                    # Adaptation
                    "patch_sensed": int(robot.patch_sensed),
                })

    # ── Per-trial aggregate metrics ───────────────────────────────────────
    if result.records:
        df = pd.DataFrame(result.records)
        # Collapse to swarm-level (one row per timestep, any-robot flags)
        ts_df = df.groupby("timestep").agg(
            time=("time", "first"),
            req1_red_viol=("req1_red_violation", "max"),
            req1_amber_viol=("req1_amber_violation", "max"),
            req2_viol=("req2_violation", "max"),
        ).reset_index()

        result.metrics = {
            "trial_id":            trial_id,
            "t_red_viol":          float(ts_df["req1_red_viol"].sum() * dt),
            "t_amber_viol":        float(ts_df["req1_amber_viol"].sum() * dt),
            "t_req2_viol":         float(ts_df["req2_viol"].sum() * dt),
            "n_red_entries":       int(ts_df["req1_red_viol"].diff().clip(lower=0).sum()),
            "n_amber_entries":     int(ts_df["req1_amber_viol"].diff().clip(lower=0).sum()),
            "n_req2_entries":      int(ts_df["req2_viol"].diff().clip(lower=0).sum()),
            "any_req1_red":        int(ts_df["req1_red_viol"].any()),
            "any_req1_amber":      int(ts_df["req1_amber_viol"].any()),
            "any_req2":            int(ts_df["req2_viol"].any()),
        }

    if progress:
        print(f"  Trial {trial_id:4d} done | "
              f"t_red={result.metrics.get('t_red_viol', 0):.1f}s "
              f"t_amber={result.metrics.get('t_amber_viol', 0):.1f}s "
              f"t_req2={result.metrics.get('t_req2_viol', 0):.1f}s")

    return result


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_batch(cfg: dict,
              n_trials: Optional[int] = None,
              seed: Optional[int] = None,
              output_dir: Optional[str] = None,
              aggregated_dir: Optional[str] = None,
              save_raw: bool = True,
              verbose: bool = True,
              trial_offset: int = 0) -> pd.DataFrame:
    """
    Run multiple trials and save results.

    Returns
    -------
    DataFrame of per-trial aggregated metrics.
    """
    n_trials    = n_trials or cfg["simulation"]["num_trials"]
    output_dir  = output_dir  or cfg.get("logging", {}).get("output_dir", "data/raw")
    aggregated_dir = aggregated_dir or cfg.get("logging", {}).get("aggregated_dir", "data/aggregated")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(aggregated_dir, exist_ok=True)

    master_rng = random.Random(seed)
    all_metrics: List[dict] = []

    mode = cfg.get("mode", "baseline")

    t0 = time.perf_counter()
    print(f"[Batch] mode={mode!r}  n_trials={n_trials}  seed={seed}")

    for i in range(n_trials):
        tid = trial_offset + i
        trial_rng = random.Random(master_rng.randint(0, 2**31))

        result = run_trial(tid, cfg, rng=trial_rng, progress=False)

        if save_raw and result.records:
            df_raw = pd.DataFrame(result.records)
            out_path = os.path.join(output_dir, f"trial_{tid:05d}.csv")
            df_raw.to_csv(out_path, index=False)

        if result.metrics:
            result.metrics["mode"] = mode
            all_metrics.append(result.metrics)

        if verbose and (i + 1) % max(1, n_trials // 20) == 0:
            elapsed = time.perf_counter() - t0
            pct = (i + 1) / n_trials * 100
            print(f"  {pct:5.1f}%  trial {tid}  elapsed={elapsed:.1f}s")

    df_agg = pd.DataFrame(all_metrics)
    agg_path = os.path.join(aggregated_dir, f"metrics_{mode}.csv")
    df_agg.to_csv(agg_path, index=False)

    elapsed = time.perf_counter() - t0
    print(f"[Batch] complete  {n_trials} trials in {elapsed:.1f}s  → {agg_path}")
    return df_agg


# ---------------------------------------------------------------------------
# State-count snapshot (for PRISM / analysis)
# ---------------------------------------------------------------------------

def state_counts(robots: List[Robot]) -> Dict[str, int]:
    """Return dict of state_name → count for a list of robots."""
    counts: Dict[str, int] = {
        "SEARCHING": 0, "PICKUP": 0, "DROPOFF": 0,
        "AVOIDANCE_S": 0, "AVOIDANCE_P": 0, "AVOIDANCE_D": 0,
    }
    for r in robots:
        counts[r.state_name] = counts.get(r.state_name, 0) + 1
    return counts
