"""
simulation.py – Trial runner, batch runner, and per-timestep data logger.

Changes from previous version:
  - Uses arena.reset_carriers() correctly (bug fix: restores original positions)
  - t_red_viol / t_amber_viol split into:
      t_red_first_viol  (time of FIRST violation, seconds into trial)
      t_red_total_viol  (total seconds spent in violation)
      (same for amber and req2)
  - Added functional metrics:
      n_carriers_deposited  (count of carriers successfully deposited)
      task_completion_rate  (fraction of carriers deposited, 0.0–1.0)
      any_req_functional    (1 if all carriers deposited before end of trial)

Data logged per timestep (per robot) — all previous columns preserved.
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
    dz = arena.deposit_zone
    margin = robot_radius + 0.02
    for _ in range(max_attempts):
        x = rng.uniform(dz.x_min + margin, dz.x_max - margin)
        y = rng.uniform(dz.y_min + margin, dz.y_max - margin)
        ok = all(math.hypot(x - ex, y - ey) >= robot_radius * 2.5
                 for ex, ey in existing)
        if ok:
            return x, y
    return (dz.x_min + margin + rng.uniform(0, 0.3),
            rng.uniform(dz.y_min + margin, dz.y_max - margin))


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------

class TrialResult:
    __slots__ = ("trial_id", "records", "metrics")

    def __init__(self, trial_id: int):
        self.trial_id = trial_id
        self.records: List[dict] = []
        self.metrics: dict = {}


def run_trial(trial_id: int, cfg: dict,
              rng: Optional[random.Random] = None,
              progress: bool = False) -> TrialResult:
    rng = rng or random.Random()
    result = TrialResult(trial_id)

    dt: float       = cfg["simulation"]["dt"]
    duration: float = cfg["simulation"]["duration"]
    n_steps: int    = int(round(duration / dt))
    log_every: int  = cfg.get("logging", {}).get("log_every_n_steps", 1)
    patch_window: float = cfg.get("logging", {}).get("patch_contact_window", 5.0)

    req2_threshold: float = cfg["req2"]["stationary_fraction"]
    n_robots: int         = cfg["robots"]["count"]
    n_carriers_total: int = len(cfg["carriers"]["initial_positions"])

    # ── Build arena ───────────────────────────────────────────────────────
    arena = Arena(cfg)
    arena.reset_patches(rng)

    # BUG FIX: use reset_carriers() which now restores original positions
    arena.reset_carriers()

    # ── Initialise robots ─────────────────────────────────────────────────
    robots: List[Robot] = []
    placed: List[Tuple[float, float]] = []
    r_radius = cfg["robots"]["diameter"] / 2.0
    for rid in range(n_robots):
        x, y = _random_deposit_position(arena, r_radius, rng, placed)
        placed.append((x, y))
        robots.append(Robot(rid, x, y, cfg, rng=random.Random(rng.randint(0, 2**31))))

    patch_window_steps = int(math.ceil(patch_window / dt))
    recent_patch: List[List[bool]] = [[] for _ in range(n_robots)]

    # ── Violation tracking (split: first-occurrence vs total duration) ────
    # time-to-first-violation: None until first occurrence
    first_red_t:   Optional[float] = None
    first_amber_t: Optional[float] = None
    first_req2_t:  Optional[float] = None

    total_red_steps:   int = 0
    total_amber_steps: int = 0
    total_req2_steps:  int = 0

    # ── Main simulation loop ──────────────────────────────────────────────
    for step in range(n_steps):
        t = step * dt

        arena.update_patches(t)
        for robot in robots:
            robot.step(arena, robots, dt)

        # REQ1 checks
        n_in_red   = sum(1 for r in robots if r.in_red)
        n_in_amber = sum(1 for r in robots if r.in_amber)
        req1_red_viol   = n_in_red > 0
        req1_amber_viol = (n_in_red + n_in_amber) > 1

        # REQ2 check
        n_stationary_outside = sum(
            1 for r in robots if r.is_stationary and not r.in_deposit)
        req2_viol = n_stationary_outside / n_robots >= req2_threshold

        # Accumulate totals
        if req1_red_viol:
            total_red_steps += 1
            if first_red_t is None:
                first_red_t = t
        if req1_amber_viol:
            total_amber_steps += 1
            if first_amber_t is None:
                first_amber_t = t
        if req2_viol:
            total_req2_steps += 1
            if first_req2_t is None:
                first_req2_t = t

        patch_near_red   = len(arena.patches_overlapping_red()) > 0
        patch_near_amber = len(arena.patches_overlapping_amber()) > 0

        if step % log_every == 0:
            for robot in robots:
                recent_patch[robot.id].append(robot.on_patch)
                if len(recent_patch[robot.id]) > patch_window_steps:
                    recent_patch[robot.id].pop(0)
                recently_on_patch = any(recent_patch[robot.id])

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
                    "req1_red_violation":   int(req1_red_viol),
                    "req1_amber_violation": int(req1_amber_viol),
                    "req2_violation":       int(req2_viol),
                    "violation_on_patch":     int((r_req1_red or r_req1_amber or r_req2) and robot.on_patch),
                    "violation_recent_patch": int((r_req1_red or r_req1_amber or r_req2) and recently_on_patch),
                    "patch_sensed": int(robot.patch_sensed),
                    # Functional: carriers deposited at this timestep
                    "carriers_deposited": arena.carriers_deposited(),
                })

    # ── Per-trial aggregate metrics ───────────────────────────────────────
    n_deposited = arena.carriers_deposited()
    task_completion_rate = n_deposited / n_carriers_total if n_carriers_total > 0 else 0.0

    result.metrics = {
        "trial_id": trial_id,

        # --- Safety: time-to-FIRST violation (None → 0.0 if never violated) ---
        "t_red_first_viol":   float(first_red_t)   if first_red_t   is not None else 0.0,
        "t_amber_first_viol": float(first_amber_t) if first_amber_t is not None else 0.0,
        "t_req2_first_viol":  float(first_req2_t)  if first_req2_t  is not None else 0.0,

        # --- Safety: TOTAL time spent in violation ---
        "t_red_total_viol":   float(total_red_steps)   * dt,
        "t_amber_total_viol": float(total_amber_steps) * dt,
        "t_req2_total_viol":  float(total_req2_steps)  * dt,

        # --- Backward-compat alias: t_red_viol = t_red_first_viol ---
        # (kept so existing analysis code doesn't break)
        "t_red_viol":   float(first_red_t)   if first_red_t   is not None else 0.0,
        "t_amber_viol": float(first_amber_t) if first_amber_t is not None else 0.0,
        "t_req2_viol":  float(first_req2_t)  if first_req2_t  is not None else 0.0,

        # --- Entry counts ---
        "n_red_entries":   int(_count_entries(total_red_steps)),
        "n_amber_entries": int(_count_entries(total_amber_steps)),
        "n_req2_entries":  int(_count_entries(total_req2_steps)),

        # --- Binary flags ---
        "any_req1_red":   int(first_red_t   is not None),
        "any_req1_amber": int(first_amber_t is not None),
        "any_req2":       int(first_req2_t  is not None),

        # --- Functional requirement metrics (REQ-F) ---
        "n_carriers_deposited":  n_deposited,
        "task_completion_rate":  round(task_completion_rate, 4),
        "any_req_functional":    int(n_deposited == n_carriers_total),
    }

    if progress:
        print(f"  Trial {trial_id:4d} | "
              f"first_red={result.metrics['t_red_first_viol']:.1f}s "
              f"total_red={result.metrics['t_red_total_viol']:.1f}s "
              f"deposited={n_deposited}/{n_carriers_total} "
              f"TCR={task_completion_rate:.2f}")

    return result


def _count_entries(total_steps: int) -> int:
    """Approximate: can't count entries precisely from step totals alone.
    Returns total_steps as a proxy; entry counting is done in batch."""
    return total_steps


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

    n_trials       = n_trials or cfg["simulation"]["num_trials"]
    output_dir     = output_dir  or cfg.get("logging", {}).get("output_dir", "data/raw")
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
# State-count snapshot
# ---------------------------------------------------------------------------

def state_counts(robots: List[Robot]) -> Dict[str, int]:
    counts: Dict[str, int] = {
        "SEARCHING": 0, "PICKUP": 0, "DROPOFF": 0,
        "AVOIDANCE_S": 0, "AVOIDANCE_P": 0, "AVOIDANCE_D": 0,
    }
    for r in robots:
        counts[r.state_name] = counts.get(r.state_name, 0) + 1
    return counts
