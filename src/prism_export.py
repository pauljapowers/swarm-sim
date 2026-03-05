"""
prism_export.py – Utilities for extracting PRISM model parameters from
                  LF simulation data and generating .pm model files.

Approach
--------
1.  Load per-timestep raw CSV(s) from LF simulation.
2.  Compute empirical transition probabilities for the PFSM states.
3.  Discretise probabilities via Equal Width Discretisation (EWD) into L1–L5.
4.  Emit a CTMC PRISM model file parameterised by those discrete levels.
5.  Emit example property files (.props) mirroring the paper's CSL formulae.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Equal Width Discretisation
# ---------------------------------------------------------------------------

def ewd(value: float, n_levels: int = 5) -> int:
    """
    Map a probability value ∈ [0, 1] to a discrete level L1–Ln.
    EWD: divide [0,1] into n_levels equal-width bins.
    Returns level index 1..n_levels.
    """
    if np.isnan(value) or value <= 0:
        return 1
    if value >= 1:
        return n_levels
    level = int(value * n_levels) + 1
    return min(level, n_levels)


def ewd_midpoint(level: int, n_levels: int = 5) -> float:
    """Inverse: return midpoint probability for level l."""
    width = 1.0 / n_levels
    return (level - 1 + 0.5) * width


# ---------------------------------------------------------------------------
# Parameter extraction from LF data
# ---------------------------------------------------------------------------

def extract_prism_params(raw_dir: str,
                          sample_every: int = 50,
                          n_levels: int = 5,
                          max_trials: Optional[int] = None) -> Dict[str, object]:
    """
    Scan per-trial CSV files in raw_dir, sample every `sample_every` timesteps,
    compute:
      - Probability of each state at each sampled timestep (averaged over trials).
      - Probability of entering red zone.
      - Probability of ≥2 robots in amber zone.
      - Probability of REQ2 violation.
    Returns dict of parameter name → (mean, ewd_level).
    """
    files = sorted(Path(raw_dir).glob("trial_*.csv"))
    if max_trials:
        files = files[:max_trials]
    if not files:
        raise FileNotFoundError(f"No trial CSV files found in {raw_dir}")

    states = ["SEARCHING", "PICKUP", "DROPOFF",
              "AVOIDANCE_S", "AVOIDANCE_P", "AVOIDANCE_D"]

    # Accumulate per-timestep averages
    state_probs: Dict[str, List[float]] = {s: [] for s in states}
    p_red_list: List[float] = []
    p_amber_critical_list: List[float] = []
    p_req2_list: List[float] = []
    n_robots: int = 5   # default

    for fpath in files:
        df = pd.read_csv(fpath)
        if df.empty:
            continue
        n_robots = df["robot_id"].nunique()
        max_ts = df["timestep"].max()
        sampled_ts = list(range(0, int(max_ts) + 1, sample_every))

        df_s = df[df["timestep"].isin(sampled_ts)]

        for ts in sampled_ts:
            ts_df = df_s[df_s["timestep"] == ts]
            if ts_df.empty:
                continue
            total = len(ts_df)
            for s in states:
                cnt = (ts_df["state"] == s).sum()
                state_probs[s].append(cnt / total if total > 0 else 0.0)
            # REQ1 red: any robot in red
            p_red_list.append(int(ts_df["in_red"].any()))
            # REQ1 amber critical: ≥2 in amber (or 1 in red + 1 in amber)
            n_a = (ts_df["in_amber"].sum() + ts_df["in_red"].sum())
            p_amber_critical_list.append(int(n_a >= 2))
            # REQ2
            p_req2_list.append(int(ts_df["req2_violation"].any()))

    params: Dict[str, object] = {}
    for s in states:
        arr = np.array(state_probs[s])
        mean_p = float(np.mean(arr)) if len(arr) else 0.0
        params[f"p_{s.lower()}"] = (mean_p, ewd(mean_p, n_levels))

    for key, lst in [("p_unsafe_red", p_red_list),
                     ("p_unsafe_amber_critical", p_amber_critical_list),
                     ("p_req2_viol", p_req2_list)]:
        arr = np.array(lst, dtype=float)
        mean_p = float(np.mean(arr)) if len(arr) else 0.0
        params[key] = (mean_p, ewd(mean_p, n_levels))

    params["n_robots"] = n_robots
    params["n_levels"] = n_levels
    return params


# ---------------------------------------------------------------------------
# PRISM model generator
# ---------------------------------------------------------------------------

_PRISM_TEMPLATE = """\
// {model_name}
// Continuous-Time Markov Chain (CTMC) model of cloakroom swarm.
// Parameters derived from {source} LF simulation data.
// Probability levels L1–L{n_levels} via Equal Width Discretisation (EWD).

ctmc

// ── Constants: probability levels ───────────────────────────────────────────
const double L1 = {L1:.4f};
const double L2 = {L2:.4f};
const double L3 = {L3:.4f};
const double L4 = {L4:.4f};
const double L5 = {L5:.4f};

// ── State indices ────────────────────────────────────────────────────────────
// 0=SEARCHING  1=PICKUP  2=DROPOFF
// 3=AVOIDANCE_S  4=AVOIDANCE_P  5=AVOIDANCE_D

// ── Transition rates (derived from EWD levels) ───────────────────────────────
const double r_search_to_pickup   = {r_s2p};    // L{l_s2p}
const double r_pickup_to_dropoff  = {r_p2d};    // L{l_p2d}
const double r_dropoff_to_search  = {r_d2s};    // L{l_d2s}
const double r_to_avoid           = {r_avoid};  // L{l_avoid}
const double r_from_avoid         = 0.5;         // fixed return rate

// ── Module: swarm (counting abstraction, N robots) ───────────────────────────
// n_s = # robots in SEARCHING, etc.

module swarm

  n_s : [0..{N}] init {N};
  n_p : [0..{N}] init 0;
  n_d : [0..{N}] init 0;
  n_as: [0..{N}] init 0;
  n_ap: [0..{N}] init 0;
  n_ad: [0..{N}] init 0;

  // SEARCHING → PICKUP  (one robot per transition)
  [] n_s > 0 -> n_s * r_search_to_pickup :
      (n_s' = n_s - 1) & (n_p' = n_p + 1);

  // PICKUP → DROPOFF
  [] n_p > 0 -> n_p * r_pickup_to_dropoff :
      (n_p' = n_p - 1) & (n_d' = n_d + 1);

  // DROPOFF → SEARCHING (carrier deposited)
  [] n_d > 0 -> n_d * r_dropoff_to_search :
      (n_d' = n_d - 1) & (n_s' = n_s + 1);

  // SEARCHING → AVOIDANCE_S
  [] n_s > 0 -> n_s * r_to_avoid :
      (n_s' = n_s - 1) & (n_as' = n_as + 1);

  // PICKUP → AVOIDANCE_P
  [] n_p > 0 -> n_p * r_to_avoid :
      (n_p' = n_p - 1) & (n_ap' = n_ap + 1);

  // DROPOFF → AVOIDANCE_D
  [] n_d > 0 -> n_d * r_to_avoid :
      (n_d' = n_d - 1) & (n_ad' = n_ad + 1);

  // AVOIDANCE_S → SEARCHING
  [] n_as > 0 -> n_as * r_from_avoid :
      (n_as' = n_as - 1) & (n_s' = n_s + 1);

  // AVOIDANCE_P → PICKUP
  [] n_ap > 0 -> n_ap * r_from_avoid :
      (n_ap' = n_ap - 1) & (n_p' = n_p + 1);

  // AVOIDANCE_D → DROPOFF
  [] n_ad > 0 -> n_ad * r_from_avoid :
      (n_ad' = n_ad - 1) & (n_d' = n_d + 1);

endmodule

// ── Safety labels ─────────────────────────────────────────────────────────────
// Probabilities derived from simulation data.

// Approximation: probability of any robot entering red zone at given state
// is modelled as a separate independent process with rate r_unsafe_red.
const double r_unsafe_red            = {r_unsafe_red:.6f};   // {pct_red:.1f}%
const double r_unsafe_amber_critical = {r_unsafe_amber:.6f}; // {pct_amber:.1f}%
const double r_req2_viol             = {r_req2:.6f};          // {pct_req2:.1f}%

module safety_monitor

  unsafe_red   : bool init false;
  unsafe_amber : bool init false;
  unsafe_req2  : bool init false;

  [] !unsafe_red   -> r_unsafe_red            : (unsafe_red'   = true);
  [] !unsafe_amber -> r_unsafe_amber_critical : (unsafe_amber' = true);
  [] !unsafe_req2  -> r_req2_viol             : (unsafe_req2'  = true);

endmodule

// ── Rewards ───────────────────────────────────────────────────────────────────
rewards "main_states"
  n_s > 0 | n_p > 0 | n_d > 0 : n_s + n_p + n_d;
endrewards

rewards "avoidance_states"
  n_as > 0 | n_ap > 0 | n_ad > 0 : n_as + n_ap + n_ad;
endrewards

// ── Labels ────────────────────────────────────────────────────────────────────
label "unsafe_fireexitsblocked" = unsafe_red = true;
label "unsafe_amber_critical"   = unsafe_amber = true;
label "unsafe_amber"            = unsafe_amber = true;
label "unsafe_red"              = unsafe_red = true;
label "unsafe_density"          = unsafe_req2 = true;
"""


def generate_prism_model(params: Dict, model_name: str, source: str = "baseline",
                          output_path: Optional[str] = None) -> str:
    """
    Generate a PRISM CTMC model string from extracted parameters.
    Returns the model string; optionally writes to output_path.
    """
    n_levels = params.get("n_levels", 5)
    width = 1.0 / n_levels
    level_midpoints = {f"L{i+1}": ewd_midpoint(i + 1, n_levels)
                       for i in range(n_levels)}

    def rate_from_level(level: int) -> float:
        return ewd_midpoint(level, n_levels)

    def get_level(key: str) -> Tuple[float, int]:
        val = params.get(key, (0.01, 1))
        if isinstance(val, tuple):
            return val
        return (val, ewd(val, n_levels))

    _, l_avoid = get_level("p_avoidance_s")
    _, l_s2p   = get_level("p_searching")
    _, l_p2d   = get_level("p_pickup")
    _, l_d2s   = get_level("p_dropoff")
    p_red,    _ = get_level("p_unsafe_red")
    p_amber,  _ = get_level("p_unsafe_amber_critical")
    p_req2,   _ = get_level("p_req2_viol")

    model_str = _PRISM_TEMPLATE.format(
        model_name=model_name,
        source=source,
        n_levels=n_levels,
        **level_midpoints,
        r_s2p   = f"{rate_from_level(l_s2p):.4f}",
        l_s2p   = l_s2p,
        r_p2d   = f"{rate_from_level(l_p2d):.4f}",
        l_p2d   = l_p2d,
        r_d2s   = f"{rate_from_level(l_d2s):.4f}",
        l_d2s   = l_d2s,
        r_avoid = f"{rate_from_level(l_avoid):.4f}",
        l_avoid = l_avoid,
        N       = params.get("n_robots", 5),
        r_unsafe_red   = p_red,
        pct_red        = p_red * 100,
        r_unsafe_amber = p_amber,
        pct_amber      = p_amber * 100,
        r_req2         = p_req2,
        pct_req2       = p_req2 * 100,
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(model_str)
        print(f"[PRISM] Wrote model → {output_path}")
    return model_str


# ---------------------------------------------------------------------------
# Property file generator
# ---------------------------------------------------------------------------

_PROPS_TEMPLATE = """\
// {model_name} – PRISM CSL property specifications
// Mirrors properties from Abeywickrama et al. IEEE ERAS 2025

// ── REQ 1: Fire exit properties ───────────────────────────────────────────────

// P1: Probability of a robot eventually entering the red zone within T timesteps
P=? [ F<=T "unsafe_fireexitsblocked" ]

// P2: Probability of at least 2 robots entering the amber zone within T timesteps
P=? [ F<=T "unsafe_amber_critical" ]

// P3: Probability of a (single) robot entering the amber zone within T timesteps
P=? [ F<=T "unsafe_amber" ]

// P4: Filter – sum over states satisfying red-zone entry
filter(sum, P=? [ X "unsafe_red" ])

// P5: Filter – average over states satisfying red-zone entry
filter(avg, P=? [ X "unsafe_red" ])


// ── REQ 2: Density / stationary robots ───────────────────────────────────────

// P6: Probability of density violation within T timesteps
P=? [ F<=T "unsafe_density" ]


// ── Reward-based progress properties ─────────────────────────────────────────

// R1: Cumulative reward for main states (SEARCHING, PICKUP, DROPOFF) up to T
R{{"main_states"}}=? [C<=T]

// R2: Cumulative reward for avoidance states up to T
R{{"avoidance_states"}}=? [C<=T]


// ── State-rate analysis ───────────────────────────────────────────────────────

// P7: Probability of SEARCHING state reaching level >=3 within first 99 steps
P=? [ F[0,99] (n_s >= 3) ]

// P8: Probability of DROPOFF state reaching level >=3 between steps 100-199
P=? [ F[100,199] (n_d >= 3) ]

// P9: Probability of being in SEARCHING (n_s>=1) within 99 steps
//     while staying in DROPOFF (n_d>=1) beforehand; at least 0.25
P>=0.25 [ n_d>=1 U<=99.0 n_s>=1 ]


// ── Counterexample / witness properties ──────────────────────────────────────

// CTL A1: Invariant – fire exit never blocked (counterexample if violated)
A[ G !"unsafe_red" ]

// CTL E1: Witness – system eventually reaches DROPOFF state
E [ F (n_d > 0) ]


// ── Patch-extended properties (Modes B and C) ─────────────────────────────────

// P10: Probability of red-zone violation (unsafe state) occurring
P=? [ F<=T "unsafe_fireexitsblocked" ]

// (Condition on patch active – handled externally via separate models
//  for baseline vs patches_no_adapt vs patches_adapt)
"""


def generate_prism_props(model_name: str, output_path: Optional[str] = None) -> str:
    props = _PROPS_TEMPLATE.format(model_name=model_name)
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(props)
        print(f"[PRISM] Wrote properties → {output_path}")
    return props


# ---------------------------------------------------------------------------
# All-in-one export for all three modes
# ---------------------------------------------------------------------------

def export_all_prism_models(aggregated_dir: str,
                             raw_dir_base: str,
                             prism_dir: str = "prism_models",
                             n_levels: int = 5,
                             sample_every: int = 50) -> None:
    """
    Generate PRISM models for all three key modes:
      baseline, patches_no_adapt, patches_adapt
    """
    model_map = {
        "baseline":         "cloakroom_baseline.pm",
        "patches_no_adapt": "cloakroom_patches_no_adapt.pm",
        "patches_adapt":    "cloakroom_patches_adapt.pm",
    }

    for mode, filename in model_map.items():
        raw_dir = os.path.join(raw_dir_base, "raw", mode)
        if not os.path.isdir(raw_dir):
            print(f"[PRISM] Skipping {mode!r} – no raw data at {raw_dir}")
            continue
        try:
            params = extract_prism_params(
                raw_dir, sample_every=sample_every, n_levels=n_levels)
        except FileNotFoundError as e:
            print(f"[PRISM] {e}")
            continue

        out_pm   = os.path.join(prism_dir, filename)
        out_props = os.path.join(prism_dir, filename.replace(".pm", ".props"))
        generate_prism_model(params, model_name=filename, source=mode,
                             output_path=out_pm)
        generate_prism_props(model_name=filename, output_path=out_props)

    print(f"[PRISM] All models written to {prism_dir}/")
