"""
experiments.py – Scripted experiment runs across all four modes and spill types.

Usage:
    python -m src.experiments --n_trials 50 --seed 42 --output_dir data/
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from typing import Dict, Optional

import yaml
import pandas as pd

from .simulation import run_batch
from .analysis import (
    compare_modes, plot_violation_probs, plot_time_in_unsafe,
    load_aggregated
)


# ---------------------------------------------------------------------------
# Mode configurations
# ---------------------------------------------------------------------------

MODE_FLAGS = {
    "baseline":         {"patches_enabled": False, "robot_patch_adaptation_enabled": False},
    "patches_no_adapt": {"patches_enabled": True,  "robot_patch_adaptation_enabled": False},
    "patches_adapt":    {"patches_enabled": True,  "robot_patch_adaptation_enabled": True},
    "adapt_only":       {"patches_enabled": False, "robot_patch_adaptation_enabled": True},
}


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def build_cfg_for_mode(base_cfg: dict, mode: str,
                        spill_override: Optional[str] = None) -> dict:
    """
    Deep-copy base config and apply mode flags.
    Optionally override all spill types to a single type (WATER/OIL/ICE).
    """
    cfg = copy.deepcopy(base_cfg)
    cfg["mode"] = mode
    flags = MODE_FLAGS[mode]
    cfg["patches_enabled"] = flags["patches_enabled"]
    cfg["robot_patch_adaptation_enabled"] = flags["robot_patch_adaptation_enabled"]

    if spill_override and cfg["patches_enabled"]:
        # Override patch types in friction config
        existing = cfg.get("friction", {}).get("patches", [])
        if existing:
            for p in existing:
                p["spill_type"] = spill_override
        else:
            # Use placement strategy with single spill type
            cfg.setdefault("friction", {})["spill_override"] = spill_override
            # We'll handle this in arena.reset_patches by storing spill_override
            # For simplicity, store it in a predictable key:
            cfg["friction"]["_forced_spill"] = spill_override

    return cfg


def run_all_modes(base_cfg: dict,
                  n_trials: int = 100,
                  seed: int = 42,
                  output_base: str = "data",
                  modes: Optional[list] = None,
                  spill_types: Optional[list] = None) -> Dict[str, pd.DataFrame]:
    """
    Run experiments for all requested modes and spill types.
    Returns dict of mode → metrics DataFrame.
    """
    modes = modes or list(MODE_FLAGS.keys())
    spill_types = spill_types or ["mixed"]   # "mixed" uses config default

    results: Dict[str, pd.DataFrame] = {}

    for mode in modes:
        for spill in spill_types:
            label = mode if spill == "mixed" else f"{mode}_{spill.lower()}"
            raw_dir  = os.path.join(output_base, "raw",  label)
            agg_dir  = os.path.join(output_base, "aggregated")
            os.makedirs(raw_dir, exist_ok=True)

            # Skip if only baseline/adapt_only with spill types (not meaningful)
            if spill != "mixed" and mode in ("baseline", "adapt_only"):
                continue

            forced = None if spill == "mixed" else spill
            cfg = build_cfg_for_mode(base_cfg, mode, spill_override=forced)

            print(f"\n{'='*60}")
            print(f" Mode: {mode!r}   Spill: {spill!r}   label: {label!r}")
            print(f"{'='*60}")

            df = run_batch(
                cfg,
                n_trials=n_trials,
                seed=seed,
                output_dir=raw_dir,
                aggregated_dir=agg_dir,
                save_raw=True,
                verbose=True,
            )
            # Override mode column with label for comparison
            df["mode"] = label
            results[label] = df

    return results


def generate_comparison_plots(results: Dict[str, pd.DataFrame],
                               output_dir: str = "data/plots") -> None:
    """Generate comparison plots across modes."""
    os.makedirs(output_dir, exist_ok=True)

    if not results:
        print("[plots] No results to plot.")
        return

    comp_df = compare_modes(results)
    print("\n=== Comparison Table ===")
    print(comp_df.to_string())
    comp_df.to_csv(os.path.join(output_dir, "comparison_table.csv"))

    fig = plot_violation_probs(
        comp_df,
        output_path=os.path.join(output_dir, "violation_probs.png"),
        title="Violation Probabilities Across Modes",
    )
    plt.close(fig)

    fig2 = plot_time_in_unsafe(
        results,
        output_path=os.path.join(output_dir, "time_unsafe_boxplot.png"),
        title="Time in Unsafe States Across Modes",
    )
    plt.close(fig2)

    print(f"[plots] Saved to {output_dir}/")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run swarm cloakroom simulation experiments")
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument("--n_trials",  type=int, default=50,
                        help="Number of trials per mode (default 50; use 1000 for paper)")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--output",    default="data",
                        help="Base output directory")
    parser.add_argument("--modes",     nargs="+",
                        choices=list(MODE_FLAGS.keys()),
                        default=list(MODE_FLAGS.keys()),
                        help="Which modes to run")
    parser.add_argument("--spill_types", nargs="+",
                        choices=["mixed", "WATER", "OIL", "ICE"],
                        default=["mixed"],
                        help="Spill types to test (mixed = config default)")
    parser.add_argument("--plots_only", action="store_true",
                        help="Skip simulation, only generate plots from existing data")
    args = parser.parse_args()

    base_cfg = load_config(args.config)

    if args.plots_only:
        agg_dir = os.path.join(args.output, "aggregated")
        results = load_aggregated(agg_dir)
        if not results:
            print(f"[error] No aggregated data found in {agg_dir}")
            sys.exit(1)
    else:
        results = run_all_modes(
            base_cfg,
            n_trials=args.n_trials,
            seed=args.seed,
            output_base=args.output,
            modes=args.modes,
            spill_types=args.spill_types,
        )

    generate_comparison_plots(results, output_dir=os.path.join(args.output, "plots"))


if __name__ == "__main__":
    main()
