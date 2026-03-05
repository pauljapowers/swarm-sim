"""
db.py – Supabase integration for saving experiment results.
"""
import os
import uuid
from typing import Optional
import pandas as pd
from supabase import create_client, Client

def get_client() -> Optional[Client]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        print("[db] No Supabase credentials found — skipping DB save.")
        return None
    return create_client(url, key)

def save_run_to_supabase(cfg: dict, df_metrics: pd.DataFrame,
                          run_label: str = "", notes: str = "") -> Optional[str]:
    """
    Save a completed experiment run to Supabase.
    Returns the run_id (uuid) or None if credentials missing.
    """
    sb = get_client()
    if sb is None:
        return None

    from .analysis import summarise_mode, bootstrap_ci
    import numpy as np

    # 1. Insert experiment_runs row
    run_row = {
        "run_label":          run_label or cfg.get("mode", "unknown"),
        "mode":               cfg.get("mode", "unknown"),
        "n_trials":           len(df_metrics),
        "seed":               cfg.get("_seed", None),
        "spill_type":         cfg.get("friction", {}).get("_forced_spill", "mixed"),
        "dt":                 cfg["simulation"]["dt"],
        "duration_s":         cfg["simulation"]["duration"],
        "patches_enabled":    cfg.get("patches_enabled", False),
        "adaptation_enabled": cfg.get("robot_patch_adaptation_enabled", False),
        "n_robots":           cfg["robots"]["count"],
        "notes":              notes,
    }
    resp = sb.table("experiment_runs").insert(run_row).execute()
    run_id = resp.data[0]["id"]

    # 2. Insert trial_metrics rows (batch)
    records = df_metrics.rename(columns={
        "any_req1_red":   "any_req1_red",
        "any_req1_amber": "any_req1_amber",
        "any_req2":       "any_req2",
    }).copy()
    records["run_id"] = run_id
    records["patch_contact_viol_count"] = 0
    keep = ["run_id","trial_id","mode","t_red_viol","t_amber_viol","t_req2_viol",
            "n_red_entries","n_amber_entries","n_req2_entries",
            "any_req1_red","any_req1_amber","any_req2","patch_contact_viol_count"]
    keep = [c for c in keep if c in records.columns]
    # Insert in chunks of 500 to stay within API limits
    chunk = records[keep].to_dict(orient="records")
    for i in range(0, len(chunk), 500):
        sb.table("trial_metrics").insert(chunk[i:i+500]).execute()

    # 3. Compute and insert aggregate_stats
    def ci(col):
        arr = df_metrics[col].values.astype(float)
        m, lo, hi = bootstrap_ci(arr)
        return m, lo, hi

    p_red_m,  p_red_lo,  p_red_hi  = ci("any_req1_red")
    p_amb_m,  p_amb_lo,  p_amb_hi  = ci("any_req1_amber")
    p_req2_m, p_req2_lo, p_req2_hi = ci("any_req2")

    agg_row = {
        "run_id": run_id,
        "mode":   cfg.get("mode"),
        "p_req1_red_mean":    p_red_m,  "p_req1_red_ci_lo":  p_red_lo,  "p_req1_red_ci_hi":  p_red_hi,
        "p_req1_amber_mean":  p_amb_m,  "p_req1_amber_ci_lo":p_amb_lo,  "p_req1_amber_ci_hi":p_amb_hi,
        "p_req2_mean":        p_req2_m, "p_req2_ci_lo":      p_req2_lo, "p_req2_ci_hi":      p_req2_hi,
        "t_red_mean_s":   float(df_metrics["t_red_viol"].mean()),
        "t_red_std_s":    float(df_metrics["t_red_viol"].std()),
        "t_red_p95_s":    float(df_metrics["t_red_viol"].quantile(0.95)),
        "t_amber_mean_s": float(df_metrics["t_amber_viol"].mean()),
        "t_amber_std_s":  float(df_metrics["t_amber_viol"].std()),
        "t_req2_mean_s":  float(df_metrics["t_req2_viol"].mean()),
        "t_req2_std_s":   float(df_metrics["t_req2_viol"].std()),
        "n_trials":       len(df_metrics),
    }
    sb.table("aggregate_stats").insert(agg_row).execute()
    print(f"[db] Saved run {run_id} → Supabase ({len(df_metrics)} trials)")
    return run_id