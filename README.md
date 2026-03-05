# Swarm Cloakroom Simulation

A reproducible **low-fidelity (LF) swarm robotics simulation** replicating and extending the cloakroom experiment from:

> Abeywickrama, D.B. et al. "Autonomous Robotic Swarms: A Corroborative Approach for Verification and Validation." *IEEE ERAS 2025*.

Developed for PhD research on **safety assurance frameworks for AI multi-agent teams**, focusing on normative concerns, uncertainty quantification, and dynamic assurance.

---

## Structure

```
swarm_sim/
  config.yaml           # Global configuration (modes, friction, PFSM probs)
  requirements.txt
  app.py                # Streamlit dashboard
  src/
    __init__.py
    arena.py            # Arena, zones, carriers, friction patches
    robot.py            # Robot PFSM + friction-aware kinematics
    simulation.py       # Trial runner, batch runner, data logger
    analysis.py         # Metrics, statistics, plots
    experiments.py      # Scripted multi-mode runs (CLI)
    prism_export.py     # PRISM CTMC model generation
  data/
    raw/                # Per-trial CSVs (trial_XXXXX.csv)
    aggregated/         # Per-mode metrics (metrics_<mode>.csv)
    plots/              # Output figures
  prism_models/         # Generated .pm and .props files
  notebooks/
    data_analysis.ipynb
    prism_export.ipynb
  tests/
    test_all.py
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run a quick multi-mode experiment (50 trials per mode)
```bash
python -m src.experiments --n_trials 50 --seed 42 --modes baseline patches_no_adapt patches_adapt
```

### 3. Run 1000-trial baseline (full paper replication)
```bash
python -m src.experiments --n_trials 1000 --seed 42 --modes baseline
```

### 4. Launch the interactive dashboard
```bash
streamlit run app.py
```

---

## Coordinate System

Origin at the **centre** of the 3.70 m × 3.70 m arena.  
Walls at `x, y ∈ [−1.85, +1.85]`.

| Zone         | x               | y               |
|--------------|-----------------|-----------------|
| Red zone     | [−1.85, −1.00]  | [−1.85,  0.00]  |
| Amber zone   | ±0.50 m margin around red zone |
| Deposit zone | [+1.00, +1.85]  | [−1.85, +1.85]  |

---

## Four Experimental Modes

| Mode | `patches_enabled` | `robot_patch_adaptation_enabled` | Description |
|------|:-----------------:|:--------------------------------:|-------------|
| **A – baseline**         | ✗ | ✗ | Original LF replication |
| **B – patches_no_adapt** | ✓ | ✗ | Friction patches, robots unaware |
| **C – patches_adapt**    | ✓ | ✓ | Patches + normative adaptation |
| **D – adapt_only**       | ✗ | ✓ | Adaptation code on (sanity check) |

Set via `mode:` in `config.yaml` or override with `--modes` CLI flag.

---

## Friction Model

Normalised effective friction coefficient `mu_eff ∈ [0, 1]`:

| Surface      | `mu_eff` | Empirical COF |
|--------------|:--------:|:-------------:|
| Baseline     | 1.00     | ~0.4–0.6      |
| Water spill  | 0.50     | ~0.2–0.3      |
| Oil spill    | 0.30     | ~0.08–0.2     |
| Ice patch    | 0.15     | ~0.05–0.1     |

Robot velocity update each timestep:
```python
self.velocity += mu_eff * (desired_velocity - self.velocity)
self.position += self.velocity * dt
```

Patches appear/disappear via a two-state Markov chain with configurable on/off durations.

---

## Safety Requirements

**REQ1 – Do not block fire exit:**
- (i) No robot shall enter the red zone at any time.
- (ii) No more than one robot shall be in the amber zone at any time.

**REQ2 – Swarm density:**
- Fewer than 10% of robots remain stationary (no movement > 10 s) outside the deposit zone at any time.

---

## Data Logging

Each per-trial CSV (`data/raw/<mode>/trial_XXXXX.csv`) contains one row per robot per timestep:

| Column | Description |
|--------|-------------|
| `trial_id`, `timestep`, `time` | Identifiers |
| `robot_id`, `state` | Robot identity and PFSM state |
| `x`, `y`, `vx`, `vy`, `speed` | Kinematics |
| `in_red`, `in_amber`, `in_deposit` | Zone membership |
| `is_stationary` | REQ2 flag |
| `on_patch`, `mu_eff`, `patch_type` | Friction info |
| `patch_near_red`, `patch_near_amber` | Patch/zone proximity |
| `req1_red_violation`, `req1_amber_violation`, `req2_violation` | Swarm-level violation flags |
| `violation_on_patch`, `violation_recent_patch` | Patch attribution |
| `patch_sensed` | Adaptation state |

---

## PRISM Integration

Extract parameters and generate CTMC models:
```python
from src.prism_export import extract_prism_params, generate_prism_model

params = extract_prism_params("data/raw/baseline", sample_every=50)
generate_prism_model(params, "cloakroom_baseline.pm",
                     output_path="prism_models/cloakroom_baseline.pm")
```

Or generate all models at once:
```python
from src.prism_export import export_all_prism_models
export_all_prism_models("data/aggregated", "data", "prism_models")
```

Three models generated:
- `cloakroom_baseline.pm`
- `cloakroom_patches_no_adapt.pm`
- `cloakroom_patches_adapt.pm`

Each paired with a `.props` file containing ~10 CSL properties mirroring the paper.

---

## Configuration Reference (`config.yaml`)

Key settings:
```yaml
mode: "baseline"                  # or patches_no_adapt / patches_adapt / adapt_only
robots:
  count: 5
  max_speed: 2.0                  # m/s
pfsm:
  Ps: 0.05                        # prob find carrier
  Pa: 0.15                        # prob enter avoidance
friction:
  num_patches: 3
  placement_strategy: "random"    # random / near_red / near_deposit
  on_duration_min: 5.0            # seconds
adaptation:
  speed_reduction_factor: 0.50
  extra_clearance_red: 0.20       # extra metres clearance near red zone
```

---

## Citation

```
Abeywickrama, D.B., Lee, S., Bennett, C., Abu-Aisheh, R., Didiot-Cook, T.,
Jones, S., Hauert, S., & Eder, K. (2025). Autonomous Robotic Swarms:
A Corroborative Approach for Verification and Validation.
IEEE Engineering Reliable Autonomous Systems (ERAS 2025).
```
