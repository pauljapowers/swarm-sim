"""
arena.py – Arena, zones, carriers, and friction patches for the cloakroom simulation.

Coordinate system: origin at the centre of the 3.70 m × 3.70 m arena.
  x ∈ [-1.85, +1.85],  y ∈ [-1.85, +1.85]
  Red zone:     bottom-left corner, x ∈ [-1.85, -1.00], y ∈ [-1.85, 0.00]
  Amber zone:   0.50 m margin surrounding the red zone (but NOT overlapping the walls)
  Deposit zone: right wall,         x ∈ [+1.00, +1.85], y ∈ [-1.85, +1.85]

  Collection point: left-centre of arena (where robots begin retrieving items)
  Drop-off point:   centre of deposit zone (functional goal target)
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Zone definitions
# ---------------------------------------------------------------------------

class Zone:
    """Axis-aligned rectangular zone."""

    def __init__(self, x_min: float, x_max: float, y_min: float, y_max: float,
                 name: str = ""):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.name = name

    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max

    @property
    def centre(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2.0,
                (self.y_min + self.y_max) / 2.0)

    def __repr__(self):
        return (f"Zone({self.name!r}, x=[{self.x_min},{self.x_max}], "
                f"y=[{self.y_min},{self.y_max}])")


# ---------------------------------------------------------------------------
# Carrier
# ---------------------------------------------------------------------------

@dataclass
class Carrier:
    id: int
    x: float
    y: float
    radius: float
    active: bool = True

    # BUG FIX: store origin positions so reset_carriers() can restore them
    _origin_x: float = field(init=False)
    _origin_y: float = field(init=False)

    def __post_init__(self):
        self._origin_x = self.x
        self._origin_y = self.y

    def __repr__(self):
        return f"Carrier(id={self.id}, pos=({self.x:.3f},{self.y:.3f}), active={self.active})"


# ---------------------------------------------------------------------------
# Friction patches
# ---------------------------------------------------------------------------

MU_EFF = {
    "BASELINE": 1.00,
    "WATER":    0.50,
    "OIL":      0.30,
    "ICE":      0.15,
}

# Human-readable labels and colours for dashboard legend
PATCH_DISPLAY = {
    "WATER": {"label": "Water spill (μ=0.50)", "color": "#3399ff"},
    "OIL":   {"label": "Oil spill (μ=0.30)",   "color": "#cc6600"},
    "ICE":   {"label": "Ice patch (μ=0.15)",   "color": "#aaddff"},
}


@dataclass
class FrictionPatch:
    """
    Axis-aligned rectangular friction patch.
    Activation follows a two-state Markov (on/off) schedule.
    """
    id: int
    x: float
    y: float
    width: float
    height: float
    spill_type: str    # "WATER" | "OIL" | "ICE"
    mu_eff: float = field(init=False)

    on_duration_min: float = 5.0
    on_duration_max: float = 30.0
    off_duration_min: float = 5.0
    off_duration_max: float = 30.0

    active: bool = field(default=False, init=False)
    _next_toggle: float = field(default=0.0, init=False)
    _schedule: List[Tuple[float, bool]] = field(default_factory=list, init=False)
    _schedule_idx: int = field(default=0, init=False)

    def __post_init__(self):
        self.mu_eff = MU_EFF.get(self.spill_type.upper(), 1.0)
        self._schedule = self._generate_schedule(200.0)
        self._schedule_idx = 0
        if self._schedule:
            self.active = self._schedule[0][1]

    def _generate_schedule(self, total_time: float) -> List[Tuple[float, bool]]:
        events: List[Tuple[float, bool]] = []
        t = 0.0
        state = bool(random.getrandbits(1))
        while t < total_time:
            events.append((t, state))
            duration = (random.uniform(self.on_duration_min, self.on_duration_max)
                        if state else
                        random.uniform(self.off_duration_min, self.off_duration_max))
            t += duration
            state = not state
        return events

    def update(self, t: float) -> None:
        if not self._schedule:
            return
        while (self._schedule_idx + 1 < len(self._schedule) and
               self._schedule[self._schedule_idx + 1][0] <= t):
            self._schedule_idx += 1
        self.active = self._schedule[self._schedule_idx][1]

    def contains(self, x: float, y: float) -> bool:
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def overlaps_zone(self, zone: Zone) -> bool:
        return (self.x < zone.x_max and self.x + self.width > zone.x_min and
                self.y < zone.y_max and self.y + self.height > zone.y_min)

    @property
    def display_color(self) -> str:
        return PATCH_DISPLAY.get(self.spill_type.upper(), {}).get("color", "#999999")

    @property
    def display_label(self) -> str:
        return PATCH_DISPLAY.get(self.spill_type.upper(), {}).get("label", self.spill_type)

    def __repr__(self):
        return (f"FrictionPatch(id={self.id}, type={self.spill_type}, "
                f"mu={self.mu_eff}, active={self.active})")


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class Arena:
    """
    Main arena class. Owns zones, carriers, friction patches,
    and the functional collection/drop-off points.
    """

    def __init__(self, cfg: dict):
        ac = cfg["arena"]
        self.width: float  = ac["width"]
        self.height: float = ac["height"]
        self.half_w = self.width / 2.0
        self.half_h = self.height / 2.0

        # Zones
        zc = cfg["zones"]
        rc = zc["red"]
        self.red_zone = Zone(rc["x_min"], rc["x_max"],
                             rc["y_min"], rc["y_max"], "red")

        am = zc["amber"]["margin"]
        self.amber_zone = Zone(
            max(-self.half_w, rc["x_min"] - am),
            min(self.half_w,  rc["x_max"] + am),
            max(-self.half_h, rc["y_min"] - am),
            min(self.half_h,  rc["y_max"] + am),
            "amber",
        )

        dc = zc["deposit"]
        self.deposit_zone = Zone(dc["x_min"], dc["x_max"],
                                 dc["y_min"], dc["y_max"], "deposit")

        # ── Functional points (REQ-F1/F2) ───────────────────────────────
        # Collection point: centre-left of arena — where carriers start
        # and where robots begin a retrieval task.
        self.collection_point: Tuple[float, float] = (
            cfg.get("functional", {}).get("collection_x", -0.5),
            cfg.get("functional", {}).get("collection_y",  0.0),
        )
        # Drop-off point: centre of deposit zone — the functional goal.
        self.dropoff_point: Tuple[float, float] = self.deposit_zone.centre

        # Carriers
        cc = cfg["carriers"]
        self.carriers: List[Carrier] = []
        for i, pos in enumerate(cc["initial_positions"]):
            self.carriers.append(
                Carrier(id=i, x=pos[0], y=pos[1],
                        radius=cc["diameter"] / 2.0)
            )

        # Friction patches
        self.friction_cfg = cfg.get("friction", {})
        self.patches: List[FrictionPatch] = []
        self._patches_enabled = cfg.get("patches_enabled", False)

    # ── Patch management ────────────────────────────────────────────────────

    def reset_patches(self, rng: Optional[random.Random] = None) -> None:
        """(Re-)generate patches for a new trial."""
        self.patches = []
        if not self._patches_enabled:
            return

        fc = self.friction_cfg
        n = fc.get("num_patches", 0)
        size = fc.get("patch_size", 0.5)
        strategy = fc.get("placement_strategy", "random")
        explicit = fc.get("patches", [])

        # BUG FIX: honour _forced_spill override from experiments.py
        forced_spill: Optional[str] = fc.get("_forced_spill", None)

        if explicit:
            for i, p in enumerate(explicit):
                stype = forced_spill if forced_spill else p["spill_type"]
                self.patches.append(FrictionPatch(
                    id=i, x=p["x"], y=p["y"],
                    width=p.get("width", size), height=p.get("height", size),
                    spill_type=stype,
                    on_duration_min=fc.get("on_duration_min", 5),
                    on_duration_max=fc.get("on_duration_max", 30),
                    off_duration_min=fc.get("off_duration_min", 5),
                    off_duration_max=fc.get("off_duration_max", 30),
                ))
        else:
            spill_types = ["WATER", "OIL", "ICE"]
            _rng = rng or random
            for i in range(n):
                # Use forced spill if set, otherwise cycle through types
                stype = forced_spill if forced_spill else spill_types[i % len(spill_types)]
                x, y = self._sample_patch_position(strategy, size, _rng)
                self.patches.append(FrictionPatch(
                    id=i, x=x, y=y,
                    width=size, height=size,
                    spill_type=stype,
                    on_duration_min=fc.get("on_duration_min", 5),
                    on_duration_max=fc.get("on_duration_max", 30),
                    off_duration_min=fc.get("off_duration_min", 5),
                    off_duration_max=fc.get("off_duration_max", 30),
                ))

    def _sample_patch_position(self, strategy: str, size: float,
                                rng: random.Random) -> Tuple[float, float]:
        margin = size / 2.0
        x_lo, x_hi = -self.half_w + margin, self.half_w - size - margin
        y_lo, y_hi = -self.half_h + margin, self.half_h - size - margin

        if strategy == "near_red":
            cx = (self.red_zone.x_min + self.red_zone.x_max) / 2
            cy = (self.red_zone.y_min + self.red_zone.y_max) / 2
            x = float(np.clip(rng.gauss(cx, 0.5), x_lo, x_hi))
            y = float(np.clip(rng.gauss(cy, 0.5), y_lo, y_hi))
        elif strategy == "near_deposit":
            cx = (self.deposit_zone.x_min + self.deposit_zone.x_max) / 2
            cy = (self.deposit_zone.y_min + self.deposit_zone.y_max) / 2
            x = float(np.clip(rng.gauss(cx, 0.5), x_lo, x_hi))
            y = float(np.clip(rng.gauss(cy, 0.5), y_lo, y_hi))
        else:
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
        return x, y

    def update_patches(self, t: float) -> None:
        for p in self.patches:
            p.update(t)

    def get_friction_at(self, x: float, y: float) -> float:
        mu = 1.0
        for p in self.patches:
            if p.active and p.contains(x, y):
                mu = min(mu, p.mu_eff)
        return mu

    def patches_overlapping_red(self) -> List[FrictionPatch]:
        return [p for p in self.patches if p.active and
                p.overlaps_zone(self.red_zone)]

    def patches_overlapping_amber(self) -> List[FrictionPatch]:
        return [p for p in self.patches if p.active and
                p.overlaps_zone(self.amber_zone)]

    def get_active_patch_types(self) -> List[str]:
        """Return list of unique active spill types (for legend rendering)."""
        return list({p.spill_type for p in self.patches if p.active})

    # ── Arena geometry helpers ──────────────────────────────────────────────

    def clamp_to_arena(self, x: float, y: float,
                       radius: float = 0.0) -> Tuple[float, float]:
        x = float(np.clip(x, -self.half_w + radius, self.half_w - radius))
        y = float(np.clip(y, -self.half_h + radius, self.half_h - radius))
        return x, y

    def is_inside(self, x: float, y: float) -> bool:
        return -self.half_w <= x <= self.half_w and -self.half_h <= y <= self.half_h

    # ── Carrier helpers ─────────────────────────────────────────────────────

    def reset_carriers(self) -> None:
        """
        BUG FIX: restore carriers to their original positions AND mark active.
        Previously only set active=True, leaving carriers at moved positions
        from the previous trial — breaking reproducibility across trials.
        """
        for c in self.carriers:
            c.x = c._origin_x
            c.y = c._origin_y
            c.active = True

    def get_active_carriers(self) -> List[Carrier]:
        return [c for c in self.carriers if c.active]

    def remove_carrier(self, carrier_id: int) -> None:
        for c in self.carriers:
            if c.id == carrier_id:
                c.active = False
                break

    def carriers_deposited(self) -> int:
        """Return count of carriers that have been successfully deposited."""
        return sum(1 for c in self.carriers if not c.active)

    # ── Wall helpers ────────────────────────────────────────────────────────

    def distance_to_nearest_wall(self, x: float, y: float) -> float:
        return min(
            x + self.half_w,
            self.half_w - x,
            y + self.half_h,
            self.half_h - y,
        )

    def wall_avoidance_direction(self, x: float, y: float) -> Tuple[float, float]:
        d_left   = x + self.half_w
        d_right  = self.half_w - x
        d_bottom = y + self.half_h
        d_top    = self.half_h - y
        min_d = min(d_left, d_right, d_bottom, d_top)

        if min_d < 0.30:
            if min_d == d_left:
                return (1.0, 0.0)
            elif min_d == d_right:
                return (-1.0, 0.0)
            elif min_d == d_bottom:
                return (0.0, 1.0)
            else:
                return (0.0, -1.0)
        return (0.0, 0.0)
