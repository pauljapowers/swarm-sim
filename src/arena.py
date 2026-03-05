"""
arena.py – Arena, zones, carriers, and friction patches for the cloakroom simulation.

Coordinate system: origin at the centre of the 3.70 m × 3.70 m arena.
  x ∈ [-1.85, +1.85],  y ∈ [-1.85, +1.85]
  Red zone:     bottom-left corner, x ∈ [-1.85, -1.00], y ∈ [-1.85, 0.00]
  Amber zone:   0.50 m margin surrounding the red zone (but NOT overlapping the walls)
  Deposit zone: right wall,         x ∈ [+1.00, +1.85], y ∈ [-1.85, +1.85]
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
    radius: float        # LF abstraction: same as robot radius
    active: bool = True  # False once deposited and removed

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


@dataclass
class FrictionPatch:
    """
    Axis-aligned rectangular friction patch.
    Activation follows a two-state Markov (on/off) schedule generated at
    construction time.
    """
    id: int
    x: float           # left edge
    y: float           # bottom edge
    width: float
    height: float
    spill_type: str    # "WATER" | "OIL" | "ICE"
    mu_eff: float = field(init=False)

    # Activation schedule
    on_duration_min: float = 5.0
    on_duration_max: float = 30.0
    off_duration_min: float = 5.0
    off_duration_max: float = 30.0

    # Runtime state
    active: bool = field(default=False, init=False)
    _next_toggle: float = field(default=0.0, init=False)
    _schedule: List[Tuple[float, bool]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.mu_eff = MU_EFF.get(self.spill_type.upper(), 1.0)
        # Pre-generate a schedule of (time, state) events for 200 s
        self._schedule = self._generate_schedule(200.0)
        self._schedule_idx = 0
        if self._schedule:
            self.active = self._schedule[0][1]
            self._schedule_idx = 0

    def _generate_schedule(self, total_time: float) -> List[Tuple[float, bool]]:
        """Generate alternating on/off events up to total_time."""
        events: List[Tuple[float, bool]] = []
        t = 0.0
        # Randomly start in either state
        state = bool(random.getrandbits(1))
        while t < total_time:
            events.append((t, state))
            if state:
                duration = random.uniform(self.on_duration_min, self.on_duration_max)
            else:
                duration = random.uniform(self.off_duration_min, self.off_duration_max)
            t += duration
            state = not state
        return events

    def update(self, t: float) -> None:
        """Update active flag based on current simulation time."""
        if not self._schedule:
            return
        # Walk schedule forward
        while (self._schedule_idx + 1 < len(self._schedule) and
               self._schedule[self._schedule_idx + 1][0] <= t):
            self._schedule_idx += 1
        self.active = self._schedule[self._schedule_idx][1]

    def contains(self, x: float, y: float) -> bool:
        return (self.x <= x <= self.x + self.width and
                self.y <= y <= self.y + self.height)

    def overlaps_zone(self, zone: Zone) -> bool:
        """True if this patch has any overlap with a zone rectangle."""
        return (self.x < zone.x_max and self.x + self.width > zone.x_min and
                self.y < zone.y_max and self.y + self.height > zone.y_min)

    def __repr__(self):
        return (f"FrictionPatch(id={self.id}, type={self.spill_type}, "
                f"mu={self.mu_eff}, active={self.active})")


# ---------------------------------------------------------------------------
# Arena
# ---------------------------------------------------------------------------

class Arena:
    """
    Main arena class.  Owns zones, carriers, and friction patches.
    """

    def __init__(self, cfg: dict):
        ac = cfg["arena"]
        self.width: float  = ac["width"]
        self.height: float = ac["height"]
        self.half_w = self.width / 2.0
        self.half_h = self.height / 2.0

        # Build zones
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

        if explicit:
            for i, p in enumerate(explicit):
                self.patches.append(FrictionPatch(
                    id=i, x=p["x"], y=p["y"],
                    width=p.get("width", size), height=p.get("height", size),
                    spill_type=p["spill_type"],
                    on_duration_min=fc.get("on_duration_min", 5),
                    on_duration_max=fc.get("on_duration_max", 30),
                    off_duration_min=fc.get("off_duration_min", 5),
                    off_duration_max=fc.get("off_duration_max", 30),
                ))
        else:
            spill_types = ["WATER", "OIL", "ICE"]
            _rng = rng or random
            for i in range(n):
                stype = spill_types[i % len(spill_types)]
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
        """Return (x, y) lower-left corner of patch given placement strategy."""
        margin = size / 2.0
        x_lo, x_hi = -self.half_w + margin, self.half_w - size - margin
        y_lo, y_hi = -self.half_h + margin, self.half_h - size - margin

        if strategy == "near_red":
            # Bias within 1 m of red zone centre
            cx = (self.red_zone.x_min + self.red_zone.x_max) / 2
            cy = (self.red_zone.y_min + self.red_zone.y_max) / 2
            x = float(np.clip(rng.gauss(cx, 0.5), x_lo, x_hi))
            y = float(np.clip(rng.gauss(cy, 0.5), y_lo, y_hi))
        elif strategy == "near_deposit":
            cx = (self.deposit_zone.x_min + self.deposit_zone.x_max) / 2
            cy = (self.deposit_zone.y_min + self.deposit_zone.y_max) / 2
            x = float(np.clip(rng.gauss(cx, 0.5), x_lo, x_hi))
            y = float(np.clip(rng.gauss(cy, 0.5), y_lo, y_hi))
        else:  # "random"
            x = rng.uniform(x_lo, x_hi)
            y = rng.uniform(y_lo, y_hi)
        return x, y

    def update_patches(self, t: float) -> None:
        """Advance patch activation schedules to time t."""
        for p in self.patches:
            p.update(t)

    def get_friction_at(self, x: float, y: float) -> float:
        """
        Return the effective friction coefficient at (x, y).
        If multiple active patches cover the point, return the minimum mu_eff.
        Returns 1.0 if no active patch covers the point.
        """
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

    # ── Arena geometry helpers ──────────────────────────────────────────────

    def clamp_to_arena(self, x: float, y: float,
                       radius: float = 0.0) -> Tuple[float, float]:
        """Clamp (x, y) so that a circle of given radius stays inside walls."""
        x = float(np.clip(x, -self.half_w + radius, self.half_w - radius))
        y = float(np.clip(y, -self.half_h + radius, self.half_h - radius))
        return x, y

    def is_inside(self, x: float, y: float) -> bool:
        return -self.half_w <= x <= self.half_w and -self.half_h <= y <= self.half_h

    # ── Carrier helpers ─────────────────────────────────────────────────────

    def reset_carriers(self) -> None:
        for c in self.carriers:
            c.active = True
            # restore to original initial positions from first construction
            # (positions stored from cfg; here we just leave them as-is,
            #  they were set at init and reset_carriers is called between trials)

    def get_active_carriers(self) -> List[Carrier]:
        return [c for c in self.carriers if c.active]

    def remove_carrier(self, carrier_id: int) -> None:
        for c in self.carriers:
            if c.id == carrier_id:
                c.active = False
                break

    # ── Nearest obstacle helpers ────────────────────────────────────────────

    def distance_to_nearest_wall(self, x: float, y: float) -> float:
        """Return minimum distance from (x,y) to any wall."""
        return min(
            x + self.half_w,    # left wall
            self.half_w - x,    # right wall
            y + self.half_h,    # bottom wall
            self.half_h - y,    # top wall
        )

    def wall_avoidance_direction(self, x: float, y: float) -> Tuple[float, float]:
        """
        Return a unit-vector pointing away from the nearest wall,
        or (0, 0) if well clear.
        """
        d_left   = x + self.half_w
        d_right  = self.half_w - x
        d_bottom = y + self.half_h
        d_top    = self.half_h - y
        min_d = min(d_left, d_right, d_bottom, d_top)

        if min_d < 0.30:   # only push if within 30 cm of wall
            if min_d == d_left:
                return (1.0, 0.0)
            elif min_d == d_right:
                return (-1.0, 0.0)
            elif min_d == d_bottom:
                return (0.0, 1.0)
            else:
                return (0.0, -1.0)
        return (0.0, 0.0)
