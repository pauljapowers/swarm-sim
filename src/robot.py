"""
robot.py – Robot class with PFSM states and friction-aware kinematics.

States (matching paper's PFSM):
  SEARCHING, PICKUP, DROPOFF,
  AVOIDANCE_S, AVOIDANCE_P, AVOIDANCE_D

Dynamics incorporate a first-order response scaled by mu_eff:
  velocity += mu_eff * (desired_velocity - velocity)
  position += velocity * dt
"""

from __future__ import annotations

import math
import random
from enum import IntEnum
from typing import List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .arena import Arena, Carrier


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------

class RobotState(IntEnum):
    SEARCHING   = 0
    PICKUP      = 1
    DROPOFF     = 2
    AVOIDANCE_S = 3
    AVOIDANCE_P = 4
    AVOIDANCE_D = 5


STATE_NAMES = {
    RobotState.SEARCHING:   "SEARCHING",
    RobotState.PICKUP:      "PICKUP",
    RobotState.DROPOFF:     "DROPOFF",
    RobotState.AVOIDANCE_S: "AVOIDANCE_S",
    RobotState.AVOIDANCE_P: "AVOIDANCE_P",
    RobotState.AVOIDANCE_D: "AVOIDANCE_D",
}


# ---------------------------------------------------------------------------
# Robot
# ---------------------------------------------------------------------------

class Robot:
    """
    Individual robot with probabilistic finite-state machine behaviour
    and friction-modulated kinematics.
    """

    def __init__(self, robot_id: int, x: float, y: float, cfg: dict,
                 rng: Optional[random.Random] = None):
        self.id = robot_id
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.desired_vx = 0.0
        self.desired_vy = 0.0

        self.rng = rng or random.Random()

        # Config shortcuts
        rc = cfg["robots"]
        self.radius = rc["diameter"] / 2.0
        self.max_speed: float = rc["max_speed"]
        self._base_max_speed: float = rc["max_speed"]
        self.avoidance_range: float = rc["avoidance_range"]
        self.avoidance_margin: float = rc["avoidance_margin"]
        self.heading_change_interval: float = rc["heading_change_interval"]

        pfsm = cfg["pfsm"]
        self.Ps = pfsm["Ps"]
        self.Pp = pfsm["Pp"]
        self.Pd = pfsm["Pd"]
        self.Pa = pfsm["Pa"]

        self.dt: float = cfg["simulation"]["dt"]
        self.stationary_threshold: float = cfg["req2"]["stationary_threshold"]

        # Adaptation config
        adapt = cfg.get("adaptation", {})
        self.speed_reduction_factor: float = adapt.get("speed_reduction_factor", 0.5)
        self.extra_clearance_red: float     = adapt.get("extra_clearance_red", 0.20)
        self.slip_threshold: float          = adapt.get("slip_detection_threshold", 0.30)
        self.slip_window: int               = adapt.get("slip_detection_window", 5)
        self.adaptation_enabled: bool       = cfg.get("robot_patch_adaptation_enabled", False)

        # PFSM state
        self.state = RobotState.SEARCHING
        self._pre_avoidance_state = RobotState.SEARCHING
        self.carried_carrier_id: Optional[int] = None

        # Heading and timing
        self._heading: float = self.rng.uniform(0, 2 * math.pi)
        self._heading_timer: float = 0.0

        # REQ2 stationary tracking
        self._stationary_timer: float = 0.0
        self._last_pos: Tuple[float, float] = (x, y)
        self._movement_threshold: float = 0.01   # 1 cm counts as moving

        # Adaptation / slip detection
        self.patch_sensed: bool = False
        self._slip_history: List[float] = []     # recent direction errors (rad)
        self._mu_eff_current: float = 1.0

        # Zone membership (updated externally each step for logging)
        self.in_red:     bool = False
        self.in_amber:   bool = False
        self.in_deposit: bool = False
        self.on_patch:   bool = False
        self.mu_eff:     float = 1.0
        self.patch_type: str = "NONE"

    # ── Properties ──────────────────────────────────────────────────────────

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)

    @property
    def is_stationary(self) -> bool:
        """True if robot has not moved for > stationary_threshold seconds."""
        return self._stationary_timer >= self.stationary_threshold

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state]

    # ── Main update step ────────────────────────────────────────────────────

    def step(self, arena: "Arena", all_robots: List["Robot"],
             dt: Optional[float] = None) -> None:
        """
        Advance one simulation timestep.
          1. Query friction at current position.
          2. Possibly update patch_sensed (if adaptation enabled).
          3. Run PFSM transition logic to compute desired_velocity.
          4. Apply friction-modulated kinematics.
          5. Clamp to arena walls.
          6. Update zone membership and stationary timer.
        """
        dt = dt or self.dt

        # 1. Friction query
        mu = arena.get_friction_at(self.x, self.y)
        self._mu_eff_current = mu
        self.mu_eff = mu

        # On-patch flag
        self.on_patch = (mu < 1.0)
        if self.on_patch:
            # Find patch type for logging
            for p in arena.patches:
                if p.active and p.contains(self.x, self.y):
                    self.patch_type = p.spill_type
                    break
        else:
            self.patch_type = "NONE"

        # 2. Adaptation: speed and clearance
        if self.adaptation_enabled:
            self._update_slip_detection()
            if self.patch_sensed:
                self.max_speed = self._base_max_speed * self.speed_reduction_factor
            else:
                self.max_speed = self._base_max_speed
        else:
            self.max_speed = self._base_max_speed

        # 3. PFSM + desired velocity
        self._update_pfsm(arena, all_robots)

        # 4. Friction-modulated velocity update
        self.vx += mu * (self.desired_vx - self.vx)
        self.vy += mu * (self.desired_vy - self.vy)

        # Cap to max_speed
        spd = math.hypot(self.vx, self.vy)
        if spd > self.max_speed:
            scale = self.max_speed / spd
            self.vx *= scale
            self.vy *= scale

        # Integrate position
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt

        # Wall clamping with bounce
        nx, ny = arena.clamp_to_arena(nx, ny, self.radius)
        if abs(nx - (self.x + self.vx * dt)) > 1e-9:
            self.vx = -self.vx * 0.3   # soft wall bounce
        if abs(ny - (self.y + self.vy * dt)) > 1e-9:
            self.vy = -self.vy * 0.3

        self.x, self.y = nx, ny

        # 5. Zone membership
        self.in_red     = arena.red_zone.contains(self.x, self.y)
        self.in_amber   = (arena.amber_zone.contains(self.x, self.y) and
                           not self.in_red)
        self.in_deposit = arena.deposit_zone.contains(self.x, self.y)

        # 6. Stationary timer (REQ2)
        dx = self.x - self._last_pos[0]
        dy = self.y - self._last_pos[1]
        dist = math.hypot(dx, dy)
        if dist < self._movement_threshold:
            self._stationary_timer += dt
        else:
            self._stationary_timer = 0.0
        self._last_pos = (self.x, self.y)

    # ── PFSM logic ───────────────────────────────────────────────────────────

    def _update_pfsm(self, arena: "Arena",
                     all_robots: List["Robot"]) -> None:
        """Compute desired_velocity and handle state transitions."""
        state = self.state

        if state == RobotState.SEARCHING:
            self._behaviour_searching(arena, all_robots)
        elif state == RobotState.PICKUP:
            self._behaviour_pickup(arena, all_robots)
        elif state == RobotState.DROPOFF:
            self._behaviour_dropoff(arena, all_robots)
        elif state in (RobotState.AVOIDANCE_S,
                       RobotState.AVOIDANCE_P,
                       RobotState.AVOIDANCE_D):
            self._behaviour_avoidance(arena, all_robots)

    def _behaviour_searching(self, arena: "Arena",
                              all_robots: List["Robot"]) -> None:
        # Check avoidance first
        avoid_dir = self._compute_avoidance_direction(arena, all_robots)
        if avoid_dir is not None:
            dx, dy = avoid_dir
            self.desired_vx = dx * self.max_speed
            self.desired_vy = dy * self.max_speed
            if self.rng.random() < self.Pa:
                self._pre_avoidance_state = RobotState.SEARCHING
                self.state = RobotState.AVOIDANCE_S
            return

        # Try to find carrier
        active_carriers = arena.get_active_carriers()
        if active_carriers and self.rng.random() < self.Ps:
            # Pick nearest carrier
            nearest = min(active_carriers,
                          key=lambda c: math.hypot(c.x - self.x, c.y - self.y))
            # Transition to PICKUP
            self.carried_carrier_id = nearest.id
            self.state = RobotState.PICKUP
            return

        # Random walk
        self._heading_timer += self.dt
        if self._heading_timer >= self.heading_change_interval:
            self._heading = self.rng.uniform(0, 2 * math.pi)
            self._heading_timer = 0.0

        # Bias away from red zone if adaptation enabled and patch_sensed
        if self.adaptation_enabled and self.patch_sensed:
            self._heading = self._bias_heading_away_from_red(arena)

        self.desired_vx = math.cos(self._heading) * self.max_speed
        self.desired_vy = math.sin(self._heading) * self.max_speed

    def _behaviour_pickup(self, arena: "Arena",
                          all_robots: List["Robot"]) -> None:
        # Get target carrier
        target = next((c for c in arena.carriers
                       if c.id == self.carried_carrier_id and c.active), None)
        if target is None:
            # Carrier gone (picked by another robot or already deposited)
            self.carried_carrier_id = None
            self.state = RobotState.SEARCHING
            return

        # Check avoidance
        avoid_dir = self._compute_avoidance_direction(arena, all_robots,
                                                       exclude_id=target.id)
        if avoid_dir is not None:
            dx, dy = avoid_dir
            self.desired_vx = dx * self.max_speed
            self.desired_vy = dy * self.max_speed
            if self.rng.random() < self.Pa:
                self._pre_avoidance_state = RobotState.PICKUP
                self.state = RobotState.AVOIDANCE_P
            return

        # Move toward carrier
        dx = target.x - self.x
        dy = target.y - self.y
        dist = math.hypot(dx, dy)
        if dist < 1e-9:
            dist = 1e-9
        # Move at full speed toward carrier
        self.desired_vx = (dx / dist) * self.max_speed
        self.desired_vy = (dy / dist) * self.max_speed

        # Successfully pick up when very close
        pickup_radius = self.radius + target.radius + 0.05
        if dist <= pickup_radius:
            if self.rng.random() < self.Pp:
                self.state = RobotState.DROPOFF

    def _behaviour_dropoff(self, arena: "Arena",
                           all_robots: List["Robot"]) -> None:
        # Check avoidance
        avoid_dir = self._compute_avoidance_direction(arena, all_robots)
        if avoid_dir is not None:
            dx, dy = avoid_dir
            self.desired_vx = dx * self.max_speed
            self.desired_vy = dy * self.max_speed
            if self.rng.random() < self.Pa:
                self._pre_avoidance_state = RobotState.DROPOFF
                self.state = RobotState.AVOIDANCE_D
            return

        # Move toward deposit zone centre
        tx = (arena.deposit_zone.x_min + arena.deposit_zone.x_max) / 2.0
        ty = (arena.deposit_zone.y_min + arena.deposit_zone.y_max) / 2.0

        # Blend random walk with goal bias (0.7 goal, 0.3 random)
        self._heading_timer += self.dt
        if self._heading_timer >= self.heading_change_interval:
            self._heading = self.rng.uniform(0, 2 * math.pi)
            self._heading_timer = 0.0

        goal_dx = tx - self.x
        goal_dy = ty - self.y
        goal_dist = math.hypot(goal_dx, goal_dy)
        if goal_dist > 1e-9:
            goal_dx /= goal_dist
            goal_dy /= goal_dist
        else:
            goal_dx = goal_dy = 0.0

        rand_dx = math.cos(self._heading)
        rand_dy = math.sin(self._heading)

        blend_dx = 0.7 * goal_dx + 0.3 * rand_dx
        blend_dy = 0.7 * goal_dy + 0.3 * rand_dy
        bd = math.hypot(blend_dx, blend_dy)
        if bd > 1e-9:
            blend_dx /= bd
            blend_dy /= bd

        self.desired_vx = blend_dx * self.max_speed
        self.desired_vy = blend_dy * self.max_speed

        # Drop off when inside deposit zone
        if arena.deposit_zone.contains(self.x, self.y):
            if self.rng.random() < self.Pd:
                # Remove carrier
                if self.carried_carrier_id is not None:
                    arena.remove_carrier(self.carried_carrier_id)
                    self.carried_carrier_id = None
                self.state = RobotState.SEARCHING

    def _behaviour_avoidance(self, arena: "Arena",
                             all_robots: List["Robot"]) -> None:
        """Steer away from obstacles; return to pre-avoidance state when clear."""
        avoid_dir = self._compute_avoidance_direction(arena, all_robots)
        if avoid_dir is not None:
            dx, dy = avoid_dir
            self.desired_vx = dx * self.max_speed
            self.desired_vy = dy * self.max_speed
        else:
            # Obstacles cleared; return to previous main state
            self.state = self._pre_avoidance_state
            # Nudge heading away from last obstacle
            self._heading = self.rng.uniform(0, 2 * math.pi)
            self.desired_vx = math.cos(self._heading) * self.max_speed * 0.5
            self.desired_vy = math.sin(self._heading) * self.max_speed * 0.5

    # ── Avoidance helpers ────────────────────────────────────────────────────

    def _compute_avoidance_direction(
        self, arena: "Arena", all_robots: List["Robot"],
        exclude_id: Optional[int] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Return a unit vector pointing away from the nearest obstacle
        within avoidance_range, or None if no obstacle is close enough.
        Obstacles: other robots, active carriers, walls.
        """
        avoid_range = self.avoidance_range

        # Adaptation: expand clearance near red zone
        if self.adaptation_enabled and self.patch_sensed:
            near_red = (
                abs(self.x - arena.red_zone.x_max) < 0.5 or
                abs(self.y - arena.red_zone.y_max) < 0.5
            )
            if near_red:
                avoid_range += self.extra_clearance_red

        repulse_x, repulse_y = 0.0, 0.0
        triggered = False

        # Other robots
        for r in all_robots:
            if r.id == self.id:
                continue
            dx = self.x - r.x
            dy = self.y - r.y
            dist = math.hypot(dx, dy)
            min_dist = (self.radius + r.radius + self.avoidance_margin)
            if dist < avoid_range and dist > 1e-9:
                triggered = True
                weight = max(0, (avoid_range - dist) / avoid_range)
                repulse_x += weight * dx / dist
                repulse_y += weight * dy / dist

        # Active carriers
        for c in arena.get_active_carriers():
            if exclude_id is not None and c.id == exclude_id:
                continue
            dx = self.x - c.x
            dy = self.y - c.y
            dist = math.hypot(dx, dy)
            if dist < avoid_range and dist > 1e-9:
                triggered = True
                weight = max(0, (avoid_range - dist) / avoid_range)
                repulse_x += weight * dx / dist
                repulse_y += weight * dy / dist

        # Walls (simplified)
        wx, wy = arena.wall_avoidance_direction(self.x, self.y)
        if wx != 0 or wy != 0:
            triggered = True
            repulse_x += wx
            repulse_y += wy

        if not triggered:
            return None

        mag = math.hypot(repulse_x, repulse_y)
        if mag < 1e-9:
            # Ambiguous direction – pick random
            a = self.rng.uniform(0, 2 * math.pi)
            return (math.cos(a), math.sin(a))
        return (repulse_x / mag, repulse_y / mag)

    # ── Adaptation helpers ───────────────────────────────────────────────────

    def _update_slip_detection(self) -> None:
        """
        Detect whether robot is slipping by comparing desired heading
        to actual motion heading over the last slip_window timesteps.
        """
        des_spd = math.hypot(self.desired_vx, self.desired_vy)
        act_spd = math.hypot(self.vx, self.vy)
        if des_spd < 0.01 or act_spd < 0.01:
            self._slip_history.append(0.0)
        else:
            des_angle = math.atan2(self.desired_vy, self.desired_vx)
            act_angle = math.atan2(self.vy, self.vx)
            err = abs(math.atan2(math.sin(des_angle - act_angle),
                                  math.cos(des_angle - act_angle)))
            self._slip_history.append(err)

        # Keep only last N samples
        if len(self._slip_history) > self.slip_window:
            self._slip_history.pop(0)

        avg_slip = (sum(self._slip_history) / len(self._slip_history)
                    if self._slip_history else 0.0)

        # Also consider direct mu reading (simpler design abstraction)
        if self._mu_eff_current < 1.0:
            self.patch_sensed = True
        elif avg_slip > self.slip_threshold:
            self.patch_sensed = True
        else:
            self.patch_sensed = False

    def _bias_heading_away_from_red(self, arena: "Arena") -> float:
        """
        Return a heading biased away from the red zone centre.
        Used when adaptation is active.
        """
        red_cx = (arena.red_zone.x_min + arena.red_zone.x_max) / 2
        red_cy = (arena.red_zone.y_min + arena.red_zone.y_max) / 2
        away_angle = math.atan2(self.y - red_cy, self.x - red_cx)
        # Add noise
        noise = self.rng.gauss(0, 0.5)
        return away_angle + noise

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self, x: float, y: float) -> None:
        """Reset robot to initial conditions for a new trial."""
        self.x, self.y = x, y
        self.vx = self.vy = 0.0
        self.desired_vx = self.desired_vy = 0.0
        self.state = RobotState.SEARCHING
        self._pre_avoidance_state = RobotState.SEARCHING
        self.carried_carrier_id = None
        self._heading = self.rng.uniform(0, 2 * math.pi)
        self._heading_timer = 0.0
        self._stationary_timer = 0.0
        self._last_pos = (x, y)
        self.patch_sensed = False
        self._slip_history = []
        self.max_speed = self._base_max_speed
        self.in_red = self.in_amber = self.in_deposit = self.on_patch = False
        self.mu_eff = 1.0
        self.patch_type = "NONE"
