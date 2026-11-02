"""
Approach prediction and safety state machine.

States (in order of severity):
  CLEAR            → No hand detected within warning distance
  WARNING          → Hand approaching or within warning distance
  PROTECTIVE_ACTION → Hand within action distance (simulated stop signal)
  FAULT            → Camera or detection failure

Conservative rules:
  - Does NOT revert from WARNING to CLEAR until clear_hold_frames consecutive
    safe frames (prevents oscillation near boundary).
  - Does NOT revert from FAULT automatically; requires explicit reset().
"""
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Optional, Tuple

STATES = ("CLEAR", "WARNING", "PROTECTIVE_ACTION", "FAULT")


class ApproachPredictor:
    def __init__(self, config: dict):
        pred_cfg = config.get("predictor", {})
        self._vel_window: int = pred_cfg.get("velocity_window", 6)
        self._approach_threshold: float = pred_cfg.get("approach_speed_threshold_px_s", 80.0)
        self._clear_hold: int = pred_cfg.get("clear_hold_frames", 10)

        dz_cfg = config.get("danger_zone", {})
        self._warn_dist: float = float(dz_cfg.get("warning_distance_px", 120))
        self._action_dist: float = float(dz_cfg.get("action_distance_px", 40))

        # History: (distance_px, wall_clock_s)
        self._history: Deque[Tuple[float, float]] = deque(maxlen=self._vel_window)
        self._state: str = "CLEAR"
        self._clear_counter: int = 0   # consecutive safe frames
        self._velocity: float = 0.0    # pixels/s (positive = approaching)

    # ------------------------------------------------------------------
    def update(
        self,
        distance_px: float,
        timestamp: float,
        hand_detected: bool,
        camera_fault: bool,
        detection_fault: bool,
    ) -> str:
        """
        Update internal state given the current distance and context.
        Returns the new state string.
        """
        # FAULT takes priority and is sticky
        if camera_fault or detection_fault:
            self._state = "FAULT"
            self._history.clear()
            return self._state

        if not hand_detected:
            # No hand detected but not faulted — stay in current state
            # (conservative: do not drop to CLEAR immediately)
            if self._state == "CLEAR":
                pass  # remain CLEAR
            else:
                # Already in WARNING/ACTION, keep it until clear_hold satisfied
                pass
            return self._state

        # Hand detected — compute velocity
        self._history.append((distance_px, timestamp))
        self._velocity = self._compute_velocity()

        # Determine new state
        if distance_px <= self._action_dist:
            self._state = "PROTECTIVE_ACTION"
            self._clear_counter = 0

        elif distance_px <= self._warn_dist or (
            self._velocity > self._approach_threshold
            and distance_px <= self._warn_dist * 1.5
        ):
            self._state = "WARNING"
            self._clear_counter = 0

        else:
            # Safe distance
            self._clear_counter += 1
            if self._clear_counter >= self._clear_hold:
                self._state = "CLEAR"
            # else stay in current state (hold off reverting immediately)

        return self._state

    # ------------------------------------------------------------------
    def _compute_velocity(self) -> float:
        """Estimate approach velocity (px/s). Positive = getting closer."""
        if len(self._history) < 2:
            return 0.0
        dists = [d for d, _ in self._history]
        times = [t for _, t in self._history]
        dt = times[-1] - times[0]
        if dt < 1e-6:
            return 0.0
        # Positive velocity = distance decreasing = approaching
        return (dists[0] - dists[-1]) / dt

    @property
    def state(self) -> str:
        return self._state

    @property
    def velocity(self) -> float:
        return self._velocity

    def reset(self) -> None:
        """Reset state after fault recovery."""
        self._state = "CLEAR"
        self._history.clear()
        self._clear_counter = 0
        self._velocity = 0.0
