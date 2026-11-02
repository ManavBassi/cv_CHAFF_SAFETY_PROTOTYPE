"""Unit tests for ApproachPredictor state machine."""
from __future__ import annotations

import time

import pytest

from src.predictor import ApproachPredictor

CONFIG = {
    "predictor": {
        "velocity_window": 4,
        "approach_speed_threshold_px_s": 80.0,
        "clear_hold_frames": 5,
    },
    "danger_zone": {
        "warning_distance_px": 100,
        "action_distance_px": 30,
    },
}


@pytest.fixture
def pred():
    return ApproachPredictor(CONFIG)


class TestApproachPredictor:
    def _tick(self, pred, dist, t_offset, hand=True, cam_fault=False, det_fault=False):
        return pred.update(dist, t_offset, hand, cam_fault, det_fault)

    def test_initial_state_is_clear(self, pred):
        assert pred.state == "CLEAR"

    def test_camera_fault_sets_fault(self, pred):
        state = self._tick(pred, 500, 0.0, cam_fault=True)
        assert state == "FAULT"

    def test_detection_fault_sets_fault(self, pred):
        state = self._tick(pred, 500, 0.0, det_fault=True)
        assert state == "FAULT"

    def test_fault_is_sticky(self, pred):
        self._tick(pred, 500, 0.0, cam_fault=True)
        # Even with no fault next frame, state stays FAULT (no auto-recovery)
        state = self._tick(pred, 500, 0.1, hand=True)
        assert state == "FAULT"

    def test_reset_clears_fault(self, pred):
        self._tick(pred, 500, 0.0, cam_fault=True)
        pred.reset()
        assert pred.state == "CLEAR"

    def test_action_zone_triggers_protective_action(self, pred):
        # Distance inside action threshold (30 px)
        state = self._tick(pred, 10, 0.0)
        assert state == "PROTECTIVE_ACTION"

    def test_warning_zone_triggers_warning(self, pred):
        # Distance between action (30) and warning (100) thresholds
        state = self._tick(pred, 70, 0.0)
        assert state == "WARNING"

    def test_safe_distance_stays_clear(self, pred):
        state = self._tick(pred, 500, 0.0)
        assert state == "CLEAR"

    def test_state_does_not_revert_to_clear_immediately(self, pred):
        """After WARNING, single safe frame should not revert to CLEAR."""
        self._tick(pred, 70, 0.0)   # → WARNING
        state = self._tick(pred, 500, 0.1)  # safe, but clear_hold not reached
        assert state == "WARNING"

    def test_state_reverts_to_clear_after_hold(self, pred):
        """After clear_hold_frames=5 consecutive safe frames, should go CLEAR."""
        self._tick(pred, 70, 0.0)   # → WARNING
        t = 0.1
        for _ in range(6):          # 6 safe frames > clear_hold=5
            state = self._tick(pred, 500, t)
            t += 0.033
        assert state == "CLEAR"

    def test_velocity_estimation(self, pred):
        """Approaching hand (decreasing distance) → positive velocity."""
        dist = 300.0
        t = 0.0
        for _ in range(6):
            pred.update(dist, t, True, False, False)
            dist -= 10  # approaching
            t += 0.033
        assert pred.velocity > 0

    def test_no_velocity_with_one_sample(self, pred):
        pred.update(200, 0.0, True, False, False)
        assert pred.velocity == 0.0
