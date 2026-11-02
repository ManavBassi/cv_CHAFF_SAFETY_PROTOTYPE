"""
Integration test: simulates a hand approaching the danger zone over 40+ frames.
Verifies the entire pipeline: detector uncertainty → predictor → state transitions.
Does NOT require a real camera (uses synthetic distances directly).
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest

from src.danger_zone import DangerZone
from src.predictor import ApproachPredictor

FULL_CONFIG = {
    "camera": {"device_index": 0, "width": 640, "height": 480,
                "fps": 30, "freeze_timeout_s": 2.0},
    "detection": {
        "mediapipe": {"max_num_hands": 2, "min_detection_confidence": 0.6,
                      "min_tracking_confidence": 0.5, "model_complexity": 0},
        "missed_frames_warning": 5,
        "missed_frames_fault": 15,
    },
    "danger_zone": {
        "polygon": [[0.60, 0.20], [1.00, 0.20], [1.00, 0.80], [0.60, 0.80]],
        "warning_distance_px": 120,
        "action_distance_px": 40,
    },
    "predictor": {
        "velocity_window": 6,
        "approach_speed_threshold_px_s": 80.0,
        "clear_hold_frames": 10,
    },
    "output": {"simulated_mode": True, "audio_beep": False},
}


class TestIntegration:
    def test_clear_to_protective_action_transition(self):
        """Hand starting at 400px and moving to <40px triggers PROTECTIVE_ACTION."""
        pred = ApproachPredictor(FULL_CONFIG)
        zone = DangerZone(FULL_CONFIG)
        zone.update_frame_size(1000, 800)

        states_seen = set()
        dist = 400.0
        t = 0.0

        for _ in range(50):
            state = pred.update(dist, t, hand_detected=True,
                                camera_fault=False, detection_fault=False)
            states_seen.add(state)
            dist = max(0.0, dist - 9.0)   # approach at ~270px/s
            t += 0.033

        assert "WARNING" in states_seen, "WARNING state was never entered"
        assert "PROTECTIVE_ACTION" in states_seen, "PROTECTIVE_ACTION was never triggered"

    def test_fault_on_camera_failure(self):
        pred = ApproachPredictor(FULL_CONFIG)
        state = pred.update(500, 0.0, hand_detected=False,
                             camera_fault=True, detection_fault=False)
        assert state == "FAULT"

    def test_stays_warning_without_clear_hold(self):
        pred = ApproachPredictor(FULL_CONFIG)
        # Enter WARNING
        pred.update(80, 0.0, True, False, False)
        assert pred.state == "WARNING"

        # Single safe frame → should NOT revert to CLEAR
        pred.update(500, 0.033, True, False, False)
        assert pred.state == "WARNING"

    def test_protective_action_from_zero_distance(self):
        pred = ApproachPredictor(FULL_CONFIG)
        state = pred.update(0.0, 0.0, True, False, False)
        assert state == "PROTECTIVE_ACTION"

    def test_danger_zone_distance_triggers_correct_states(self):
        """
        Use DangerZone to calculate distances and feed into predictor.
        Points chosen so we cross the zone boundary.
        """
        pred = ApproachPredictor(FULL_CONFIG)
        zone = DangerZone(FULL_CONFIG)
        zone.update_frame_size(1000, 800)

        # Point far outside zone → CLEAR
        d = zone.distance_to_zone((50, 400))   # ~350px outside
        state = pred.update(d, 0.0, True, False, False)
        # Might stay CLEAR (dist > warning threshold 120)
        assert state == "CLEAR"

        # Point very close to zone edge (warning band)
        pred2 = ApproachPredictor(FULL_CONFIG)
        d2 = zone.distance_to_zone((490, 400))  # ~110px outside (within warning 120)
        state2 = pred2.update(d2, 0.0, True, False, False)
        assert state2 == "WARNING"

        # Point inside zone
        pred3 = ApproachPredictor(FULL_CONFIG)
        d3 = zone.distance_to_zone((700, 400))  # inside → negative
        state3 = pred3.update(d3, 0.0, True, False, False)
        assert state3 == "PROTECTIVE_ACTION"
