"""Unit tests for HandDetector module.
Uses synthetic blank frames — MediaPipe will return no detections,
so these tests verify the uncertainty/fault tracking logic.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.detector import HandDetector

CONFIG = {
    "detection": {
        "mediapipe": {
            "max_num_hands": 2,
            "min_detection_confidence": 0.6,
            "min_tracking_confidence": 0.5,
            "model_complexity": 0,
        },
        "missed_frames_warning": 3,
        "missed_frames_fault": 8,
    }
}


@pytest.fixture(scope="module")
def detector():
    d = HandDetector(CONFIG)
    yield d
    d.close()


def blank_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestHandDetector:
    def test_no_detection_on_blank_frame(self, detector):
        detector.reset()
        results = detector.detect(blank_frame())
        assert results == []

    def test_missed_count_increments(self, detector):
        detector.reset()
        for i in range(3):
            detector.detect(blank_frame())
        assert detector.missed_count == 3

    def test_uncertain_after_warning_threshold(self, detector):
        detector.reset()
        for _ in range(3):  # missed_frames_warning=3
            detector.detect(blank_frame())
        assert detector.uncertain is True

    def test_not_uncertain_below_threshold(self, detector):
        detector.reset()
        for _ in range(2):  # below threshold of 3
            detector.detect(blank_frame())
        assert detector.uncertain is False

    def test_fault_after_fault_threshold(self, detector):
        detector.reset()
        for _ in range(8):  # missed_frames_fault=8
            detector.detect(blank_frame())
        assert detector.fault is True

    def test_reset_clears_missed_count(self, detector):
        detector.reset()
        for _ in range(5):
            detector.detect(blank_frame())
        detector.reset()
        assert detector.missed_count == 0
        assert detector.uncertain is False
        assert detector.fault is False

    def test_detect_returns_list(self, detector):
        detector.reset()
        result = detector.detect(blank_frame())
        assert isinstance(result, list)
