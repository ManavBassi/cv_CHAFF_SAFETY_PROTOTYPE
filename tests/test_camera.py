"""Unit tests for CameraInput module."""
from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.camera import CameraInput, CameraStatus

BASE_CONFIG = {
    "camera": {
        "device_index": 0,
        "width": 640,
        "height": 480,
        "fps": 30,
        "freeze_timeout_s": 0.5,
    }
}


def _make_frame(val: int = 42) -> np.ndarray:
    frame = np.full((480, 640, 3), val, dtype=np.uint8)
    return frame


class TestCameraInput:
    def _camera_with_mock(self, frames):
        """
        Create a CameraInput backed by a mock VideoCapture.
        `frames` is a list of (ret, frame) tuples returned per .read() call.
        """
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = frames

        with patch("src.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = CameraInput(BASE_CONFIG)
        cam._cap = mock_cap  # replace internal cap with our mock
        return cam

    def test_ok_frame_returned(self):
        frame = _make_frame(100)
        cam = self._camera_with_mock([(True, frame)] * 5)
        f, ts, status = cam.read()
        assert status == CameraStatus.OK
        assert f is not None

    def test_disconnected_on_read_fail(self):
        cam = self._camera_with_mock([(False, None)] * 3)
        _, _, status = cam.read()
        assert status == CameraStatus.DISCONNECTED

    def test_freeze_detected_after_timeout(self):
        """Same frame read repeatedly should trigger FROZEN after freeze_timeout_s."""
        frame = _make_frame(77)
        # First call: establishes last_frame; subsequent: identical → freeze count builds
        cam = self._camera_with_mock([(True, frame.copy())] * 20)
        cam._last_new_frame_time = time.monotonic() - 1.0  # pretend it's been 1 s

        f, ts, status = cam.read()
        assert status == CameraStatus.FROZEN

    def test_no_freeze_when_frames_change(self):
        """Different frames must not trigger FROZEN."""
        frames = [(True, _make_frame(i)) for i in range(5, 55, 1)]
        cam = self._camera_with_mock(frames)
        cam._last_new_frame_time = time.monotonic() - 1.0

        for _ in range(10):
            f, ts, status = cam.read()
        assert status == CameraStatus.OK

    def test_is_open_true_when_cap_open(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("src.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = CameraInput(BASE_CONFIG)
        assert cam.is_open()

    def test_release_closes_cap(self):
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        with patch("src.camera.cv2.VideoCapture", return_value=mock_cap):
            cam = CameraInput(BASE_CONFIG)
        cam.release()
        mock_cap.release.assert_called_once()
