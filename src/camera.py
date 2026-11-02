"""
Camera input module.
Handles USB camera open, frame acquisition, freeze detection, and disconnection.
Returns (frame, timestamp, status) where status ∈ {"OK", "FROZEN", "DISCONNECTED"}.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class CameraStatus(str, Enum):
    OK = "OK"
    FROZEN = "FROZEN"
    DISCONNECTED = "DISCONNECTED"


class CameraInput:
    def __init__(self, config: dict):
        cam_cfg = config["camera"]
        self._device_index: int = cam_cfg.get("device_index", 0)
        self._width: int = cam_cfg.get("width", 1280)
        self._height: int = cam_cfg.get("height", 720)
        self._fps: int = cam_cfg.get("fps", 30)
        self._freeze_timeout: float = cam_cfg.get("freeze_timeout_s", 2.0)

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame: Optional[np.ndarray] = None
        self._last_new_frame_time: float = time.monotonic()
        self._open_camera()

    # ------------------------------------------------------------------
    def _open_camera(self) -> None:
        if self._cap is not None:
            self._cap.release()
        self._cap = cv2.VideoCapture(self._device_index)
        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self._cap.set(cv2.CAP_PROP_FPS, self._fps)
            # Minimise internal buffer to reduce acquisition latency
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    # ------------------------------------------------------------------
    def read(self) -> Tuple[Optional[np.ndarray], float, CameraStatus]:
        """
        Returns (frame, timestamp, status).
        frame is None when status != OK.
        """
        ts = time.monotonic()

        if not self.is_open():
            return None, ts, CameraStatus.DISCONNECTED

        ret, frame = self._cap.read()

        if not ret or frame is None:
            return None, ts, CameraStatus.DISCONNECTED

        # Freeze detection: compare with previous frame
        if self._last_frame is not None:
            diff = cv2.absdiff(frame, self._last_frame)
            if diff.max() == 0:  # pixel-identical
                elapsed = ts - self._last_new_frame_time
                if elapsed > self._freeze_timeout:
                    return frame, ts, CameraStatus.FROZEN
            else:
                self._last_new_frame_time = ts

        self._last_frame = frame.copy()
        return frame, ts, CameraStatus.OK

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
