"""
Annotated display module.
Draws: danger zone, hand landmarks, bounding box, state badge,
distance, velocity, FPS, and latency onto the video frame.
"""
from __future__ import annotations

import time
from collections import deque
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.detector import HandResult


# State → (BGR colour, label text)
STATE_STYLE = {
    "CLEAR":             ((0, 200, 0),   "CLEAR"),
    "WARNING":           ((0, 215, 255), "WARNING"),
    "PROTECTIVE_ACTION": ((0, 0, 220),   "STOP"),
    "FAULT":             ((180, 0, 180), "FAULT"),
}


class Display:
    def __init__(self):
        self._fps_times: deque = deque(maxlen=30)
        self._window_name = "CV Chaff Safety Prototype"
        cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._window_name, 1280, 720)

    # ------------------------------------------------------------------
    def render(
        self,
        frame: np.ndarray,
        hands: List[HandResult],
        state: str,
        distance_px: float,
        velocity_px_s: float,
        latency_ms: float,
        missed_count: int,
    ) -> np.ndarray:
        # FPS
        now = time.monotonic()
        self._fps_times.append(now)
        fps = 0.0
        if len(self._fps_times) > 1:
            fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0])

        color, label = STATE_STYLE.get(state, ((128, 128, 128), state))

        # Draw hands
        mp_draw = __import__("mediapipe").solutions.drawing_utils
        mp_styles = __import__("mediapipe").solutions.drawing_styles
        mp_hands_module = __import__("mediapipe").solutions.hands

        for hand in hands:
            # Bounding box
            x, y, w, h = hand.bounding_box_px
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Wrist dot
            cv2.circle(frame, hand.wrist_px, 6, color, -1)

            # Index fingertip dot
            cv2.circle(frame, hand.index_tip_px, 6, (255, 255, 0), -1)

            # Confidence label
            cv2.putText(
                frame,
                f"{hand.handedness} {hand.confidence:.2f}",
                (x, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA,
            )

        # State badge (top-left)
        badge_h = 36
        cv2.rectangle(frame, (0, 0), (320, badge_h), color, -1)
        cv2.putText(
            frame, f"STATE: {label}", (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA,
        )

        # HUD overlay (top-right)
        h_frame, w_frame = frame.shape[:2]
        hud_lines = [
            f"FPS: {fps:5.1f}",
            f"Latency: {latency_ms:5.1f} ms",
            f"Dist: {distance_px:5.0f} px",
            f"Vel:  {velocity_px_s:+6.0f} px/s",
            f"Missed: {missed_count}",
        ]
        y_start = 20
        for line in hud_lines:
            (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            x_pos = w_frame - tw - 12
            cv2.putText(
                frame, line, (x_pos, y_start),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA,
            )
            y_start += 22

        # Missed-detection warning strip
        if missed_count > 0:
            warn_text = f"⚠ No hand detected ({missed_count} frames)"
            cv2.putText(
                frame, warn_text, (8, h_frame - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1, cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    def show(self, frame: np.ndarray) -> bool:
        """Display frame. Returns False if user presses 'q' or closes window."""
        cv2.imshow(self._window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key != ord("q") and key != 27  # 27 = ESC

    def close(self) -> None:
        cv2.destroyAllWindows()
