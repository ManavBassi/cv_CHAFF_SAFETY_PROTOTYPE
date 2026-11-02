"""
Hand detection and tracking module.
Primary: MediaPipe Hands (Apache 2.0).
Tracks consecutive missed detections and sets uncertain=True after threshold.
Conservative policy: is_safe() returns False when uncertain.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HandResult:
    """One detected hand."""
    landmarks_norm: List[Tuple[float, float, float]]  # 21 (x, y, z) normalised
    wrist_px: Tuple[int, int]                          # pixel position of wrist
    index_tip_px: Tuple[int, int]                     # pixel position of index fingertip
    bounding_box_px: Tuple[int, int, int, int]        # (x, y, w, h)
    confidence: float
    handedness: str                                    # "Left" or "Right"


class HandDetector:
    def __init__(self, config: dict):
        det_cfg = config["detection"]
        mp_cfg = det_cfg.get("mediapipe", {})

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=mp_cfg.get("max_num_hands", 2),
            min_detection_confidence=mp_cfg.get("min_detection_confidence", 0.6),
            min_tracking_confidence=mp_cfg.get("min_tracking_confidence", 0.5),
            model_complexity=mp_cfg.get("model_complexity", 0),
        )
        self._missed_warning = det_cfg.get("missed_frames_warning", 5)
        self._missed_fault = det_cfg.get("missed_frames_fault", 15)
        self._missed_count: int = 0
        self._uncertain: bool = False

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[HandResult]:
        """
        Run inference on a BGR frame.
        Updates internal missed-detection counter.
        Returns list of HandResult (may be empty).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._missed_count += 1
            self._uncertain = self._missed_count >= self._missed_warning
            return []

        self._missed_count = 0
        self._uncertain = False

        hands: List[HandResult] = []
        for lm_list, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            landmarks_norm = [(lm.x, lm.y, lm.z) for lm in lm_list.landmark]

            # Pixel coords for wrist (landmark 0) and index tip (landmark 8)
            wrist = lm_list.landmark[0]
            tip = lm_list.landmark[8]
            wrist_px = (int(wrist.x * w), int(wrist.y * h))
            tip_px = (int(tip.x * w), int(tip.y * h))

            # Bounding box from all landmarks
            xs = [lm.x * w for lm in lm_list.landmark]
            ys = [lm.y * h for lm in lm_list.landmark]
            x0, y0 = int(min(xs)), int(min(ys))
            x1, y1 = int(max(xs)), int(max(ys))
            bbox = (x0, y0, x1 - x0, y1 - y0)

            conf = handedness.classification[0].score
            side = handedness.classification[0].label

            hands.append(HandResult(
                landmarks_norm=landmarks_norm,
                wrist_px=wrist_px,
                index_tip_px=tip_px,
                bounding_box_px=bbox,
                confidence=conf,
                handedness=side,
            ))
        return hands

    @property
    def uncertain(self) -> bool:
        return self._uncertain

    @property
    def fault(self) -> bool:
        """True when missed detections exceed fault threshold."""
        return self._missed_count >= self._missed_fault

    @property
    def missed_count(self) -> int:
        return self._missed_count

    def reset(self) -> None:
        self._missed_count = 0
        self._uncertain = False

    def close(self) -> None:
        self._hands.close()
