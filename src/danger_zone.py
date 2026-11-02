"""
Danger-zone definition and distance calculation.
Loads a normalised polygon from config and converts to pixel coords.
Provides point-to-polygon distance, inside test, and frame annotation.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import cv2
import numpy as np


class DangerZone:
    def __init__(self, config: dict):
        dz_cfg = config["danger_zone"]
        poly_norm = dz_cfg["polygon"]

        self._warn_dist: int = dz_cfg.get("warning_distance_px", 120)
        self._action_dist: int = dz_cfg.get("action_distance_px", 40)

        # Store normalised coords; pixel coords computed per-frame
        self._poly_norm: List[Tuple[float, float]] = [(p[0], p[1]) for p in poly_norm]
        self._frame_w: int = 1
        self._frame_h: int = 1
        self._poly_px: np.ndarray = np.zeros((len(self._poly_norm), 2), dtype=np.float32)

    # ------------------------------------------------------------------
    def update_frame_size(self, w: int, h: int) -> None:
        if w != self._frame_w or h != self._frame_h:
            self._frame_w = w
            self._frame_h = h
            self._poly_px = np.array(
                [(x * w, y * h) for x, y in self._poly_norm], dtype=np.float32
            )

    # ------------------------------------------------------------------
    def distance_to_zone(self, point_px: Tuple[int, int]) -> float:
        """
        Signed distance from point to the polygon boundary (pixels).
        Negative = inside the polygon.
        """
        pt = np.array([[float(point_px[0]), float(point_px[1])]], dtype=np.float32)
        # pointPolygonTest: positive = outside, negative = inside, 0 = on boundary
        dist = cv2.pointPolygonTest(self._poly_px, (pt[0][0], pt[0][1]), measureDist=True)
        return -float(dist)  # invert: negative means safe (outside), positive means getting closer

    def is_inside(self, point_px: Tuple[int, int]) -> bool:
        d = cv2.pointPolygonTest(
            self._poly_px, (float(point_px[0]), float(point_px[1])), measureDist=False
        )
        return d >= 0  # 0 = on boundary, positive = inside

    def in_warning_zone(self, dist: float) -> bool:
        """dist is the value from distance_to_zone (negative = inside)."""
        return dist <= self._warn_dist  # approaching or inside

    def in_action_zone(self, dist: float) -> bool:
        return dist <= self._action_dist

    @property
    def warning_distance(self) -> int:
        return self._warn_dist

    @property
    def action_distance(self) -> int:
        return self._action_dist

    # ------------------------------------------------------------------
    def draw(
        self,
        frame: np.ndarray,
        state: str = "CLEAR",
        warn_ring: bool = False,
    ) -> np.ndarray:
        """Draw the danger zone and optional warning ring onto the frame."""
        if self._poly_px.size == 0:
            return frame

        pts = self._poly_px.astype(np.int32).reshape((-1, 1, 2))

        # Zone fill colour based on state
        color_map = {
            "CLEAR": (0, 200, 0),
            "WARNING": (0, 215, 255),
            "PROTECTIVE_ACTION": (0, 0, 220),
            "FAULT": (180, 0, 180),
        }
        zone_color = color_map.get(state, (0, 200, 0))

        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], zone_color)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
        cv2.polylines(frame, [pts], isClosed=True, color=zone_color, thickness=2)

        # Label
        cx = int(self._poly_px[:, 0].mean())
        cy = int(self._poly_px[:, 1].mean())
        cv2.putText(
            frame, "DANGER ZONE", (cx - 50, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, zone_color, 1, cv2.LINE_AA,
        )
        return frame
