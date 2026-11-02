"""Unit tests for DangerZone module."""
from __future__ import annotations

import numpy as np
import pytest

from src.danger_zone import DangerZone

# Square zone from (0.4, 0.25) to (1.0, 0.75) in normalised coords
# At 1000×800 that's pixels (400,200)–(1000,600)
CONFIG = {
    "danger_zone": {
        "polygon": [
            [0.40, 0.25],
            [1.00, 0.25],
            [1.00, 0.75],
            [0.40, 0.75],
        ],
        "warning_distance_px": 100,
        "action_distance_px": 30,
    }
}


@pytest.fixture
def zone():
    z = DangerZone(CONFIG)
    z.update_frame_size(1000, 800)
    return z


class TestDangerZone:
    def test_inside_zone_returns_negative_distance(self, zone):
        # Centre of zone: ~(700, 400)
        d = zone.distance_to_zone((700, 400))
        assert d < 0  # negative = inside

    def test_outside_zone_returns_positive_distance(self, zone):
        # Far left of frame
        d = zone.distance_to_zone((50, 400))
        assert d > 0

    def test_is_inside_true_for_centre(self, zone):
        assert zone.is_inside((700, 400)) is True

    def test_is_inside_false_for_far_left(self, zone):
        assert zone.is_inside((50, 400)) is False

    def test_warning_zone_triggered_within_warning_dist(self, zone):
        # ~350px from zone left edge (400) → distance ≈ 50 px (within 100)
        d = zone.distance_to_zone((350, 400))
        assert zone.in_warning_zone(d)

    def test_warning_zone_not_triggered_far_away(self, zone):
        d = zone.distance_to_zone((0, 400))      # 400 px from zone edge
        assert not zone.in_warning_zone(d)       # 400 > warning_distance 100

    def test_action_zone_triggered_very_close(self, zone):
        d = zone.distance_to_zone((385, 400))    # 15 px from zone edge → < 30
        assert zone.in_action_zone(d)

    def test_draw_does_not_raise(self, zone):
        frame = np.zeros((800, 1000, 3), dtype=np.uint8)
        result = zone.draw(frame, state="CLEAR")
        assert result.shape == (800, 1000, 3)

    def test_update_frame_size_recomputes_polygon(self, zone):
        zone.update_frame_size(640, 480)
        d = zone.distance_to_zone((320, 240))  # centre at new size
        # Should still work without exception
        assert isinstance(d, float)
