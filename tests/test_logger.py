"""Unit tests for EventLogger."""
from __future__ import annotations

import csv
import json
import tempfile
import time
from pathlib import Path

import pytest

from src.logger import EventLogger, EventRecord, LatencyRecord


@pytest.fixture
def tmp_config(tmp_path):
    return {
        "logging": {
            "log_dir": str(tmp_path / "logs"),
            "event_log_file": "events.jsonl",
            "latency_log_file": "latency.csv",
            "log_level": "WARNING",
        }
    }


class TestEventLogger:
    def test_log_dir_created(self, tmp_config):
        log = EventLogger(tmp_config)
        log.close()
        assert Path(tmp_config["logging"]["log_dir"]).is_dir()

    def test_event_written_to_jsonl(self, tmp_config):
        log = EventLogger(tmp_config)
        rec = EventRecord(
            timestamp=time.time(),
            frame_id=1,
            state="WARNING",
            hand_detected=True,
            confidence=0.85,
            distance_px=75.0,
            velocity_px_s=120.0,
            latency_total_ms=22.5,
        )
        log.log_event(rec)
        log.close()

        event_file = Path(tmp_config["logging"]["log_dir"]) / "events.jsonl"
        lines = event_file.read_text().strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["state"] == "WARNING"
        assert data["confidence"] == 0.85

    def test_latency_written_to_csv(self, tmp_config):
        log = EventLogger(tmp_config)
        lat = LatencyRecord(
            frame_id=1,
            timestamp=time.time(),
            acquisition_ms=2.5,
            inference_ms=18.0,
            decision_ms=0.3,
            output_ms=0.1,
            total_ms=20.9,
        )
        log.log_latency(lat)
        log.close()

        lat_file = Path(tmp_config["logging"]["log_dir"]) / "latency.csv"
        rows = list(csv.DictReader(lat_file.open()))
        assert len(rows) == 1
        assert float(rows[0]["inference_ms"]) == pytest.approx(18.0)

    def test_next_frame_id_increments(self, tmp_config):
        log = EventLogger(tmp_config)
        ids = [log.next_frame_id() for _ in range(5)]
        log.close()
        assert ids == [1, 2, 3, 4, 5]

    def test_multiple_events_appended(self, tmp_config):
        log = EventLogger(tmp_config)
        for i in range(3):
            log.log_event(EventRecord(
                timestamp=time.time(), frame_id=i, state="CLEAR",
                hand_detected=False, confidence=0.0,
                distance_px=500.0, velocity_px_s=0.0, latency_total_ms=10.0,
            ))
        log.close()

        event_file = Path(tmp_config["logging"]["log_dir"]) / "events.jsonl"
        lines = [l for l in event_file.read_text().strip().split("\n") if l]
        assert len(lines) == 3
