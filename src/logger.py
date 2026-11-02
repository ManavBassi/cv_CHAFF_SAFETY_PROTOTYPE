"""
Event and performance logger.
Writes structured JSON event log and CSV latency log.
No imports from other src modules — this is a leaf dependency.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class LatencyRecord:
    frame_id: int
    timestamp: float
    acquisition_ms: float = 0.0
    inference_ms: float = 0.0
    decision_ms: float = 0.0
    output_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class EventRecord:
    timestamp: float
    frame_id: int
    state: str
    hand_detected: bool
    confidence: float
    distance_px: float
    velocity_px_s: float
    latency_total_ms: float
    note: str = ""


class EventLogger:
    def __init__(self, config: dict):
        log_cfg = config.get("logging", {})
        self.log_dir = Path(log_cfg.get("log_dir", "logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Event JSONL
        event_file = self.log_dir / log_cfg.get("event_log_file", "events.jsonl")
        self._event_fh = open(event_file, "a", encoding="utf-8")

        # Latency CSV
        latency_file = self.log_dir / log_cfg.get("latency_log_file", "latency.csv")
        write_header = not latency_file.exists() or latency_file.stat().st_size == 0
        self._latency_fh = open(latency_file, "a", newline="", encoding="utf-8")
        self._latency_writer = csv.DictWriter(
            self._latency_fh,
            fieldnames=[
                "frame_id", "timestamp", "acquisition_ms", "inference_ms",
                "decision_ms", "output_ms", "total_ms"
            ],
        )
        if write_header:
            self._latency_writer.writeheader()

        level = log_cfg.get("log_level", "INFO")
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(message)s",
        )
        self.logger = logging.getLogger("chaff_safety")
        self._frame_count = 0

    # ------------------------------------------------------------------
    def log_event(self, record: EventRecord) -> None:
        self._event_fh.write(json.dumps(asdict(record)) + "\n")
        self._event_fh.flush()
        self.logger.info(
            "frame=%d state=%s hand=%s dist=%.1f vel=%.1f lat=%.1fms",
            record.frame_id, record.state, record.hand_detected,
            record.distance_px, record.velocity_px_s, record.latency_total_ms,
        )

    def log_latency(self, record: LatencyRecord) -> None:
        self._latency_writer.writerow(asdict(record))
        self._latency_fh.flush()

    def next_frame_id(self) -> int:
        self._frame_count += 1
        return self._frame_count

    def close(self) -> None:
        self._event_fh.close()
        self._latency_fh.close()
