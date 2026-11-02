"""
Per-stage latency benchmark.
Runs N frames (live camera or synthetic) and reports avg / p95 / worst latency
for each pipeline stage and for the total end-to-end time.
Results also appended to logs/latency_report.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, quantiles

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detector import HandDetector
from src.logger import load_config


def make_synthetic_frame(w: int = 1280, h: int = 720) -> np.ndarray:
    """Create a synthetic BGR test frame with a hand-like blob."""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # Plain blue background
    frame[:] = (30, 30, 60)
    # Draw a bright region that might look hand-like
    cx, cy = w // 4, h // 2
    frame[cy - 80: cy + 80, cx - 60: cx + 60] = (180, 140, 110)
    return frame


def stats(data: list) -> dict:
    if not data:
        return {"avg": 0, "p95": 0, "worst": 0, "count": 0}
    qs = quantiles(data, n=20) if len(data) >= 20 else data
    p95 = qs[18] if len(qs) > 18 else max(data)
    return {
        "avg": round(mean(data), 2),
        "p95": round(p95, 2),
        "worst": round(max(data), 2),
        "count": len(data),
    }


def main():
    ap = argparse.ArgumentParser(description="Latency benchmark")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--synthetic", action="store_true",
                    help="Use synthetic frames instead of live camera")
    args = ap.parse_args()

    config = load_config(args.config)
    detector = HandDetector(config)

    acq_times, inf_times, total_times = [], [], []

    if not args.synthetic:
        import cv2
        cap = cv2.VideoCapture(config["camera"]["device_index"])
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = None

    print(f"Benchmarking {args.frames} frames "
          f"({'synthetic' if args.synthetic else 'live camera'}) …")

    for i in range(args.frames):
        t0 = time.monotonic()

        if cap:
            ret, frame = cap.read()
            if not ret:
                print("Camera read failed at frame", i)
                break
        else:
            frame = make_synthetic_frame()

        t_acq = time.monotonic()
        detector.detect(frame)
        t_inf = time.monotonic()

        acq_times.append((t_acq - t0) * 1000)
        inf_times.append((t_inf - t_acq) * 1000)
        total_times.append((t_inf - t0) * 1000)

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{args.frames} frames processed …")

    if cap:
        cap.release()
    detector.close()

    report = {
        "benchmark_type": "synthetic" if args.synthetic else "live_camera",
        "frames": args.frames,
        "acquisition_ms": stats(acq_times),
        "inference_ms": stats(inf_times),
        "total_pipeline_ms": stats(total_times),
    }

    print("\n── Latency Report ──────────────────────────────────────────")
    for stage, s in report.items():
        if isinstance(s, dict):
            print(f"  {stage:25s}  avg={s['avg']:6.1f}ms  "
                  f"p95={s['p95']:6.1f}ms  worst={s['worst']:6.1f}ms")
    print("─" * 65)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    report_file = log_dir / "latency_report.json"

    existing = []
    if report_file.exists():
        try:
            existing = json.loads(report_file.read_text())
            if not isinstance(existing, list):
                existing = [existing]
        except json.JSONDecodeError:
            existing = []

    existing.append(report)
    report_file.write_text(json.dumps(existing, indent=2))
    print(f"Report saved to: {report_file}")


if __name__ == "__main__":
    main()
