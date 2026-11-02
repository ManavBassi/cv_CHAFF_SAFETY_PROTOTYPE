#!/usr/bin/env python3
"""
CV Chaff Safety Prototype — Main entry point.
Run: python run.py [--config config.yaml] [--no-display]

ALL MACHINE-CONTROL OUTPUTS ARE SIMULATED.
No physical actuation is enabled by default.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Ensure src/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.camera import CameraInput, CameraStatus
from src.danger_zone import DangerZone
from src.detector import HandDetector
from src.display import Display
from src.logger import EventLogger, EventRecord, LatencyRecord, load_config
from src.output import SimulatedOutput
from src.predictor import ApproachPredictor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV Chaff Safety Prototype")
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("--no-display", action="store_true", help="Headless mode (no OpenCV window)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    camera = CameraInput(config)
    detector = HandDetector(config)
    zone = DangerZone(config)
    predictor = ApproachPredictor(config)
    sim_output = SimulatedOutput(config)
    event_log = EventLogger(config)
    display = Display() if not args.no_display else None

    print("CV Chaff Safety Prototype started.")
    print("Press 'q' or ESC to quit.\n")

    try:
        while True:
            t0 = time.monotonic()

            # ── 1. Camera acquisition ──────────────────────────────────
            frame, ts_acq, cam_status = camera.read()
            t_acq = time.monotonic()

            camera_fault = cam_status != CameraStatus.OK
            if camera_fault:
                print(f"[CAMERA] Status: {cam_status}", flush=True)
                # Show fault frame if display is active
                if display and frame is not None:
                    h, w = frame.shape[:2]
                    zone.update_frame_size(w, h)
                    zone.draw(frame, state="FAULT")
                    if not display.show(frame):
                        break
                elif display:
                    blank = __import__("numpy").zeros((720, 1280, 3), dtype="uint8")
                    cv2.putText(blank, f"CAMERA {cam_status}", (200, 360),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (180, 0, 180), 3)
                    if not display.show(blank):
                        break
                predictor.update(9999, ts_acq, False, True, False)
                sim_output.process("FAULT", 9999, 0)
                time.sleep(0.05)
                continue

            h, w = frame.shape[:2]
            zone.update_frame_size(w, h)

            # ── 2. Hand detection ──────────────────────────────────────
            hands = detector.detect(frame)
            t_inf = time.monotonic()

            detection_fault = detector.fault

            # ── 3. Danger-zone and distance evaluation ─────────────────
            distance_px = 9999.0
            hand_detected = len(hands) > 0

            if hand_detected:
                # Use the closest point from any detected hand
                # (check wrist and index fingertip — whichever is nearer)
                for hand in hands:
                    for pt in (hand.wrist_px, hand.index_tip_px):
                        d = zone.distance_to_zone(pt)
                        if d < distance_px:
                            distance_px = d

            t_dec_start = time.monotonic()
            state = predictor.update(
                distance_px=distance_px,
                timestamp=ts_acq,
                hand_detected=hand_detected,
                camera_fault=camera_fault,
                detection_fault=detection_fault,
            )
            velocity = predictor.velocity
            t_dec = time.monotonic()

            # ── 4. Simulated output ────────────────────────────────────
            sim_output.process(state, distance_px, velocity)
            t_out = time.monotonic()

            # ── 5. Logging ─────────────────────────────────────────────
            frame_id = event_log.next_frame_id()
            acq_ms = (t_acq - t0) * 1000
            inf_ms = (t_inf - t_acq) * 1000
            dec_ms = (t_dec - t_dec_start) * 1000
            out_ms = (t_out - t_dec) * 1000
            total_ms = (t_out - t0) * 1000

            event_log.log_latency(LatencyRecord(
                frame_id=frame_id,
                timestamp=ts_acq,
                acquisition_ms=round(acq_ms, 2),
                inference_ms=round(inf_ms, 2),
                decision_ms=round(dec_ms, 2),
                output_ms=round(out_ms, 2),
                total_ms=round(total_ms, 2),
            ))

            if state != "CLEAR" or frame_id % 30 == 0:  # log every 30 frames in CLEAR
                conf = max((h.confidence for h in hands), default=0.0)
                event_log.log_event(EventRecord(
                    timestamp=ts_acq,
                    frame_id=frame_id,
                    state=state,
                    hand_detected=hand_detected,
                    confidence=round(conf, 3),
                    distance_px=round(distance_px, 1),
                    velocity_px_s=round(velocity, 1),
                    latency_total_ms=round(total_ms, 2),
                ))

            # ── 6. Display ─────────────────────────────────────────────
            if display:
                zone.draw(frame, state=state)
                frame = display.render(
                    frame=frame,
                    hands=hands,
                    state=state,
                    distance_px=distance_px,
                    velocity_px_s=velocity,
                    latency_ms=total_ms,
                    missed_count=detector.missed_count,
                )
                if not display.show(frame):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        camera.release()
        detector.close()
        event_log.close()
        if display:
            display.close()
        print("Logs saved to:", config["logging"]["log_dir"])


if __name__ == "__main__":
    main()
