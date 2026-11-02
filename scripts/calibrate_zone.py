"""
Interactive danger-zone calibration tool.
Click 4+ points on the live camera frame to define the zone polygon.
Press 's' to save to config.yaml, 'r' to reset, 'q' to quit.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import yaml

from src.camera import CameraInput, CameraStatus
from src.logger import load_config

_POINTS = []
_W, _H = 1, 1


def _click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        _POINTS.append((x, y))
        print(f"  Point added: ({x}, {y}) → normalised ({x/_W:.3f}, {y/_H:.3f})")


def main():
    config = load_config()
    camera = CameraInput(config)

    cv2.namedWindow("Calibrate Zone", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Calibrate Zone", _click)

    print("Click points on the frame to define the danger zone polygon.")
    print("  's' → save to config.yaml")
    print("  'r' → reset points")
    print("  'q' → quit without saving\n")

    global _W, _H

    while True:
        frame, _, status = camera.read()
        if status != CameraStatus.OK or frame is None:
            print("Camera not available.")
            break

        _H, _W = frame.shape[:2]

        # Draw existing points and lines
        for i, pt in enumerate(_POINTS):
            cv2.circle(frame, pt, 6, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (pt[0] + 8, pt[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if len(_POINTS) > 1:
            for i in range(len(_POINTS) - 1):
                cv2.line(frame, _POINTS[i], _POINTS[i + 1], (0, 255, 0), 2)
            if len(_POINTS) > 2:
                cv2.line(frame, _POINTS[-1], _POINTS[0], (0, 255, 0), 1)

        cv2.putText(frame, f"Points: {len(_POINTS)}  [s]=save [r]=reset [q]=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Calibrate Zone", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break
        elif key == ord("r"):
            _POINTS.clear()
            print("Points reset.")
        elif key == ord("s"):
            if len(_POINTS) < 3:
                print("Need at least 3 points to define a polygon.")
            else:
                norm = [[round(x / _W, 4), round(y / _H, 4)] for x, y in _POINTS]
                config["danger_zone"]["polygon"] = norm
                with open("config.yaml", "w") as f:
                    yaml.dump(config, f, default_flow_style=False)
                print(f"Saved {len(norm)} points to config.yaml:")
                for p in norm:
                    print(f"  {p}")
                break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
