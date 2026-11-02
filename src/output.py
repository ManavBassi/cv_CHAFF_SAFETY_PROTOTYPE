"""
Simulated output module.
ALL MACHINE-CONTROL OUTPUTS ARE SIMULATED.
No GPIO, serial port, or real machine interface is enabled.
"""
from __future__ import annotations

import sys
import time
from typing import Optional


class SimulatedOutput:
    def __init__(self, config: dict):
        out_cfg = config.get("output", {})
        self._simulated: bool = out_cfg.get("simulated_mode", True)
        self._audio: bool = out_cfg.get("audio_beep", True)
        self._last_state: str = "CLEAR"
        self._last_action_time: float = 0.0

        if not self._simulated:
            raise RuntimeError(
                "simulated_mode is False. Real machine actuation is DISABLED "
                "in this prototype. Set simulated_mode=true in config.yaml."
            )

    # ------------------------------------------------------------------
    def process(self, state: str, distance_px: float, velocity_px_s: float) -> None:
        """
        Emit simulated protective-action signal based on state.
        Returns immediately — no blocking I/O.
        """
        if state == self._last_state:
            # Only emit on state change (avoid log spam), except for PROTECTIVE_ACTION
            if state != "PROTECTIVE_ACTION":
                return

        self._last_state = state

        if state == "WARNING":
            self._emit("⚠  WARNING: Hand approaching danger zone "
                       f"| dist={distance_px:.0f}px vel={velocity_px_s:.0f}px/s")
            self._beep(1)

        elif state == "PROTECTIVE_ACTION":
            now = time.monotonic()
            if now - self._last_action_time > 0.25:  # rate-limit to 4 Hz
                self._emit("🛑  [SIMULATED STOP] Protective action triggered "
                           f"| dist={distance_px:.0f}px vel={velocity_px_s:.0f}px/s")
                self._beep(3)
                self._last_action_time = now

        elif state == "FAULT":
            self._emit("❌  FAULT: Camera or detection system failure")
            self._beep(2)

        elif state == "CLEAR":
            self._emit("✅  CLEAR")

    # ------------------------------------------------------------------
    def _emit(self, msg: str) -> None:
        print(f"[SIM OUTPUT] {msg}", flush=True)

    def _beep(self, count: int = 1) -> None:
        if self._audio:
            for _ in range(count):
                sys.stdout.write("\a")
            sys.stdout.flush()
