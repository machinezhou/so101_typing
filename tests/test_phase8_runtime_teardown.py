from __future__ import annotations

from pathlib import Path
import signal
import sys
import unittest
from unittest.mock import Mock, patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import phase8_single_key_integration as phase8


class Phase8RuntimeTeardownTests(unittest.TestCase):
    def setUp(self) -> None:
        self.saved_watcher = phase8.phase5.ACTIVE_SCREEN_WATCHER
        self.saved_raise = phase8.phase5.raise_if_screen_event
        self.saved_wait = phase8.phase5.wait_motion_stable

    def tearDown(self) -> None:
        phase8.phase5.ACTIVE_SCREEN_WATCHER = self.saved_watcher
        phase8.phase5.raise_if_screen_event = self.saved_raise
        phase8.phase5.wait_motion_stable = self.saved_wait

    def _devices(self, connected: bool):
        top = Mock()
        wrist = Mock()
        side = Mock()
        robot = Mock()
        robot.is_connected = connected
        return top, wrist, side, robot

    def test_teardown_stops_devices_restores_hooks_and_disconnects(self) -> None:
        top, wrist, side, robot = self._devices(True)
        phase8.phase5.ACTIVE_SCREEN_WATCHER = object()

        original_raise = object()
        original_wait = object()
        previous_sigint = object()

        with patch.object(phase8.signal, "signal") as restore_signal:
            phase8._stop_runtime_devices(
                top=top,
                wrist=wrist,
                side=side,
                robot=robot,
                original_raise_if_screen_event=original_raise,
                original_wait_motion_stable=original_wait,
                signal_installed=True,
                previous_sigint=previous_sigint,
            )

        self.assertIsNone(phase8.phase5.ACTIVE_SCREEN_WATCHER)
        top.stop.assert_called_once_with()
        wrist.stop.assert_called_once_with()
        side.stop.assert_called_once_with()
        self.assertIs(phase8.phase5.raise_if_screen_event, original_raise)
        self.assertIs(phase8.phase5.wait_motion_stable, original_wait)
        restore_signal.assert_called_once_with(signal.SIGINT, previous_sigint)
        robot.disconnect.assert_called_once_with()

    def test_teardown_skips_signal_restore_and_disconnect_when_inactive(self) -> None:
        top, wrist, side, robot = self._devices(False)

        with patch.object(phase8.signal, "signal") as restore_signal:
            phase8._stop_runtime_devices(
                top=top,
                wrist=wrist,
                side=side,
                robot=robot,
                original_raise_if_screen_event=object(),
                original_wait_motion_stable=object(),
                signal_installed=False,
                previous_sigint=object(),
            )

        restore_signal.assert_not_called()
        robot.disconnect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
