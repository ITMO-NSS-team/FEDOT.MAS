"""Tests for SignalStopper."""

from __future__ import annotations

import signal

from fedotmas.optimize._state import OptimizationState
from fedotmas.optimize._stopping import SignalStopper


class TestSignalStopper:
    def test_not_triggered_by_default(self):
        s = SignalStopper()
        state = OptimizationState()
        assert s.should_stop(state, 0) is False

    def test_triggered_stops(self):
        s = SignalStopper()
        s._triggered = True
        state = OptimizationState()
        assert s.should_stop(state, 0) is True

    def test_reset(self):
        s = SignalStopper()
        s._triggered = True
        s.reset()
        assert s.should_stop(OptimizationState(), 0) is False

    def test_install_uninstall(self):
        s = SignalStopper()
        prev_handler = signal.getsignal(signal.SIGINT)
        s.install()
        assert signal.getsignal(signal.SIGINT) != prev_handler
        s.uninstall()
        assert signal.getsignal(signal.SIGINT) is prev_handler

    def test_context_manager(self):
        prev_handler = signal.getsignal(signal.SIGINT)
        s = SignalStopper()
        with s:
            assert signal.getsignal(signal.SIGINT) != prev_handler
        assert signal.getsignal(signal.SIGINT) is prev_handler

    def test_context_manager_returns_self(self):
        s = SignalStopper()
        with s as entered:
            assert entered is s
