"""Stage state machine.

The 12 pipeline stages are strictly ordered. The machine starts at ``INIT`` and
can only move to the *immediate* successor stage. Any attempt to skip a stage or
transition out of order raises :class:`StageError` — the pipeline fails loud.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from enum import IntEnum


class StageError(RuntimeError):
    """Raised on any out-of-order or skipped stage transition."""


class Stage(IntEnum):
    INIT = 0
    DATA_LOADED = 1
    DATA_VALIDATED = 2
    TEXT_PREPROCESSED = 3
    SPLIT_CREATED = 4
    FEATURES_FIT = 5
    MODELS_TRAINED = 6
    MODELS_EVALUATED = 7
    WINNER_SELECTED = 8
    ARTIFACTS_SAVED = 9
    TEST_PREDICTIONS_GENERATED = 10
    REPORT_EXPORTED = 11


# Canonical ordered list — also used by validate.py / README to display stages.
STAGE_ORDER = list(Stage)


class StageMachine:
    """Enforces ordered stage transitions and records per-stage timings."""

    def __init__(self, logger):
        self.current = Stage.INIT
        self._logger = logger
        self.timings: dict[str, float] = {}
        logger.stage_banner(Stage.INIT.name, int(Stage.INIT), note="pipeline start")

    def advance(self, expected: Stage, nxt: Stage) -> None:
        """Assert we are at ``expected`` and move to the immediate successor ``nxt``.

        Raises :class:`StageError` if we are not at ``expected`` or if ``nxt`` is
        not exactly ``expected + 1``.
        """
        if self.current != expected:
            raise StageError(
                f"Out-of-order stage: expected to be at {expected.name} "
                f"but machine is at {self.current.name}"
            )
        if int(nxt) != int(expected) + 1:
            raise StageError(
                f"Illegal transition {expected.name} -> {nxt.name}: "
                f"target must be the immediate successor ({Stage(int(expected) + 1).name})"
            )
        self.current = nxt

    @contextmanager
    def stage(self, target: Stage):
        """Context manager that enters ``target`` (must be current+1), times the
        work, advances on success, and logs a clear ✓/✗ with elapsed ms."""
        expected_prev = Stage(int(target) - 1) if int(target) > 0 else None
        if expected_prev is None or self.current != expected_prev:
            raise StageError(
                f"Cannot enter {target.name}: machine is at {self.current.name}, "
                f"expected {expected_prev.name if expected_prev else 'n/a'}"
            )
        self._logger.stage_start(target.name, int(target))
        t0 = time.perf_counter()
        try:
            yield
        except Exception as exc:  # log the failure, then re-raise (fail loud)
            self._logger.stage_fail(target.name, exc)
            raise
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.advance(expected_prev, target)
        self.timings[target.name] = elapsed_ms
        self._logger.stage_done(target.name, elapsed_ms)
