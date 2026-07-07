"""Fast, offline unit tests for the deterministic pipeline."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline import data
from pipeline.preprocessing import preprocess_text
from pipeline.safeguards import Safeguards
from pipeline.select import select_winner
from pipeline.split import make_split
from pipeline.stages import Stage, StageError, StageMachine


class _NullLogger:
    def stage_banner(self, *a, **k): ...
    def stage_start(self, *a, **k): ...
    def stage_done(self, *a, **k): ...
    def stage_fail(self, *a, **k): ...


# --- preprocessing (the shared train/serve function) ---------------------
def test_preprocess_lowercases_trims_collapses():
    assert preprocess_text("  The   APP   is  GREAT ") == "the app is great"


def test_preprocess_handles_none_and_nonstring():
    assert preprocess_text(None) == ""
    assert preprocess_text(12345) == "12345"


# --- stage machine cannot skip or reorder --------------------------------
def test_stage_machine_rejects_skip():
    m = StageMachine(_NullLogger())
    with pytest.raises(StageError):
        # cannot jump straight to DATA_VALIDATED, skipping DATA_LOADED
        with m.stage(Stage.DATA_VALIDATED):
            pass


def test_stage_machine_allows_ordered_advance():
    m = StageMachine(_NullLogger())
    with m.stage(Stage.DATA_LOADED):
        pass
    assert m.current == Stage.DATA_LOADED
    with m.stage(Stage.DATA_VALIDATED):
        pass
    assert m.current == Stage.DATA_VALIDATED


# --- validation enforces required columns / rules ------------------------
def test_validate_requires_two_labels():
    train = pd.DataFrame({"id": ["1", "2"], "text": ["a", "b"], "label": ["x", "x"]})
    test = pd.DataFrame({"id": ["9"], "text": ["c"]})
    with pytest.raises(data.ValidationError):
        data.validate(train, test)


def test_validate_rejects_duplicate_ids():
    train = pd.DataFrame({"id": ["1", "1"], "text": ["a", "b"], "label": ["x", "y"]})
    test = pd.DataFrame({"id": ["9"], "text": ["c"]})
    with pytest.raises(data.ValidationError):
        data.validate(train, test)


# --- deterministic winner selection & tie-breaks -------------------------
def test_select_winner_tie_breaks_by_precision_then_name():
    metrics = {
        "naive_bayes": {"macro_f1": 0.8, "macro_precision": 0.8,
                        "macro_recall": 0.8, "accuracy": 0.8},
        "linear_svm": {"macro_f1": 0.8, "macro_precision": 0.9,
                       "macro_recall": 0.7, "accuracy": 0.8},
        "logistic_regression": {"macro_f1": 0.8, "macro_precision": 0.9,
                                "macro_recall": 0.7, "accuracy": 0.8},
    }
    # tie on f1: linear_svm & logistic_regression share top precision -> alphabetical
    result = select_winner(metrics, "macro_f1")
    assert result["winner"] == "linear_svm"


# --- stratification guard falls back instead of crashing -----------------
def test_split_falls_back_when_class_is_singleton():
    df = pd.DataFrame({
        "id": [str(i) for i in range(6)],
        "text": [f"t{i}" for i in range(6)],
        "processed": [f"t{i}" for i in range(6)],
        "label": ["a", "a", "a", "a", "a", "b"],  # 'b' is a singleton
    })
    sg = Safeguards()
    train_df, val_df, stratified = make_split(df, seed=42, val_frac=0.34, safeguards=sg)
    assert stratified is False
    assert len(train_df) + len(val_df) == 6
    assert any(f["check"] == "stratification_fallback" for f in sg.findings)
