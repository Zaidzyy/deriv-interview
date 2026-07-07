"""Filesystem helpers: config/CSV loading, JSON writing, artifact path constants.

All artifacts are written to the repo root using the EXACT filenames the spec
names, so the evaluator (and validate.py) can find them by name.
"""
from __future__ import annotations

import json
import os

# --- artifact path constants (exact filenames from the spec) -------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(name: str) -> str:
    return os.path.join(ROOT, name)


TRAIN_CSV = _p("train.csv")
TEST_CSV = _p("test.csv")
CONFIG_JSON = _p("config.json")

DATA_VALIDATION_REPORT = _p("data_validation_report.json")
PREPROCESSING_PREVIEW = _p("preprocessing_preview.json")
SPLIT_REPORT = _p("split_report.json")
METRICS = _p("metrics.json")
MODEL_SELECTION_REPORT = _p("model_selection_report.json")
ERROR_ANALYSIS = _p("error_analysis.json")
SAFEGUARDS_REPORT = _p("safeguards_report.json")
RUN_MANIFEST = _p("run_manifest.json")
CROSS_VALIDATION_REPORT = _p("cross_validation_report.json")
TEST_PREDICTIONS = _p("test_predictions.csv")

VECTORIZER_FILE = _p("tfidf_vectorizer.joblib")
WINNER_MODEL_FILE = _p("winner_model.joblib")
MODEL_META = _p("model_meta.json")


def model_file(name: str) -> str:
    """Path for an individual fitted model, e.g. model_logistic_regression.joblib."""
    return _p(f"model_{name}.joblib")


# --- IO helpers ----------------------------------------------------------
def load_config(path: str = CONFIG_JSON) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: str, obj) -> None:
    """Deterministic, human-readable JSON write (sorted keys, 2-space indent)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
        fh.write("\n")


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(o):
    # numpy scalars / arrays -> native python for clean JSON.
    import numpy as np

    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
