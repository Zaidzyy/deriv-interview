"""Defensive-guard tests: added alongside existing tests; no pipeline behavior change.

Covers:
  * partition_models splits supported vs unknown model names
  * predict.py returns a label + score for each winner type
      - LogisticRegression -> predict_proba  (score_kind == 'proba')
      - LinearSVC          -> decision_function (score_kind == 'decision')
      - MultinomialNB      -> predict_proba  (score_kind == 'proba')
  * a >2-class (multiclass) dataset runs the full pipeline end-to-end and
    validate.py stays green (23/23) -- isolated in a tmp dir so the real repo
    artifacts are untouched.
"""
from __future__ import annotations

import importlib
import json
import os

import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from pipeline.models import partition_models
from pipeline.preprocessing import preprocess_text


# --- Guard 1: partition supported / unknown model names ------------------
def test_partition_models_splits_supported_and_unknown():
    supported, unknown = partition_models(
        ["logistic_regression", "mystery_model", "naive_bayes", "linear_svm"]
    )
    assert supported == ["logistic_regression", "naive_bayes", "linear_svm"]
    assert unknown == ["mystery_model"]


# --- predict.py returns label + score for each model type ----------------
def _dump_winner_and_patch(tmp_path, monkeypatch, model):
    """Fit vectorizer+model on a tiny binary corpus, save as the winner artifacts,
    and point io_utils at them so predict.predict_text loads them."""
    from pipeline import io_utils

    texts = ["good great nice smooth", "bad awful terrible broken",
             "great and nice", "awful and broken"]
    labels = ["pos", "neg", "pos", "neg"]
    vec = TfidfVectorizer()
    X = vec.fit_transform([preprocess_text(t) for t in texts])
    model.fit(X, labels)

    vpath = os.path.join(tmp_path, "v.joblib")
    mpath = os.path.join(tmp_path, "m.joblib")
    metapath = os.path.join(tmp_path, "meta.json")
    joblib.dump(vec, vpath)
    joblib.dump(model, mpath)
    with open(metapath, "w", encoding="utf-8") as fh:
        json.dump({"winner": type(model).__name__, "classes": ["neg", "pos"]}, fh)

    monkeypatch.setattr(io_utils, "VECTORIZER_FILE", vpath)
    monkeypatch.setattr(io_utils, "WINNER_MODEL_FILE", mpath)
    monkeypatch.setattr(io_utils, "MODEL_META", metapath)


@pytest.mark.parametrize("model,expected_kind", [
    (LogisticRegression(max_iter=1000, random_state=42), "proba"),
    (LinearSVC(random_state=42), "decision"),
    (MultinomialNB(), "proba"),
])
def test_predict_returns_label_and_score(tmp_path, monkeypatch, model, expected_kind):
    _dump_winner_and_patch(tmp_path, monkeypatch, model)
    from predict import predict_text

    result = predict_text("good and great and nice")
    assert result["predicted_label"] in {"pos", "neg"}
    assert result["confidence_or_score"] is not None
    assert isinstance(result["confidence_or_score"], float)
    assert result["score_kind"] == expected_kind


# --- Multiclass (>2 classes) full pipeline + validate green --------------
_ARTIFACT_ATTRS = [
    "TRAIN_CSV", "TEST_CSV", "CONFIG_JSON",
    "DATA_VALIDATION_REPORT", "PREPROCESSING_PREVIEW", "SPLIT_REPORT", "METRICS",
    "MODEL_SELECTION_REPORT", "ERROR_ANALYSIS", "SAFEGUARDS_REPORT", "RUN_MANIFEST",
    "CROSS_VALIDATION_REPORT", "TEST_PREDICTIONS", "VECTORIZER_FILE",
    "WINNER_MODEL_FILE", "MODEL_META",
]

_MULTICLASS_TRAIN = """id,text,label
1,"totally love this amazing product",positive
2,"fantastic support solved it fast",positive
3,"great value would buy again",positive
4,"wonderful clean and helpful interface",positive
5,"terrible crashes and lost my data",negative
6,"awful slow and unreliable service",negative
7,"hate the confusing broken layout",negative
8,"worst update it broke everything",negative
9,"it is an app that simply exists",neutral
10,"average nothing special either way",neutral
11,"ok fine neither good nor bad",neutral
12,"a plain ordinary standard tool",neutral
"""

_MULTICLASS_TEST = """id,text
101,"this is wonderful and helpful"
102,"broken and frustrating to use"
103,"just an ordinary average thing"
"""

_MULTICLASS_CONFIG = {
    "random_seed": 42,
    "validation_split": 0.25,
    "models": ["logistic_regression", "linear_svm", "naive_bayes"],
    "vectorizer": {"type": "tfidf", "ngram_range": [1, 2], "max_features": 5000, "min_df": 1},
    "selection_metric": "macro_f1",
    "top_k_error_examples": 10,
    "class_imbalance_ratio_warn": 3.0,
    "cross_validation": {"enabled": True, "folds": 2},
}


def test_multiclass_full_pipeline_and_validate(tmp_path, monkeypatch):
    from pipeline import io_utils

    # Redirect every input + artifact path into the tmp dir.
    for attr in _ARTIFACT_ATTRS:
        base = os.path.basename(getattr(io_utils, attr))
        monkeypatch.setattr(io_utils, attr, os.path.join(tmp_path, base))
    monkeypatch.setattr(io_utils, "ROOT", str(tmp_path))  # model_file() -> tmp

    # Write the 3-class fixtures + config into tmp.
    (tmp_path / "train.csv").write_text(_MULTICLASS_TRAIN, encoding="utf-8")
    (tmp_path / "test.csv").write_text(_MULTICLASS_TEST, encoding="utf-8")
    (tmp_path / "config.json").write_text(json.dumps(_MULTICLASS_CONFIG), encoding="utf-8")
    monkeypatch.setattr(
        io_utils, "load_config",
        lambda path=None: json.loads((tmp_path / "config.json").read_text(encoding="utf-8")),
    )

    # Run the real training pipeline end-to-end on 3 classes.
    import main as main_mod
    result = main_mod.run()
    assert len(result["metrics"]) == 3  # three models evaluated

    # Multiclass sanity: three distinct labels flowed through.
    meta = json.loads((tmp_path / "model_meta.json").read_text(encoding="utf-8"))
    assert set(meta["classes"]) == {"positive", "negative", "neutral"}

    # CV report generated with per-model scores.
    cv = json.loads((tmp_path / "cross_validation_report.json").read_text(encoding="utf-8"))
    assert set(cv["results"].keys()) == {"logistic_regression", "linear_svm", "naive_bayes"}

    # validate.py must stay green against the tmp artifacts.
    import validate as validate_mod
    importlib.reload(validate_mod)  # rebuild REQUIRED_* lists from patched io_utils
    assert validate_mod.main() == 0
