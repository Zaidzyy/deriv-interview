"""MODELS_TRAINED stage: seeded model factory for the three baselines.

- logistic_regression : linear model, exposes predict_proba
- linear_svm          : LinearSVC, exposes decision_function (no proba)
- naive_bayes         : MultinomialNB, probabilistic baseline
"""
from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

SUPPORTED = ("logistic_regression", "linear_svm", "naive_bayes")


def build_model(name: str, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "linear_svm":
        return LinearSVC(random_state=seed)
    if name == "naive_bayes":
        return MultinomialNB()  # deterministic; no random_state needed
    raise ValueError(f"Unsupported model: {name!r}. Supported: {SUPPORTED}")


def build_models(cfg: dict) -> dict:
    """Instantiate every model named in config, in config order."""
    seed = int(cfg.get("random_seed", 42))
    names = cfg.get("models", list(SUPPORTED))
    return {name: build_model(name, seed) for name in names}


def scores_for(model, X):
    """Return (predicted_labels, score_array_or_None, score_kind).

    score_kind is 'proba' | 'decision' | 'none'. Used for confidence in error
    analysis and the inference CLI. LinearSVC has no calibrated probability.
    """
    preds = model.predict(X)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return preds, proba.max(axis=1), "proba"
    if hasattr(model, "decision_function"):
        dec = model.decision_function(X)
        # Binary case returns 1-D; use absolute margin as the score magnitude.
        if dec.ndim == 1:
            return preds, abs(dec), "decision"
        return preds, dec.max(axis=1), "decision"
    return preds, None, "none"
