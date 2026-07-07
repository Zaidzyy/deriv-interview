"""Optional STRETCH: k-fold cross-validation on the full training set.

Config-gated via config["cross_validation"]["enabled"]. Uses a Pipeline so the
vectorizer is refit inside each fold (no leakage). Deterministic via seed.
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from .features import build_vectorizer
from .models import build_model

# scikit-learn scorer names for our macro metrics.
_SCORER = {
    "macro_f1": "f1_macro",
    "macro_precision": "precision_macro",
    "macro_recall": "recall_macro",
    "accuracy": "accuracy",
}


def run_cross_validation(cfg: dict, X_text, y) -> dict:
    cv_cfg = cfg.get("cross_validation", {})
    folds = int(cv_cfg.get("folds", 5))
    seed = int(cfg.get("random_seed", 42))
    selection_metric = cfg.get("selection_metric", "macro_f1")
    scorer = _SCORER.get(selection_metric, "f1_macro")

    # Guard: folds cannot exceed the smallest class count.
    from collections import Counter
    min_count = min(Counter(y).values())
    folds = max(2, min(folds, min_count))

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    results = {}
    for name in cfg.get("models", []):
        pipe = Pipeline([
            ("tfidf", build_vectorizer(cfg)),
            ("clf", build_model(name, seed)),
        ])
        scores = cross_val_score(pipe, X_text, y, cv=skf, scoring=scorer)
        results[name] = {
            "scorer": scorer,
            "fold_scores": [float(s) for s in scores],
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
        }
    return {"folds": folds, "selection_metric": selection_metric, "results": results}
