"""MODELS_EVALUATED stage: consistent metrics for every model.

All numeric computation is done here in Python/scikit-learn (never guessed).
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Keys used everywhere downstream (selection_metric in config maps to these).
METRIC_KEYS = ("accuracy", "macro_precision", "macro_recall", "macro_f1")


def evaluate_model(y_true, y_pred, labels: list[str]) -> dict:
    """Compute accuracy, macro P/R/F1, confusion matrix, and per-class breakdown.

    ``labels`` fixes row/column order of the confusion matrix so it is stable
    and comparable across models.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    per_p, per_r, per_f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    per_class = {
        label: {
            "precision": float(per_p[i]),
            "recall": float(per_r[i]),
            "f1": float(per_f1[i]),
            "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }

    return {
        "accuracy": float(acc),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "confusion_matrix": {
            "labels": list(labels),
            "matrix": cm.tolist(),
        },
        "per_class": per_class,
    }
