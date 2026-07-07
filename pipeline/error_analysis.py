"""Error analysis: top-k misclassified validation examples for the winner.

Sorted by score descending so the most *confident mistakes* surface first —
those are the most useful to inspect. For models without calibrated
probabilities (LinearSVC) the score is the decision-function margin; if no score
is available it is null.
"""
from __future__ import annotations


def top_misclassified(ids, texts, y_true, y_pred, scores, score_kind: str, top_k: int) -> list[dict]:
    records = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            continue
        score = None if scores is None else float(scores[i])
        records.append(
            {
                "id": ids[i],
                "text": texts[i],
                "true_label": y_true[i],
                "predicted_label": y_pred[i],
                "confidence_or_score": score,
                "score_kind": score_kind,
                "reason": (
                    "Model predicted the wrong class"
                    + (
                        f" with {score_kind} score {score:.4f}; high scores here mean confident mistakes worth inspecting"
                        if score is not None
                        else "; no calibrated score available for this model"
                    )
                ),
            }
        )

    # Most confident mistakes first (None scores sink to the bottom).
    records.sort(key=lambda r: (r["confidence_or_score"] is not None,
                                r["confidence_or_score"] or 0.0),
                 reverse=True)
    return records[:top_k]
