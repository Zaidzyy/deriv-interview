"""WINNER_SELECTED stage: deterministic winner selection.

Reads metrics FROM THE SAVED metrics.json artifact (not in-memory results) and
applies a fixed rule, so the choice is reproducible from the artifact alone —
this is exactly what validate.py checks.

Rule:
  1. highest primary metric (config selection_metric)
  2. tie-break: higher macro_precision
  3. tie-break: alphabetically-first model name
"""
from __future__ import annotations

from .evaluate import METRIC_KEYS


def select_winner(metrics: dict, selection_metric: str) -> dict:
    if selection_metric not in METRIC_KEYS:
        raise ValueError(
            f"selection_metric {selection_metric!r} not in {METRIC_KEYS}"
        )
    if not metrics:
        raise ValueError("No metrics to select a winner from")

    # Deterministic ranking: sort key makes ties fall through to the next rule.
    # (-primary, -macro_precision, name) ascending == best first.
    def sort_key(item):
        name, m = item
        return (-m[selection_metric], -m["macro_precision"], name)

    ranked = sorted(metrics.items(), key=sort_key)
    winner_name, winner_metrics = ranked[0]

    # Explain the decision, including whether a tie-break was actually used.
    ranking = [
        {
            "model": name,
            "primary_metric": selection_metric,
            "primary_value": m[selection_metric],
            "macro_precision": m["macro_precision"],
        }
        for name, m in ranked
    ]

    reason_parts = [
        f"Selected '{winner_name}': highest {selection_metric} "
        f"({winner_metrics[selection_metric]:.4f})."
    ]
    if len(ranked) > 1:
        runner_name, runner_m = ranked[1]
        if runner_m[selection_metric] == winner_metrics[selection_metric]:
            if runner_m["macro_precision"] == winner_metrics["macro_precision"]:
                reason_parts.append(
                    f"Tie on {selection_metric} and macro_precision with "
                    f"'{runner_name}'; broke tie by alphabetical name."
                )
            else:
                reason_parts.append(
                    f"Tie on {selection_metric} with '{runner_name}'; "
                    f"broke tie by higher macro_precision."
                )

    return {
        "winner": winner_name,
        "selection_metric": selection_metric,
        "winner_metrics": {k: winner_metrics[k] for k in METRIC_KEYS},
        "ranking": ranking,
        "tie_break_rule": ["primary metric", "macro_precision", "alphabetical name"],
        "reason": " ".join(reason_parts),
        "source": "metrics.json",
    }
