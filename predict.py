"""Inference CLI — separate from training.

Loads the saved vectorizer + winning model, applies the SAME shared
preprocessing function used in training (no train/serve skew), and prints the
predicted label plus a confidence/score when the model exposes one.

    python predict.py --text "The app is easy to use"
"""
from __future__ import annotations

import argparse
import json
import sys

import joblib

from pipeline import io_utils
from pipeline.models import scores_for
from pipeline.preprocessing import preprocess_text  # the SAME function as training


def load_artifacts():
    try:
        vectorizer = joblib.load(io_utils.VECTORIZER_FILE)
        model = joblib.load(io_utils.WINNER_MODEL_FILE)
    except FileNotFoundError as exc:
        sys.exit(
            f"error: missing saved artifact ({exc.filename}). "
            f"Run `python main.py` first to train and save the model."
        )
    meta = {}
    try:
        meta = io_utils.read_json(io_utils.MODEL_META)
    except FileNotFoundError:
        pass
    return vectorizer, model, meta


def predict_text(text: str) -> dict:
    vectorizer, model, meta = load_artifacts()
    processed = preprocess_text(text)
    X = vectorizer.transform([processed])
    preds, scores, score_kind = scores_for(model, X)
    return {
        "input": text,
        "processed": processed,
        "predicted_label": preds[0],
        "confidence_or_score": None if scores is None else float(scores[0]),
        "score_kind": score_kind,
        "model": meta.get("winner"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-text inference for the winning model.")
    parser.add_argument("--text", required=True, help="Text to classify")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    args = parser.parse_args()

    result = predict_text(args.text)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"predicted_label: {result['predicted_label']}")
    if result["confidence_or_score"] is None:
        print("confidence_or_score: n/a (model exposes no score)")
    else:
        print(f"confidence_or_score: {result['confidence_or_score']:.4f} ({result['score_kind']})")


if __name__ == "__main__":
    main()
