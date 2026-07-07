"""Validation suite — verifies the pipeline's outputs and invariants.

    python validate.py

Checks (from the spec):
  * required artifacts exist
  * JSON artifacts are valid
  * required dataset columns are enforced
  * preprocessing is applied consistently in training and inference
  * at least 3 models were trained
  * the winner is selected from saved metrics using deterministic logic
  * predictions are generated for all rows in test.csv
  * the CLI can load saved artifacts and run inference

Exits non-zero if any check fails.
"""
from __future__ import annotations

import json
import os
import sys

try:  # force UTF-8 stdout so output renders on Windows cp1252 consoles
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd

from pipeline import data, io_utils
from pipeline.preprocessing import preprocess_text
from pipeline.select import select_winner

REQUIRED_ARTIFACTS = [
    io_utils.DATA_VALIDATION_REPORT,
    io_utils.PREPROCESSING_PREVIEW,
    io_utils.SPLIT_REPORT,
    io_utils.METRICS,
    io_utils.MODEL_SELECTION_REPORT,
    io_utils.ERROR_ANALYSIS,
    io_utils.TEST_PREDICTIONS,
    io_utils.VECTORIZER_FILE,
    io_utils.WINNER_MODEL_FILE,
]
REQUIRED_JSON = [
    io_utils.DATA_VALIDATION_REPORT,
    io_utils.PREPROCESSING_PREVIEW,
    io_utils.SPLIT_REPORT,
    io_utils.METRICS,
    io_utils.MODEL_SELECTION_REPORT,
    io_utils.ERROR_ANALYSIS,
]


class Checker:
    def __init__(self):
        self.failures: list[str] = []
        self.passes: list[str] = []
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
        unicode_ok = "utf" in enc
        self.MARK_OK = "✓" if unicode_ok else "[OK]"
        self.MARK_FAIL = "✗" if unicode_ok else "[X]"

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        if ok:
            self.passes.append(name)
            print(f"  {self.MARK_OK} {name}")
        else:
            self.failures.append(f"{name}: {detail}")
            print(f"  {self.MARK_FAIL} {name} -- {detail}")
        return ok


def _exists(path: str) -> bool:
    return os.path.exists(path)


def main() -> int:
    c = Checker()
    print("Validating pipeline artifacts and invariants...\n")

    # 1. required artifacts exist
    for path in REQUIRED_ARTIFACTS:
        c.check(f"artifact exists: {os.path.basename(path)}", _exists(path),
                "not found -- run `python main.py`")

    # 2. JSON artifacts are valid
    for path in REQUIRED_JSON:
        if not _exists(path):
            c.check(f"json valid: {os.path.basename(path)}", False, "missing")
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                json.load(fh)
            c.check(f"json valid: {os.path.basename(path)}", True)
        except Exception as exc:  # noqa: BLE001
            c.check(f"json valid: {os.path.basename(path)}", False, str(exc))

    # 3. required dataset columns are enforced
    try:
        train_df = data.load_csv(io_utils.TRAIN_CSV)
        test_df = data.load_csv(io_utils.TEST_CSV)
        train_ok = all(col in train_df.columns for col in data.TRAIN_REQUIRED)
        test_ok = all(col in test_df.columns for col in data.TEST_REQUIRED)
        c.check("dataset columns enforced (train id/text/label, test id/text)",
                train_ok and test_ok, "missing required columns")
    except Exception as exc:  # noqa: BLE001
        c.check("dataset columns enforced", False, str(exc))
        return _finish(c)

    # 4. at least 3 models trained (from metrics.json)
    metrics = {}
    if _exists(io_utils.METRICS):
        metrics = io_utils.read_json(io_utils.METRICS)
    c.check("at least 3 models trained", len(metrics) >= 3,
            f"found {len(metrics)} in metrics.json")

    # 5. winner selected from saved metrics using deterministic logic
    if metrics and _exists(io_utils.MODEL_SELECTION_REPORT):
        report = io_utils.read_json(io_utils.MODEL_SELECTION_REPORT)
        recomputed = select_winner(metrics, report["selection_metric"])
        c.check("winner is deterministic from metrics.json",
                recomputed["winner"] == report["winner"],
                f"report={report['winner']} recomputed={recomputed['winner']}")
        # cross-check model_meta agrees
        if _exists(io_utils.MODEL_META):
            meta = io_utils.read_json(io_utils.MODEL_META)
            c.check("model_meta winner matches selection report",
                    meta.get("winner") == report["winner"],
                    f"meta={meta.get('winner')} report={report['winner']}")
    else:
        c.check("winner is deterministic from metrics.json", False,
                "metrics.json or model_selection_report.json missing")

    # 6. predictions generated for all rows in test.csv
    if _exists(io_utils.TEST_PREDICTIONS):
        preds = pd.read_csv(io_utils.TEST_PREDICTIONS, dtype={"id": str})
        cols_ok = list(preds.columns) == ["id", "predicted_label"]
        ids_match = set(preds["id"]) == set(test_df["id"].astype(str)) and len(preds) == len(test_df)
        c.check("test_predictions.csv has columns id,predicted_label", cols_ok,
                f"columns={list(preds.columns)}")
        c.check("predictions cover all test.csv rows", ids_match,
                f"pred_rows={len(preds)} test_rows={len(test_df)}")
    else:
        c.check("predictions cover all test.csv rows", False, "test_predictions.csv missing")

    # 7. preprocessing consistent in training and inference + 8. CLI loads & infers
    try:
        from predict import predict_text  # imports the SAME preprocess_text

        sample = "  The   APP is    Great!!  "
        result = predict_text(sample)
        c.check("inference applies the shared preprocessing function",
                result["processed"] == preprocess_text(sample),
                "processed text diverged from preprocess_text()")
        classes = []
        if _exists(io_utils.MODEL_META):
            classes = io_utils.read_json(io_utils.MODEL_META).get("classes", [])
        c.check("CLI loads saved artifacts and returns a valid label",
                bool(result["predicted_label"]) and (not classes or result["predicted_label"] in classes),
                f"label={result.get('predicted_label')}")
    except Exception as exc:  # noqa: BLE001
        c.check("CLI loads saved artifacts and runs inference", False, str(exc))

    return _finish(c)


def _finish(c: Checker) -> int:
    print()
    if c.failures:
        print(f"FAILED: {len(c.failures)} check(s) failed, {len(c.passes)} passed.")
        for f in c.failures:
            print(f"  - {f}")
        return 1
    print(f"OK: all {len(c.passes)} checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
