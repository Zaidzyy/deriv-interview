"""Orchestrator for the deterministic text-classification training pipeline.

Runs the 12 stages in order through the StageMachine (which raises on any
out-of-order transition), logs each stage, and writes every required artifact.

    python main.py                # run on the committed train.csv/test.csv/config.json

Inference is intentionally separate: see predict.py.
"""
from __future__ import annotations

import datetime as _dt

import joblib
import pandas as pd

from pipeline import data, io_utils
from pipeline.cross_validation import run_cross_validation
from pipeline.error_analysis import top_misclassified
from pipeline.evaluate import METRIC_KEYS, evaluate_model
from pipeline.features import build_vectorizer
from pipeline.logging_utils import Logger
from pipeline.models import build_models, scores_for
from pipeline.preprocessing import preprocess_text
from pipeline.safeguards import Safeguards
from pipeline.select import select_winner
from pipeline.split import make_split, split_report
from pipeline.stages import Stage, StageMachine


def run() -> dict:
    logger = Logger()
    machine = StageMachine(logger)
    safeguards = Safeguards()

    # ---- DATA_LOADED ----------------------------------------------------
    with machine.stage(Stage.DATA_LOADED):
        cfg = io_utils.load_config()
        seed = int(cfg.get("random_seed", 42))
        train_df = data.load_csv(io_utils.TRAIN_CSV)
        test_df = data.load_csv(io_utils.TEST_CSV)
        logger.info(f"train.csv: {len(train_df)} rows | test.csv: {len(test_df)} rows | seed={seed}")

    # ---- DATA_VALIDATED -------------------------------------------------
    with machine.stage(Stage.DATA_VALIDATED):
        report = data.validate(train_df, test_df)
        io_utils.write_json(io_utils.DATA_VALIDATION_REPORT, report)
        labels = report["distinct_labels"]
        logger.ok(f"validation passed | labels={labels}")

    # ---- TEXT_PREPROCESSED (shared preprocess_text) ---------------------
    with machine.stage(Stage.TEXT_PREPROCESSED):
        train_df["processed"] = train_df["text"].map(preprocess_text)
        test_df["processed"] = test_df["text"].map(preprocess_text)
        preview = [
            {"id": r.id, "original": r.text, "processed": r.processed}
            for r in train_df.head(5).itertuples()
        ]
        io_utils.write_json(io_utils.PREPROCESSING_PREVIEW,
                            {"n_previewed": len(preview), "examples": preview})
        logger.info(f"preprocessed {len(train_df)} train + {len(test_df)} test texts")

    # ---- SPLIT_CREATED --------------------------------------------------
    with machine.stage(Stage.SPLIT_CREATED):
        val_frac = float(cfg.get("validation_split", 0.2))
        train_split, val_split, stratified = make_split(train_df, seed, val_frac, safeguards)
        io_utils.write_json(io_utils.SPLIT_REPORT,
                            split_report(seed, train_split, val_split, stratified))
        # Safeguard checks (findings written at REPORT_EXPORTED).
        safeguards.check_class_imbalance(train_df["label"].tolist(),
                                         float(cfg.get("class_imbalance_ratio_warn", 3.0)))
        safeguards.check_missing_val_classes(train_df["label"].tolist(),
                                             val_split["label"].tolist())
        safeguards.check_train_val_duplicates(train_split["processed"].tolist(),
                                              val_split["processed"].tolist())
        logger.info(f"train={len(train_split)} val={len(val_split)} stratified={stratified}")

    # ---- FEATURES_FIT ---------------------------------------------------
    with machine.stage(Stage.FEATURES_FIT):
        vectorizer = build_vectorizer(cfg)
        X_train = vectorizer.fit_transform(train_split["processed"])
        X_val = vectorizer.transform(val_split["processed"])
        y_train = train_split["label"].tolist()
        y_val = val_split["label"].tolist()
        logger.info(f"tfidf features: {X_train.shape[1]} | ngram={cfg['vectorizer']['ngram_range']}")

    # ---- MODELS_TRAINED -------------------------------------------------
    with machine.stage(Stage.MODELS_TRAINED):
        models = build_models(cfg)
        for name, model in models.items():
            model.fit(X_train, y_train)
            logger.ok(f"trained {name}")

    # ---- MODELS_EVALUATED -----------------------------------------------
    with machine.stage(Stage.MODELS_EVALUATED):
        metrics = {}
        for name, model in models.items():
            preds = model.predict(X_val)
            metrics[name] = evaluate_model(y_val, preds, labels)
            logger.info(f"{name}: macro_f1={metrics[name]['macro_f1']:.4f} "
                        f"acc={metrics[name]['accuracy']:.4f}")
        io_utils.write_json(io_utils.METRICS, metrics)

        cv_cfg = cfg.get("cross_validation", {})
        if cv_cfg.get("enabled"):
            cv_report = run_cross_validation(cfg, train_df["processed"].tolist(),
                                             train_df["label"].tolist())
            io_utils.write_json(io_utils.CROSS_VALIDATION_REPORT, cv_report)
            logger.ok(f"cross-validation ({cv_report['folds']} folds) written")

    # ---- WINNER_SELECTED (reads metrics.json from disk) -----------------
    with machine.stage(Stage.WINNER_SELECTED):
        saved_metrics = io_utils.read_json(io_utils.METRICS)  # re-load from artifact
        selection = select_winner(saved_metrics, cfg.get("selection_metric", "macro_f1"))
        io_utils.write_json(io_utils.MODEL_SELECTION_REPORT, selection)
        winner_name = selection["winner"]
        winner_model = models[winner_name]
        logger.ok(selection["reason"])

        # Error analysis for the winner.
        w_preds, w_scores, score_kind = scores_for(winner_model, X_val)
        errors = top_misclassified(
            val_split["id"].tolist(), val_split["text"].tolist(),
            y_val, list(w_preds), w_scores, score_kind,
            int(cfg.get("top_k_error_examples", 10)),
        )
        io_utils.write_json(io_utils.ERROR_ANALYSIS,
                            {"winner": winner_name, "score_kind": score_kind,
                             "n_misclassified": len(errors), "examples": errors})
        logger.info(f"error analysis: {len(errors)} misclassified example(s) logged")

    # ---- ARTIFACTS_SAVED ------------------------------------------------
    with machine.stage(Stage.ARTIFACTS_SAVED):
        joblib.dump(vectorizer, io_utils.VECTORIZER_FILE)
        for name, model in models.items():
            joblib.dump(model, io_utils.model_file(name))
        joblib.dump(winner_model, io_utils.WINNER_MODEL_FILE)
        io_utils.write_json(io_utils.MODEL_META, {
            "winner": winner_name,
            "classes": sorted(labels),
            "score_kind": score_kind,
            "selection_metric": cfg.get("selection_metric", "macro_f1"),
            "vectorizer_file": "tfidf_vectorizer.joblib",
            "winner_model_file": "winner_model.joblib",
        })
        logger.ok(f"saved vectorizer + {len(models)} models + winner ({winner_name})")

    # ---- TEST_PREDICTIONS_GENERATED -------------------------------------
    with machine.stage(Stage.TEST_PREDICTIONS_GENERATED):
        X_test = vectorizer.transform(test_df["processed"])
        test_preds = winner_model.predict(X_test)
        out = pd.DataFrame({"id": test_df["id"], "predicted_label": test_preds})
        out.to_csv(io_utils.TEST_PREDICTIONS, index=False)
        logger.info(f"wrote {len(out)} predictions for all test rows")

    # ---- REPORT_EXPORTED ------------------------------------------------
    with machine.stage(Stage.REPORT_EXPORTED):
        io_utils.write_json(io_utils.SAFEGUARDS_REPORT, safeguards.to_report())
        manifest = {
            "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "random_seed": seed,
            "files_read": ["train.csv", "test.csv", "config.json"],
            "models_trained": list(models.keys()),
            "winning_model": winner_name,
            "selection_metric": cfg.get("selection_metric", "macro_f1"),
            "key_metrics": {k: saved_metrics[winner_name][k] for k in METRIC_KEYS},
            "artifacts": [
                "data_validation_report.json", "preprocessing_preview.json",
                "split_report.json", "metrics.json", "model_selection_report.json",
                "error_analysis.json", "safeguards_report.json", "test_predictions.csv",
                "tfidf_vectorizer.joblib", "winner_model.joblib", "model_meta.json",
                "run_manifest.json",
            ],
        }
        io_utils.write_json(io_utils.RUN_MANIFEST, manifest)
        logger.ok("run_manifest.json + safeguards_report.json written")

    logger.summary_table(machine.timings, winner_name,
                         cfg.get("selection_metric", "macro_f1"),
                         saved_metrics[winner_name][cfg.get("selection_metric", "macro_f1")])
    return {"winner": winner_name, "metrics": saved_metrics}


if __name__ == "__main__":
    run()
