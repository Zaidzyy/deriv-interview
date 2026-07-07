# Replayable Text-Classification Pipeline

A deterministic, local (CPU-only, no external APIs, no LLM) scikit-learn pipeline that
trains three baseline text classifiers, evaluates them consistently, selects a deployable
winner in code, saves reusable artifacts, and serves single-text predictions via a CLI.

## Quick start

```bash
pip install -r requirements.txt

python main.py                                   # train + evaluate + select + export artifacts
python validate.py                               # verify artifacts & invariants (exit 0 = pass)
python predict.py --text "The app is easy to use"  # single-text inference
python -m pytest -q                              # unit tests
```

The repo ships with `train.csv`, `test.csv`, and `config.json` so it runs immediately from a
clean checkout. Generated artifacts are git-ignored and recreated on every `python main.py`
(nothing is precomputed — it always retrains).

## Pipeline stages (enforced state machine)

`pipeline/stages.py` defines a `StageMachine` that starts at `INIT` and can only move to the
*immediate* successor. Any skip or out-of-order transition raises `StageError`, so the pipeline
fails loud. Each stage is logged with a ✓/✗ banner and timing.

```
INIT → DATA_LOADED → DATA_VALIDATED → TEXT_PREPROCESSED → SPLIT_CREATED → FEATURES_FIT
     → MODELS_TRAINED → MODELS_EVALUATED → WINNER_SELECTED → ARTIFACTS_SAVED
     → TEST_PREDICTIONS_GENERATED → REPORT_EXPORTED
```

## Design decisions

- **Deterministic & reproducible.** All randomness is seeded from `config.random_seed`; reruns
  produce byte-identical `metrics.json`. All numeric work (metrics, selection) is plain Python /
  scikit-learn.
- **Shared preprocessing (no train/serve skew).** `pipeline/preprocessing.preprocess_text` is the
  single lowercase→trim→collapse-whitespace function imported by **both** training and
  `predict.py`. `validate.py` asserts inference reproduces it exactly.
- **Winner selection reads the saved artifact.** `pipeline/select.py` loads `metrics.json` from
  disk (not in-memory results) and applies a fixed rule: highest `selection_metric`, tie-break on
  macro precision, then alphabetical model name. The full decision is written to
  `model_selection_report.json`.
- **Stratification guard.** On small fixtures where a class has too few samples to stratify, the
  split falls back to a non-stratified split and emits a safeguard warning instead of crashing.
- **Training vs inference separated.** `main.py` + `pipeline/` train; `predict.py` only loads
  saved artifacts and infers.
- **Score semantics.** `logistic_regression` / `naive_bayes` report `predict_proba`; `linear_svm`
  (LinearSVC) has no calibrated probability, so its `confidence_or_score` is the decision-function
  margin (labelled via `score_kind`).

## Module layout

```
main.py            # orchestrator: runs the 12 stages through the StageMachine
predict.py         # inference CLI (loads saved vectorizer + winner, shared preprocessing)
validate.py        # validation suite (every check the spec lists)
config.json        # seed, split, models, vectorizer, selection_metric, top_k, CV flag
train.csv test.csv # sample fixtures (evaluator may replace with same-shape fixtures)
pipeline/
  stages.py          # Stage enum + StageMachine (raises on out-of-order transitions)
  logging_utils.py   # colored per-stage banners, ✓/✗, timings, summary table
  io_utils.py        # config/CSV loading, JSON writing, artifact path constants
  preprocessing.py   # THE shared preprocess_text() (train + inference)
  data.py            # load + validate (required cols, >=2 labels, non-empty text, unique ids)
  features.py        # TF-IDF vectorizer from config
  models.py          # seeded model factory + score extraction
  evaluate.py        # accuracy, macro P/R/F1, confusion matrix, per-class breakdown
  select.py          # deterministic winner selection from metrics.json
  error_analysis.py  # top-k most-confident misclassified validation examples
  safeguards.py      # class imbalance / missing-val-class / train-val duplicate checks
  cross_validation.py# optional k-fold CV (config-gated stretch)
tests/test_pipeline.py
```

## Configuration (`config.json`)

| key | meaning |
|-----|---------|
| `random_seed` | seeds split + models |
| `validation_split` | validation fraction |
| `models` | subset of `logistic_regression`, `linear_svm`, `naive_bayes` |
| `vectorizer` | tfidf `ngram_range`, `max_features`, `min_df` |
| `selection_metric` | one of `macro_f1`, `macro_precision`, `macro_recall`, `accuracy` |
| `top_k_error_examples` | number of misclassified examples to log |
| `class_imbalance_ratio_warn` | max/min class ratio that triggers a safeguard warning |
| `cross_validation.enabled` | run optional k-fold CV before selection |

## Artifacts produced

Required: `data_validation_report.json`, `preprocessing_preview.json`, `split_report.json`,
`metrics.json`, `model_selection_report.json`, `error_analysis.json`, `test_predictions.csv`
(`id,predicted_label`), `tfidf_vectorizer.joblib`, `winner_model.joblib`.

Also: `run_manifest.json`, `safeguards_report.json`, `model_meta.json`, and
`cross_validation_report.json` (when CV is enabled).

## Docker

```bash
docker build -t textpipe .
docker run --rm textpipe          # runs the pipeline then validate.py
```
