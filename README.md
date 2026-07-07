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

- **Enforced 12-stage state machine.** `pipeline/stages.py` holds the ordered stages; the machine
  starts at `INIT` and `advance()` **raises `StageError`** on any skip or reorder. Training
  therefore *cannot* run before the data is loaded, validated, and split — every run follows the
  same auditable path, logged stage-by-stage with ✓/✗ and timings.
- **Deterministic, no LLM.** Pure scikit-learn on CPU — no external APIs, no model calls. All
  numeric work (metrics, selection, counts) is exact Python, so results are repeatable rather than
  sampled.
- **Reproducibility from a single seed.** Every random step (split, models) is seeded from
  `config.random_seed`. Verified two ways: reruns produce **byte-identical `metrics.json`**, and a
  **clean-room run** (delete all artifacts → `python main.py`) regenerates everything from only
  `train.csv` / `test.csv` / `config.json` — nothing is precomputed.
- **No train/serve skew.** A single `preprocess_text()` (lowercase → trim → collapse whitespace)
  in `pipeline/preprocessing.py` is imported by **both** training and `predict.py`, so inference
  cleans text identically to training. `validate.py` asserts the two paths agree.
- **Model comparison & selection.** Three baselines (logistic regression, linear SVM,
  multinomial NB) are trained on the same split and scored on accuracy + macro precision/recall/F1
  with a confusion matrix. The winner is chosen by **macro-F1** — deliberately over accuracy,
  since classes may be imbalanced and accuracy would flatter a majority-class predictor. Selection
  is done **in code** by reading the saved `metrics.json` and applying explicit tie-breaks: higher
  macro precision, then alphabetical model name. The choice is thus reproducible from the artifact
  alone, and the full rationale is written to `model_selection_report.json`.
- **Safeguards against misleading evaluation** (`safeguards_report.json`): warns on high class
  imbalance, on any class missing from the validation split, and on exact duplicate texts shared
  between train and validation. A **stratification guard** detects singleton/tiny classes and
  falls back to a non-stratified split (with a warning) instead of crashing.
- **Fixture-swap robustness.** Because the evaluator may replace the CSVs/config: unknown or
  subset `models` entries are warned-and-skipped (still requiring ≥3 trained, else a clear error);
  an empty-vocabulary vectorizer config (e.g. `min_df` too high for the data) fails with a
  specific, actionable message rather than a raw traceback; and multiclass (>2 label) data flows
  through end-to-end.
- **Separation of concerns.** Training (`main.py` + `pipeline/`) is fully separate from inference
  (`predict.py`), which only loads the saved vectorizer + winning model. Score semantics differ by
  model: `logistic_regression` / `naive_bayes` report `predict_proba`; `linear_svm` (LinearSVC)
  has no calibrated probability, so its `confidence_or_score` is the decision-function margin,
  labelled via `score_kind`.

**How to run:** `python main.py` trains, evaluates, selects, and writes all artifacts;
`python validate.py` checks them (exit 0 = pass); `python predict.py --text "..."` serves a single
prediction. Artifacts are listed under [Artifacts produced](#artifacts-produced).

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
