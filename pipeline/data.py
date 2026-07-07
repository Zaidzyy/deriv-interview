"""DATA_LOADED and DATA_VALIDATED stages (deterministic, no LLM)."""
from __future__ import annotations

import pandas as pd

TRAIN_REQUIRED = ["id", "text", "label"]
TEST_REQUIRED = ["id", "text"]


class ValidationError(RuntimeError):
    """Raised when input data fails a required validation check."""


def load_csv(path: str) -> pd.DataFrame:
    # dtype=str for id/text keeps ids stable (no int/float coercion of "101").
    return pd.read_csv(path, dtype={"id": str, "text": str})


def _check_columns(df: pd.DataFrame, required: list[str], name: str, errors: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"{name}: missing required column(s): {missing}")


def _check_unique_ids(df: pd.DataFrame, name: str, errors: list[str]) -> None:
    if "id" not in df.columns:
        return
    dupes = df["id"][df["id"].duplicated()].unique().tolist()
    if dupes:
        errors.append(f"{name}: duplicate id(s): {dupes}")


def _check_nonempty_text(df: pd.DataFrame, name: str, errors: list[str]) -> None:
    if "text" not in df.columns:
        return
    stripped = df["text"].fillna("").astype(str).str.strip()
    n_empty = int((stripped == "").sum())
    if n_empty:
        errors.append(f"{name}: {n_empty} row(s) have empty text after stripping")


def validate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Run every required validation check. Returns a report dict.

    Raises :class:`ValidationError` (with all failures) if any check fails.
    """
    errors: list[str] = []

    _check_columns(train_df, TRAIN_REQUIRED, "train.csv", errors)
    _check_columns(test_df, TEST_REQUIRED, "test.csv", errors)

    distinct_labels = []
    if "label" in train_df.columns:
        distinct_labels = sorted(train_df["label"].dropna().unique().tolist())
        if len(distinct_labels) < 2:
            errors.append(
                f"train.csv: needs >= 2 distinct labels, found {len(distinct_labels)}: {distinct_labels}"
            )

    _check_nonempty_text(train_df, "train.csv", errors)
    _check_nonempty_text(test_df, "test.csv", errors)
    _check_unique_ids(train_df, "train.csv", errors)
    _check_unique_ids(test_df, "test.csv", errors)

    report = {
        "passed": len(errors) == 0,
        "errors": errors,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "train_columns": list(train_df.columns),
        "test_columns": list(test_df.columns),
        "distinct_labels": distinct_labels,
        "n_distinct_labels": len(distinct_labels),
    }
    if errors:
        raise ValidationError("; ".join(errors))
    return report
