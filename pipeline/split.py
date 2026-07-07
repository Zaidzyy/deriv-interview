"""SPLIT_CREATED stage: stratified train/validation split with a safety guard.

On small evaluator fixtures a class may have a single sample, which makes
stratified splitting impossible. Rather than crash, we fall back to a plain
(non-stratified) split and emit a safeguard warning — a deliberate guard against
a hard failure on their data.
"""
from __future__ import annotations

from collections import Counter

from sklearn.model_selection import train_test_split


def make_split(df, seed: int, val_frac: float, safeguards):
    """Return (train_df, val_df, stratified: bool). ``df`` must have a 'label' column."""
    labels = df["label"].tolist()
    counts = Counter(labels)
    min_count = min(counts.values())
    n_classes = len(counts)
    n_val = max(1, int(round(len(df) * val_frac)))

    # Stratify needs >= 2 samples per class AND at least one val slot per class.
    can_stratify = min_count >= 2 and n_val >= n_classes
    stratify = df["label"] if can_stratify else None
    stratified = can_stratify

    if not can_stratify:
        safeguards.add(
            "warning", "stratification_fallback",
            "Could not stratify split (a class is too small); used a "
            "non-stratified split instead",
            {"class_counts": dict(counts), "n_val": n_val, "n_classes": n_classes},
        )

    try:
        train_df, val_df = train_test_split(
            df, test_size=val_frac, random_state=seed, stratify=stratify, shuffle=True
        )
    except ValueError:
        # Belt-and-suspenders: any residual stratify error -> plain split.
        stratified = False
        safeguards.add(
            "warning", "stratification_fallback",
            "Stratified split raised ValueError; retried without stratification",
            {"class_counts": dict(counts)},
        )
        train_df, val_df = train_test_split(
            df, test_size=val_frac, random_state=seed, stratify=None, shuffle=True
        )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), stratified


def split_report(seed: int, train_df, val_df, stratified: bool) -> dict:
    labels = sorted(set(train_df["label"]).union(set(val_df["label"])))
    per_label = {}
    tr_counts = Counter(train_df["label"])
    va_counts = Counter(val_df["label"])
    for lab in labels:
        per_label[lab] = {"train": int(tr_counts.get(lab, 0)),
                          "validation": int(va_counts.get(lab, 0))}
    return {
        "random_seed": seed,
        "stratified": stratified,
        "train_size": int(len(train_df)),
        "validation_size": int(len(val_df)),
        "labels": per_label,
    }
