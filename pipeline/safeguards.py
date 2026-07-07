"""Practical evaluation safeguards (findings collected into safeguards_report.json).

Checks:
  - high class imbalance in the training data
  - any class missing from the validation split
  - exact duplicate texts shared between train and validation splits
  - (stratification-fallback warnings are appended by the split step)
"""
from __future__ import annotations

from collections import Counter


class Safeguards:
    def __init__(self):
        self.findings: list[dict] = []

    def add(self, level: str, check: str, message: str, detail=None) -> None:
        self.findings.append(
            {"level": level, "check": check, "message": message, "detail": detail}
        )

    def check_class_imbalance(self, labels, ratio_warn: float) -> None:
        counts = Counter(labels)
        if not counts:
            return
        hi, lo = max(counts.values()), min(counts.values())
        ratio = hi / lo if lo else float("inf")
        detail = {"counts": dict(counts), "max_min_ratio": round(ratio, 3),
                  "threshold": ratio_warn}
        if ratio >= ratio_warn:
            self.add("warning", "class_imbalance",
                     f"High class imbalance: max/min ratio {ratio:.2f} >= {ratio_warn}",
                     detail)
        else:
            self.add("ok", "class_imbalance",
                     f"Class balance acceptable (ratio {ratio:.2f} < {ratio_warn})",
                     detail)

    def check_missing_val_classes(self, all_labels, val_labels) -> None:
        missing = sorted(set(all_labels) - set(val_labels))
        if missing:
            self.add("warning", "missing_val_classes",
                     f"Class(es) absent from validation split: {missing}",
                     {"missing": missing})
        else:
            self.add("ok", "missing_val_classes",
                     "All classes present in validation split", None)

    def check_train_val_duplicates(self, train_texts, val_texts) -> None:
        train_set = set(train_texts)
        dupes = sorted({t for t in val_texts if t in train_set})
        if dupes:
            self.add("warning", "train_val_duplicates",
                     f"{len(dupes)} exact duplicate text(s) shared between train and validation",
                     {"count": len(dupes), "examples": dupes[:5]})
        else:
            self.add("ok", "train_val_duplicates",
                     "No exact train/validation text duplicates", None)

    def to_report(self) -> dict:
        levels = Counter(f["level"] for f in self.findings)
        return {
            "n_findings": len(self.findings),
            "n_warnings": levels.get("warning", 0),
            "findings": self.findings,
        }
