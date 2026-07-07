"""The ONE shared text preprocessing function.

Imported by BOTH training (``main.py``/pipeline) and inference (``predict.py``).
Keeping a single definition here is what prevents train/serve skew — the
evaluator explicitly checks that preprocessing is applied consistently.

Deterministic: same input string always yields the same output. No randomness,
no external state.
"""
from __future__ import annotations

import re

_WHITESPACE = re.compile(r"\s+")


def preprocess_text(text: object) -> str:
    """Lowercase, trim, and collapse internal whitespace.

    Light on purpose — aggressive cleaning (stopword/punctuation stripping) would
    remove signal on short texts. Non-string / missing values coerce to "".
    """
    if text is None:
        return ""
    s = str(text)
    s = s.strip().lower()
    s = _WHITESPACE.sub(" ", s)
    return s
