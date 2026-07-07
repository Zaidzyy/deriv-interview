"""FEATURES_FIT stage: build + fit the TF-IDF vectorizer from config."""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer(cfg: dict) -> TfidfVectorizer:
    vcfg = cfg.get("vectorizer", {})
    vtype = vcfg.get("type", "tfidf")
    if vtype != "tfidf":
        raise ValueError(f"Unsupported vectorizer type: {vtype!r} (only 'tfidf' supported)")
    ngram = vcfg.get("ngram_range", [1, 2])
    return TfidfVectorizer(
        ngram_range=tuple(ngram),
        max_features=vcfg.get("max_features", 5000),
        min_df=vcfg.get("min_df", 1),
        # Text is already lowercased/normalised by the shared preprocessing
        # function; keep the vectorizer's own preprocessing off to avoid a
        # second, divergent transform.
        lowercase=False,
    )
