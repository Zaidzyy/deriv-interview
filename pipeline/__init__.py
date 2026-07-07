"""Deterministic scikit-learn text-classification pipeline package.

Training code lives here and in ``main.py``. Inference lives in ``predict.py``.
Both training and inference import :func:`pipeline.preprocessing.preprocess_text`
so there is a single, shared preprocessing definition (no train/serve skew).
"""
