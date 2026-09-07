"""
model_utils.py
================
LightGBM training/evaluation helpers. LightGBM is chosen because:
  (1) it natively handles categorical features without one-hot blow-up,
      which matters for high-cardinality IEEE-CIS columns like card1/addr1;
  (2) it is CPU-multithreaded and trains in seconds-to-minutes per window on
      a 14-core machine with no GPU required (the Apple GPU is not used --
      see config.py);
  (3) shap.TreeExplainer computes EXACT Shapley values for tree ensembles in
      polynomial time, avoiding the sampling noise of KernelSHAP.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                              precision_score, recall_score, roc_auc_score)

from src import config

logger = logging.getLogger(__name__)


def train_model(X: pd.DataFrame, y: pd.Series, seed: int, categorical_cols, n_jobs: int = None) -> LGBMClassifier:
    params = dict(config.LGBM_PARAMS)
    params["random_state"] = seed
    if n_jobs is not None:
        params["n_jobs"] = n_jobs
    model = LGBMClassifier(**params)
    model.fit(X, y, categorical_feature=list(categorical_cols))
    return model


def find_best_threshold(model: LGBMClassifier, X: pd.DataFrame, y: pd.Series) -> float:
    """F1-maximizing decision threshold, selected ONLY from the model's own
    training split (never from the evaluation window) so the resulting F1
    scores reported per window are not test-set-leaked. This threshold is
    then held fixed -- exactly mirroring how a real fraud system freezes a
    review-queue threshold rather than re-tuning it against ground truth it
    won't see until much later (if ever). Falls back to
    config.CLASSIFICATION_THRESHOLD if the training split has too few
    positives to estimate a stable threshold.
    """
    if y.nunique() < 2 or y.sum() < 10:
        return config.CLASSIFICATION_THRESHOLD
    proba = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, proba)
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = np.argmax(f1s[:-1]) if len(thresholds) else None
    if best_idx is None:
        return config.CLASSIFICATION_THRESHOLD
    return float(thresholds[best_idx])


def evaluate_model(model: LGBMClassifier, X: pd.DataFrame, y: pd.Series,
                    threshold: float = None) -> Dict[str, float]:
    threshold = threshold if threshold is not None else config.CLASSIFICATION_THRESHOLD

    if y.nunique() < 2:
        logger.warning("Window has a single class present; AUC/PR-AUC undefined -> NaN.")
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= threshold).astype(int)
        return dict(
            roc_auc=np.nan, pr_auc=np.nan,
            f1=f1_score(y, pred, zero_division=0),
            precision=precision_score(y, pred, zero_division=0),
            recall=recall_score(y, pred, zero_division=0),
            n_rows=len(y), n_positives=int(y.sum()),
        )

    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    return dict(
        roc_auc=roc_auc_score(y, proba),
        pr_auc=average_precision_score(y, proba),
        f1=f1_score(y, pred, zero_division=0),
        precision=precision_score(y, pred, zero_division=0),
        recall=recall_score(y, pred, zero_division=0),
        n_rows=len(y),
        n_positives=int(y.sum()),
    )
