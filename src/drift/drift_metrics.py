"""
drift_metrics.py
==================
Quantifies (a) how far a window's SHAP/LIME feature ranking has moved from
the baseline window's ranking, and (b) whether that "explanation drift"
signal leads or lags the "performance decay" signal (AUC/F1 dropping), with
a permutation test for statistical significance.
"""

import logging
from typing import Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr, wasserstein_distance

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Ranking-drift metrics (SHAP or LIME importance vectors, window vs baseline)
# --------------------------------------------------------------------------
def align(a: pd.Series, b: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    idx = a.index.union(b.index)
    return a.reindex(idx, fill_value=0.0).values, b.reindex(idx, fill_value=0.0).values


def spearman_drift(baseline: pd.Series, current: pd.Series) -> float:
    """1 - Spearman rho -> 0 means identical ranking, 2 means perfectly
    reversed. We report both the raw rho and this drift score in results."""
    a, b = align(baseline, current)
    rho, _ = spearmanr(a, b)
    return 0.0 if np.isnan(rho) else rho


def kendall_drift(baseline: pd.Series, current: pd.Series) -> float:
    a, b = align(baseline, current)
    tau, _ = kendalltau(a, b)
    return 0.0 if np.isnan(tau) else tau


def topk_jaccard(baseline: pd.Series, current: pd.Series, k: int = None) -> float:
    k = k or config.RANKING_TOP_K
    top_a = set(baseline.sort_values(ascending=False).head(k).index)
    top_b = set(current.sort_values(ascending=False).head(k).index)
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / len(top_a | top_b)


def feature_distribution_shift(
    X_baseline: pd.DataFrame, X_current: pd.DataFrame, numeric_cols: Sequence[str]
) -> pd.Series:
    """Per-feature Wasserstein (earth-mover) distance between a window's raw
    feature distribution and the baseline window's -- the "global background
    shift" SHAP's own background-distribution assumption is sensitive to,
    independent of the model entirely."""
    out = {}
    for c in numeric_cols:
        try:
            out[c] = wasserstein_distance(X_baseline[c].values, X_current[c].values)
        except Exception:  # pragma: no cover - defensive
            out[c] = np.nan
    return pd.Series(out)


def ranking_drift_report(baseline: pd.Series, current: pd.Series, top_k: int = None) -> Dict[str, float]:
    rho = spearman_drift(baseline, current)
    tau = kendall_drift(baseline, current)
    jac = topk_jaccard(baseline, current, top_k)
    return dict(
        spearman_rho=rho,
        spearman_drift=1.0 - rho,
        kendall_tau=tau,
        kendall_drift=1.0 - tau,
        topk_jaccard=jac,
        topk_jaccard_drift=1.0 - jac,
    )


# --------------------------------------------------------------------------
# Lead-lag / early-warning analysis
# --------------------------------------------------------------------------
def _zscore(x: np.ndarray) -> np.ndarray:
    s = np.std(x)
    return (x - np.mean(x)) / s if s > 1e-12 else np.zeros_like(x)


def cross_correlation(signal_a: np.ndarray, signal_b: np.ndarray, max_lag: int) -> pd.Series:
    """corr(a[t], b[t+lag]) for lag in [-max_lag, max_lag].
    Positive lag > 0 means `b` at a LATER window correlates with `a` now,
    i.e. `a` (typically the explanation-drift signal) LEADS `b` (typically
    the performance-drop signal) by `lag` windows.
    """
    a = _zscore(np.asarray(signal_a, dtype=float))
    b = _zscore(np.asarray(signal_b, dtype=float))
    n = len(a)
    out = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            aa, bb = a[: n - lag] if lag > 0 else a, b[lag:]
        else:
            aa, bb = a[-lag:], b[: n + lag]
        if len(aa) < 3:
            out[lag] = np.nan
            continue
        if np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
            out[lag] = np.nan
            continue
        out[lag] = float(np.corrcoef(aa, bb)[0, 1])
    return pd.Series(out)


def best_lead_lag(signal_a: np.ndarray, signal_b: np.ndarray, max_lag: int = None) -> Tuple[int, float]:
    max_lag = max_lag or config.MAX_LAG_WINDOWS
    ccf = cross_correlation(signal_a, signal_b, max_lag)
    ccf = ccf.dropna()
    if ccf.empty:
        return 0, np.nan
    best_lag = ccf.idxmax()
    return int(best_lag), float(ccf.loc[best_lag])


def permutation_test_lead_lag(
    signal_a: np.ndarray, signal_b: np.ndarray, lag: int, n_permutations: int = None, seed: int = 0
) -> float:
    """One-sided permutation test: is the observed correlation at `lag`
    stronger than what we'd see if the explanation-drift signal carried no
    real relationship to the (shifted) performance signal? We permute
    signal_a in place (destroying its temporal structure) and recompute the
    correlation at the fixed lag, `n_permutations` times, to build a null
    distribution.
    """
    n_permutations = n_permutations or config.N_PERMUTATIONS
    rng = np.random.default_rng(seed)

    a = _zscore(np.asarray(signal_a, dtype=float))
    b = _zscore(np.asarray(signal_b, dtype=float))
    n = len(a)

    if lag >= 0:
        aa, bb = (a[: n - lag] if lag > 0 else a), b[lag:]
    else:
        aa, bb = a[-lag:], b[: n + lag]

    if len(aa) < 3 or np.std(aa) < 1e-9 or np.std(bb) < 1e-9:
        return np.nan

    observed = np.corrcoef(aa, bb)[0, 1]

    null_corrs = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(aa)
        null_corrs[i] = np.corrcoef(perm, bb)[0, 1]

    p_value = float(np.mean(np.abs(null_corrs) >= np.abs(observed)))
    return p_value


def first_crossing_index(signal: np.ndarray, threshold: float, direction: str = "above") -> int:
    """Index of the first window where `signal` crosses `threshold`.
    Returns -1 if it never crosses (treated as censored in aggregation)."""
    arr = np.asarray(signal, dtype=float)
    if direction == "above":
        hits = np.where(arr >= threshold)[0]
    else:
        hits = np.where(arr <= threshold)[0]
    return int(hits[0]) if len(hits) else -1


def relative_crossing_index(signal: np.ndarray, fraction: float) -> int:
    """Index of the first window where `signal` first reaches `fraction` of
    its own observed peak value.

    This assumes `signal` starts at (or near) 0 at the baseline window --
    true by construction for every drift/shift signal in this study, since
    each is a comparison against that same baseline. Unlike
    `first_crossing_index`, the threshold is derived from the signal's own
    range rather than a hand-picked absolute number, which matters here for
    two reasons: (1) it removes the "floor effect" where an absolute
    performance threshold could be crossed at the earliest possible window
    for every seed regardless of the true dynamics, and (2) it lets
    differently-scaled signals (SHAP drift, LIME drift, raw Wasserstein
    distance) be compared on equal footing without guessing a separate
    absolute cutoff for each.

    Returns -1 if the signal never moves (peak <= 0), matching
    `first_crossing_index`'s censoring convention.
    """
    arr = np.asarray(signal, dtype=float)
    peak = np.nanmax(arr) if arr.size else np.nan
    if not np.isfinite(peak) or peak <= 0:
        return -1
    threshold = fraction * peak
    hits = np.where(arr >= threshold)[0]
    return int(hits[0]) if len(hits) else -1


def differenced(signal: np.ndarray) -> np.ndarray:
    """First difference of `signal` (length n -> n-1).

    Two curves that both simply "jump early, then plateau" will show high,
    and permutation-test-'significant', cross-correlation at many lags
    purely because they share that trend/level shape -- not because they
    are dynamically coupled period-by-period. Differencing removes the
    shared level/trend component before cross-correlating, which is a
    standard time-series diagnostic for this exact failure mode and is used
    as the more trustworthy companion to the raw-level cross-correlation
    throughout `analysis.py`.
    """
    return np.diff(np.asarray(signal, dtype=float))
