"""
explainers.py
==============
Instrumentation for:
  (1) SHAP  -- global feature-attribution ranking per window
              (shap.TreeExplainer -> exact Shapley values for the LightGBM
               ensemble; "global background shift" = how this ranking moves
               across windows).
  (2) LIME  -- local, per-instance explanation + PERTURBATION STABILITY.
              Because LIME re-fits a local surrogate on a fresh random
              perturbation sample every call, running it multiple times on
              the *same* instance gives a direct, model-faithfulness-free
              measure of explanation stability (does the explanation change
              just because the random seed changed?), separate from
              temporal drift (does the explanation change because the
              underlying data window changed?).

Design note on the "reduced feature surface" for LIME
-------------------------------------------------------
IEEE-CIS has ~430 raw features after merging identity+transaction. Running
LIME (which fits a fresh weighted linear model per explanation) over the
full feature space is both slow and produces very noisy, hard-to-interpret
per-feature weights. Following common practice in the explanation-stability
literature, we restrict the LIME *perturbation surface* to the top-K
globally important features (by baseline-window SHAP ranking, see
config.TOP_K_FEATURES_FOR_LIME). All OTHER features are held fixed at the
explained instance's true observed values when we query the real model --
i.e. we are not simplifying the model, only the region of the input space
LIME is allowed to perturb when building its local surrogate. This is
implemented via `_make_reduced_predict_fn` below.
"""

import logging
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from joblib import Parallel, delayed
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr

from src import config

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# SHAP
# --------------------------------------------------------------------------
def _extract_positive_class_shap(shap_output) -> np.ndarray:
    """Normalize across shap-library versions/output shapes to a single
    (n_samples, n_features) array of SHAP values for the fraud (positive)
    class."""
    if isinstance(shap_output, list):
        # legacy API: [class0_values, class1_values]
        return np.asarray(shap_output[1])
    arr = np.asarray(shap_output)
    if arr.ndim == 3:
        # (n_samples, n_features, n_classes)
        return arr[:, :, -1]
    return arr  # already (n_samples, n_features)


def compute_shap_global_importance(
    model, X: pd.DataFrame, seed: int, subsample_size: int = None
) -> pd.Series:
    """Mean(|SHAP value|) per feature over a (sub-sampled) window.

    Returns a pd.Series indexed by feature name -- the "global ranking" this
    paper tracks for drift, and equivalently the vector fed into Spearman/
    Kendall/Jaccard drift metrics in drift_metrics.py.
    """
    subsample_size = subsample_size or config.SHAP_SUBSAMPLE_SIZE
    rng = np.random.default_rng(seed)

    if len(X) > subsample_size:
        idx = rng.choice(len(X), size=subsample_size, replace=False)
        X_sub = X.iloc[idx]
    else:
        X_sub = X

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X_sub)
    pos_shap = _extract_positive_class_shap(raw)
    importance = np.abs(pos_shap).mean(axis=0)
    return pd.Series(importance, index=X.columns)


# --------------------------------------------------------------------------
# LIME
# --------------------------------------------------------------------------
def _make_reduced_predict_fn(model, full_columns: Sequence[str], reduced_cols: Sequence[str],
                              categorical_cols_full: Sequence[str], base_row: pd.Series):
    """Build a predict_proba function whose *input* is only the reduced
    feature set, but which queries the REAL, full model by holding every
    other feature fixed at `base_row`'s observed value.
    """
    categorical_set = set(categorical_cols_full)
    base_values = base_row[full_columns].values

    def predict_fn(perturbed_reduced: np.ndarray) -> np.ndarray:
        n = perturbed_reduced.shape[0]
        full = np.tile(base_values, (n, 1))
        full_df = pd.DataFrame(full, columns=full_columns)
        for j, col in enumerate(reduced_cols):
            full_df[col] = perturbed_reduced[:, j]
        for c in categorical_set:
            full_df[c] = full_df[c].astype(float).round().astype(int)
        for c in full_columns:
            if c not in categorical_set:
                full_df[c] = full_df[c].astype(float)
        return model.predict_proba(full_df)

    return predict_fn


def _explain_one_instance(
    position: int,
    X_full: pd.DataFrame,
    X_reduced_values: np.ndarray,
    reduced_cols: List[str],
    categorical_idx_reduced: List[int],
    categorical_cols_full: List[str],
    model,
    n_repeats: int,
    num_samples: int,
    base_seed: int,
) -> np.ndarray:
    """Run `n_repeats` independent LIME fits on ONE instance.

    Returns an (n_repeats, n_reduced_features) matrix of local linear
    weights (dense: 0.0 where a feature wasn't selected -- with
    num_features == n_reduced_features every feature is always included).
    """
    # This function runs inside a joblib worker PROCESS (see
    # compute_lime_explanations, prefer="processes"). `model` was trained
    # with LGBM_PARAMS["n_jobs"] == config.N_JOBS (e.g. 14) baked in, so
    # without this line every one of the (up to) config.N_JOBS worker
    # processes would ALSO try to spin up config.N_JOBS LightGBM prediction
    # threads -- oversubscribing the machine by up to N_JOBS^2 threads and
    # making the parallel LIME loop slower than running it serially. Each
    # worker process has its own pickled copy of `model` (process-based
    # backend), so mutating it here is safe and does not affect the
    # original model object in the parent process or in sibling workers.
    model.set_params(n_jobs=1)

    base_row = X_full.iloc[position]
    reduced_row = X_reduced_values[position]

    predict_fn = _make_reduced_predict_fn(
        model, list(X_full.columns), reduced_cols, categorical_cols_full, base_row
    )

    weights = np.zeros((n_repeats, len(reduced_cols)))
    for r in range(n_repeats):
        # distinct-but-reproducible RNG per (instance, repeat) so repeats are
        # genuinely independent perturbation draws (this independence is
        # exactly what "perturbation stability" measures).
        explainer_seed = (base_seed * 1_000_003 + position * 97 + r) % (2**31 - 1)
        explainer = LimeTabularExplainer(
            training_data=X_reduced_values,
            feature_names=reduced_cols,
            categorical_features=categorical_idx_reduced,
            class_names=["legit", "fraud"],
            discretize_continuous=True,
            mode="classification",
            random_state=explainer_seed,
        )
        exp = explainer.explain_instance(
            reduced_row,
            predict_fn,
            num_features=len(reduced_cols),
            num_samples=num_samples,
            labels=(1,),
        )
        wmap = dict(exp.as_map()[1])
        weights[r] = np.array([wmap.get(i, 0.0) for i in range(len(reduced_cols))])
    return weights


def compute_lime_explanations(
    model,
    X_full: pd.DataFrame,
    reduced_cols: List[str],
    categorical_cols_full: List[str],
    seed: int,
    n_instances: int = None,
    n_repeats: int = None,
    num_samples: int = None,
    n_jobs: int = None,
) -> Dict[str, object]:
    """Sample instances from a window and compute both:
       (a) a global LIME importance vector (mean |weight| over all
           instances & repeats) -- comparable across windows for drift, and
       (b) local perturbation-stability scores (Spearman / top-k Jaccard
           agreement across the `n_repeats` independent re-explanations of
           the SAME instance).

    Returns a dict with keys:
       'importance'         : pd.Series indexed by reduced_cols
       'stability_spearman'  : float, mean across instances
       'stability_spearman_std': float
       'stability_jaccard'   : float, mean across instances
       'stability_jaccard_std': float
       'n_instances_used'    : int
    """
    n_instances = n_instances or config.LIME_N_INSTANCES_PER_WINDOW
    n_repeats = n_repeats or config.LIME_N_REPEATS_FOR_STABILITY
    num_samples = num_samples or config.LIME_NUM_SAMPLES
    n_jobs = n_jobs or config.N_JOBS
    top_k = min(config.RANKING_TOP_K, len(reduced_cols))

    rng = np.random.default_rng(seed)
    n_avail = len(X_full)
    n_pick = min(n_instances, n_avail)
    positions = rng.choice(n_avail, size=n_pick, replace=False)

    X_reduced_values = X_full[reduced_cols].values.astype(float)
    categorical_idx_reduced = [i for i, c in enumerate(reduced_cols) if c in categorical_cols_full]

    results = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(_explain_one_instance)(
            int(pos), X_full, X_reduced_values, reduced_cols, categorical_idx_reduced,
            categorical_cols_full, model, n_repeats, num_samples, seed,
        )
        for pos in positions
    )

    all_importances = []
    spearman_scores = []
    jaccard_scores = []

    for weight_matrix in results:  # weight_matrix: (n_repeats, n_features)
        all_importances.append(np.abs(weight_matrix).mean(axis=0))

        pair_spearman = []
        pair_jaccard = []
        for a, b in combinations(range(n_repeats), 2):
            wa, wb = weight_matrix[a], weight_matrix[b]
            if np.allclose(wa, wa[0]) or np.allclose(wb, wb[0]):
                rho = 1.0 if np.allclose(wa, wb) else 0.0
            else:
                rho, _ = spearmanr(wa, wb)
                rho = 0.0 if np.isnan(rho) else rho
            pair_spearman.append(rho)

            top_a = set(np.argsort(-np.abs(wa))[:top_k])
            top_b = set(np.argsort(-np.abs(wb))[:top_k])
            pair_jaccard.append(len(top_a & top_b) / len(top_a | top_b))

        spearman_scores.append(np.mean(pair_spearman))
        jaccard_scores.append(np.mean(pair_jaccard))

    importance_series = pd.Series(np.mean(all_importances, axis=0), index=reduced_cols)

    return dict(
        importance=importance_series,
        stability_spearman=float(np.mean(spearman_scores)),
        stability_spearman_std=float(np.std(spearman_scores)),
        stability_jaccard=float(np.mean(jaccard_scores)),
        stability_jaccard_std=float(np.std(jaccard_scores)),
        n_instances_used=n_pick,
    )
