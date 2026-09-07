"""
experiment_runner.py
======================
Orchestrates the full study:

    for each seed in config.SEEDS:
        train a "frozen" model on the baseline (earliest) window
        for each sliding window t (chronological order):
            evaluate the frozen model out-of-window        -> performance decay
            compute SHAP global ranking for the frozen model -> drift vs baseline
            compute LIME local explanations + stability      -> drift + stability vs baseline
            [control] retrain a fresh model on window t itself and repeat
                      the SHAP/LIME/perf measurements for it too

Every row of the resulting table is tagged with (seed, regime, window_index),
so downstream analysis can both average over seeds (for confidence
intervals) and compare the "frozen" vs "retrained" regimes (to separate
model-staleness-driven explanation drift from pure data-distribution drift).
"""

import argparse
import logging
import time
from typing import List

import numpy as np
import pandas as pd

from src import config
from src.data import data_loader
from src.data import feature_engineering as fe
from src.drift import drift_metrics
from src.models import explainers, model_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _flatten_row(prefix: str, d: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in d.items() if k != "importance"}


def run_single_seed(seed: int, spec: fe.FeatureSpec, windows: List[data_loader.Window],
                     n_jobs: int = None) -> pd.DataFrame:
    n_jobs = n_jobs if n_jobs is not None else config.N_JOBS
    t0 = time.time()
    logger.info("=== Seed %d: starting (n_jobs=%d) ===", seed, n_jobs)
    np.random.seed(seed)

    base_window = windows[config.BASELINE_WINDOW_INDEX]
    base_train_df, base_test_df = data_loader.split_train_test_within_window(base_window.df)
    X_train_base, y_train_base = fe.transform(base_train_df, spec)
    X_test_base, y_test_base = fe.transform(base_test_df, spec)

    frozen_model = model_utils.train_model(X_train_base, y_train_base, seed, spec.categorical_cols, n_jobs=n_jobs)
    frozen_threshold = model_utils.find_best_threshold(frozen_model, X_train_base, y_train_base)

    baseline_metrics = model_utils.evaluate_model(frozen_model, X_test_base, y_test_base, frozen_threshold)
    baseline_shap = explainers.compute_shap_global_importance(frozen_model, X_test_base, seed)
    top_features = (
        baseline_shap.sort_values(ascending=False).head(config.TOP_K_FEATURES_FOR_LIME).index.tolist()
    )
    baseline_lime = explainers.compute_lime_explanations(
        frozen_model, X_test_base, top_features, spec.categorical_cols, seed, n_jobs=n_jobs
    )
    logger.info("Seed %d: baseline window ready (n_test=%d, roc_auc=%.4f, top LIME feature=%s)",
                seed, len(y_test_base), baseline_metrics["roc_auc"], top_features[0])

    rows = []
    for w in windows:
        train_df, test_df = data_loader.split_train_test_within_window(w.df)
        X_train, y_train = fe.transform(train_df, spec)
        X_test, y_test = fe.transform(test_df, spec)

        wshift = drift_metrics.feature_distribution_shift(X_test_base, X_test, spec.numeric_cols)
        mean_wshift = float(np.nanmean(wshift.values))

        common = dict(
            seed=seed, window=w.index, start_day=w.start_day, end_day=w.end_day,
            n_test_rows=len(y_test), n_test_positives=int(y_test.sum()),
            mean_feature_wasserstein_shift=mean_wshift,
        )

        # ---------------- FROZEN regime ----------------
        # threshold is the ONE selected at baseline-training time and never
        # re-tuned -- this is what makes precision/recall/F1 decay a
        # meaningful, non-leaked signal of drift over time.
        m_frozen = model_utils.evaluate_model(frozen_model, X_test, y_test, frozen_threshold)
        shap_frozen = explainers.compute_shap_global_importance(frozen_model, X_test, seed)
        lime_frozen = explainers.compute_lime_explanations(
            frozen_model, X_test, top_features, spec.categorical_cols, seed, n_jobs=n_jobs
        )
        drift_shap_frozen = drift_metrics.ranking_drift_report(baseline_shap, shap_frozen)
        drift_lime_frozen = drift_metrics.ranking_drift_report(baseline_lime["importance"], lime_frozen["importance"])

        row_frozen = dict(common, regime="frozen")
        row_frozen.update(_flatten_row("perf", m_frozen))
        row_frozen.update(_flatten_row("shap", drift_shap_frozen))
        row_frozen.update(_flatten_row("lime", drift_lime_frozen))
        row_frozen["lime_stability_spearman"] = lime_frozen["stability_spearman"]
        row_frozen["lime_stability_spearman_std"] = lime_frozen["stability_spearman_std"]
        row_frozen["lime_stability_jaccard"] = lime_frozen["stability_jaccard"]
        row_frozen["lime_stability_jaccard_std"] = lime_frozen["stability_jaccard_std"]
        rows.append(row_frozen)

        # ---------------- RETRAINED regime (control) ----------------
        if w.index == config.BASELINE_WINDOW_INDEX:
            m_retrained, shap_retrained, lime_retrained = baseline_metrics, baseline_shap, baseline_lime
        else:
            retrained_model = model_utils.train_model(X_train, y_train, seed, spec.categorical_cols, n_jobs=n_jobs)
            retrained_threshold = model_utils.find_best_threshold(retrained_model, X_train, y_train)
            m_retrained = model_utils.evaluate_model(retrained_model, X_test, y_test, retrained_threshold)
            shap_retrained = explainers.compute_shap_global_importance(retrained_model, X_test, seed)
            lime_retrained = explainers.compute_lime_explanations(
                retrained_model, X_test, top_features, spec.categorical_cols, seed, n_jobs=n_jobs
            )

        drift_shap_retrained = drift_metrics.ranking_drift_report(baseline_shap, shap_retrained)
        drift_lime_retrained = drift_metrics.ranking_drift_report(baseline_lime["importance"], lime_retrained["importance"])

        row_retrained = dict(common, regime="retrained")
        row_retrained.update(_flatten_row("perf", m_retrained))
        row_retrained.update(_flatten_row("shap", drift_shap_retrained))
        row_retrained.update(_flatten_row("lime", drift_lime_retrained))
        row_retrained["lime_stability_spearman"] = lime_retrained["stability_spearman"]
        row_retrained["lime_stability_spearman_std"] = lime_retrained["stability_spearman_std"]
        row_retrained["lime_stability_jaccard"] = lime_retrained["stability_jaccard"]
        row_retrained["lime_stability_jaccard_std"] = lime_retrained["stability_jaccard_std"]
        rows.append(row_retrained)

        logger.info(
            "Seed %d | window %2d [%5.1f-%5.1fd] | frozen AUC=%.4f (drop=%.4f) shap_drift=%.3f lime_drift=%.3f "
            "| retrained AUC=%.4f",
            seed, w.index, w.start_day, w.end_day, m_frozen["roc_auc"],
            baseline_metrics["roc_auc"] - m_frozen["roc_auc"], drift_shap_frozen["spearman_drift"],
            drift_lime_frozen["spearman_drift"], m_retrained["roc_auc"],
        )

    df = pd.DataFrame(rows)
    logger.info("=== Seed %d: done in %.1fs (%d rows) ===", seed, time.time() - t0, len(df))

    seed_path = f"{config.TABLES_DIR}/results_seed_{seed}.parquet"
    df.to_parquet(seed_path, index=False)
    logger.info("Saved incremental results -> %s", seed_path)
    return df


def run_all_seeds(seeds: List[int] = None) -> pd.DataFrame:
    """Runs every seed and combines the results.

    Parallelization strategy
    -------------------------
    Seeds are fully independent of each other (each trains its own baseline
    model from scratch), which makes them the natural unit of parallel work
    -- unlike LightGBM's own internal thread-parallelism (bounded by
    diminishing returns from synchronization overhead) or LIME's per-
    instance parallelism (bounded by the number of instances per window).
    With `config.SEEDS` sized close to a modern laptop's core count (12 by
    default), running seeds concurrently keeps ALL cores busy through EVERY
    phase of the pipeline (model training, SHAP, and LIME alike), whereas
    only parallelizing the LIME inner loop leaves cores idle during model
    training and SHAP.

    The available core budget (`config.N_JOBS`) is split between "how many
    seeds run at once" and "how many threads/processes each seed gets
    internally" so the TOTAL concurrency never oversubscribes the machine:
        n_seed_workers = min(len(seeds), config.N_JOBS)
        inner_n_jobs   = max(1, config.N_JOBS // n_seed_workers)
    e.g. 12 seeds on a 12-core machine -> 12 seed-workers x 1 thread each
    (the common case for this study). 3 seeds on a 12-core machine -> 3
    seed-workers x 4 threads each.

    Memory note: each seed-worker is a separate process (joblib "loky"
    backend) and receives its own copy of `windows` (the full windowed
    dataset). This trades higher peak memory for much better core
    utilization; if you are memory-constrained, pass fewer seeds per
    `run_all_seeds` call (results are saved incrementally per seed and
    concatenated across calls) or set `config.N_JOBS = 1` to fall back to
    the fully sequential path below.
    """
    seeds = seeds or config.SEEDS
    if len(seeds) < 10:
        logger.warning("Fewer than 10 seeds requested (%d); the study design requires >= 10.", len(seeds))

    df_raw = data_loader.load_raw_data()
    spec = fe.build_feature_spec(df_raw)
    windows = data_loader.create_sliding_windows(df_raw)

    n_seed_workers = min(len(seeds), config.N_JOBS)
    inner_n_jobs = max(1, config.N_JOBS // n_seed_workers)

    if n_seed_workers > 1:
        logger.info("Running %d seeds across %d parallel worker(s), %d thread(s)/process(es) each.",
                    len(seeds), n_seed_workers, inner_n_jobs)
        from joblib import Parallel, delayed
        all_frames = Parallel(n_jobs=n_seed_workers, backend="loky")(
            delayed(run_single_seed)(seed, spec, windows, inner_n_jobs) for seed in seeds
        )
    else:
        logger.info("Running %d seed(s) sequentially (n_jobs=%d each).", len(seeds), inner_n_jobs)
        all_frames = [run_single_seed(seed, spec, windows, inner_n_jobs) for seed in seeds]

    combined = pd.concat(all_frames, ignore_index=True)
    combined_path = f"{config.TABLES_DIR}/results_all_seeds.parquet"
    combined.to_parquet(combined_path, index=False)
    combined.to_csv(f"{config.TABLES_DIR}/results_all_seeds.csv", index=False)
    logger.info("Saved combined results (%d rows) -> %s", len(combined), combined_path)
    return combined


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                         help="Override config.SEEDS, e.g. --seeds 0 1 2")
    parser.add_argument("--transaction-file", type=str, default=None)
    parser.add_argument("--identity-file", type=str, default=None)
    args = parser.parse_args()

    if args.transaction_file:
        config.TRANSACTION_FILE = args.transaction_file
    if args.identity_file:
        config.IDENTITY_FILE = args.identity_file

    run_all_seeds(args.seeds)


if __name__ == "__main__":
    main()
