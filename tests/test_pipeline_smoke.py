"""
test_pipeline_smoke.py
========================
An end-to-end smoke test of the full pipeline (data -> features -> model ->
SHAP -> LIME -> drift metrics -> lead-lag analysis) using the synthetic,
schema-matched dataset from make_synthetic_data.py. It runs with small
parameters purely to finish quickly; it is a correctness/wiring check, NOT
a validation of any research finding.

Run with:  python -m pytest tests/test_pipeline_smoke.py -v
       or:  python tests/test_pipeline_smoke.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd


def _configure_for_speed():
    from src import config
    config.SEEDS = [0, 1]
    config.WINDOW_SIZE_DAYS = 20
    config.STEP_SIZE_DAYS = 20
    config.MIN_WINDOW_ROWS = 200
    config.MIN_WINDOW_POSITIVES = 3
    config.LIME_N_INSTANCES_PER_WINDOW = 5
    config.LIME_N_REPEATS_FOR_STABILITY = 2
    config.LIME_NUM_SAMPLES = 200
    config.SHAP_SUBSAMPLE_SIZE = 200
    config.TOP_K_FEATURES_FOR_LIME = 10
    config.N_PERMUTATIONS = 100
    config.N_JOBS = 1
    config.LGBM_PARAMS["n_estimators"] = 60
    config.LGBM_PARAMS["n_jobs"] = 1
    return config


def test_full_pipeline_runs_on_synthetic_data(tmp_path=None):
    config = _configure_for_speed()

    from scripts.make_synthetic_data import generate_synthetic_ieee_cis
    tx, identity = generate_synthetic_ieee_cis(n_rows=20000, n_days=182, seed=42)

    tx_path = os.path.join(config.DATA_DIR, "TEST_transaction.csv")
    id_path = os.path.join(config.DATA_DIR, "TEST_identity.csv")
    tx.to_csv(tx_path, index=False)
    identity.to_csv(id_path, index=False)
    config.TRANSACTION_FILE = tx_path
    config.IDENTITY_FILE = id_path

    from src.data import data_loader
    from src.data import feature_engineering as fe
    from src.drift import drift_metrics
    from src.models import explainers, model_utils
    from src.pipeline import analysis, experiment_runner

    df = data_loader.load_raw_data()
    assert df["isFraud"].isin([0, 1]).all()
    assert (df["TransactionDT"].diff().dropna() >= 0).all(), "data must remain time-sorted"

    windows = data_loader.create_sliding_windows(df)
    assert len(windows) >= 3

    spec = fe.build_feature_spec(df)
    assert len(spec.numeric_cols) > 0 and len(spec.categorical_cols) > 0

    train_df, test_df = data_loader.split_train_test_within_window(windows[0].df)
    X_train, y_train = fe.transform(train_df, spec)
    X_test, y_test = fe.transform(test_df, spec)
    assert list(X_train.columns) == list(X_test.columns)

    model = model_utils.train_model(X_train, y_train, seed=0, categorical_cols=spec.categorical_cols)
    metrics = model_utils.evaluate_model(model, X_test, y_test)
    assert 0.0 <= metrics["roc_auc"] <= 1.0

    # DataFrame vs ndarray predictions must match (required for LIME's predict_fn contract)
    p1 = model.predict_proba(X_test)[:20]
    p2 = model.predict_proba(X_test.values)[:20]
    assert np.allclose(p1, p2)

    shap_imp = explainers.compute_shap_global_importance(model, X_test, seed=0)
    assert (shap_imp >= 0).all()
    top_feats = shap_imp.sort_values(ascending=False).head(10).index.tolist()

    lime_res = explainers.compute_lime_explanations(
        model, X_test, top_feats, spec.categorical_cols, seed=0
    )
    assert 0.0 <= lime_res["stability_jaccard"] <= 1.0
    assert set(lime_res["importance"].index) == set(top_feats)

    report = drift_metrics.ranking_drift_report(shap_imp, shap_imp)
    assert abs(report["spearman_rho"] - 1.0) < 1e-9, "a ranking compared to itself must be perfectly correlated"

    combined = experiment_runner.run_all_seeds(config.SEEDS)
    assert set(combined["regime"].unique()) == {"frozen", "retrained"}
    assert combined["seed"].nunique() == len(config.SEEDS)
    assert not combined["perf_roc_auc"].isna().all()

    per_seed, summary_df = analysis.run_full_analysis()
    assert len(per_seed) == len(config.SEEDS)
    for f in ["performance_decay.png", "explanation_drift.png", "lead_time_distribution.png"]:
        assert os.path.exists(os.path.join(config.FIGURES_DIR, f))

    os.remove(tx_path)
    os.remove(id_path)
    print("\nSMOKE TEST PASSED -- pipeline is wired correctly end-to-end.")


if __name__ == "__main__":
    test_full_pipeline_runs_on_synthetic_data()
