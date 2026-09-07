"""
make_synthetic_data.py
=======================
Generates a synthetic dataset that mimics the *schema* of IEEE-CIS
(TransactionDT, TransactionAmt, ProductCD, card1-6, addr1-2, C1-14, D1-15,
M1-9, a handful of V-columns, and a few identity id_/DeviceType columns)
with a KNOWN, INJECTED concept-drift point and drift-carrying feature.

Purpose
-------
1. Lets the whole pipeline (data_loader -> feature_engineering -> modeling
   -> SHAP/LIME -> drift metrics -> lead-lag analysis) be unit/smoke tested
   without requiring the ~700MB real Kaggle download.
2. Serves as a controlled benchmark in the paper itself: because we KNOW
   which feature's relationship with fraud flips and exactly which window
   it flips in, we can report whether the SHAP/LIME drift signal detects it
   at (or before) the correct window -- a sanity/validity check for the
   whole methodology, independent of whatever is found on the real data.

This file produces data with the exact column names required by
feature_engineering.py so it is a drop-in stand-in for
(train_transaction.csv, train_identity.csv).
"""

import numpy as np
import pandas as pd


def _make_categorical(rng, n, categories, p=None):
    return rng.choice(categories, size=n, p=p)


def generate_synthetic_ieee_cis(
    n_rows: int = 60000,
    n_days: int = 182,
    drift_day: int = 91,
    drift_feature: str = "card1",
    fraud_rate: float = 0.035,
    seed: int = 0,
):
    """
    Returns (transaction_df, identity_df) shaped like the real IEEE-CIS files.

    Concept drift design
    ---------------------
    Before `drift_day`: fraud probability is driven mainly by
        - `drift_feature` bucket A (a specific merchant/card cohort)
        - large TransactionAmt
    After `drift_day`: the `drift_feature` signal is REVERSED / dies out and
    fraud instead becomes driven by a different feature (`C1`, a proxy for a
    "velocity" count feature), simulating a fraud ring switching tactics.
    This is exactly the scenario the paper studies: a feature that "was
    highly predictive... may become obsolete next month due to concept
    drift".
    """
    rng = np.random.default_rng(seed)

    seconds_per_day = 86400
    max_dt = n_days * seconds_per_day
    # Transactions are not uniform across the day/week; emulate mild diurnal
    # clustering but keep it simple and monotonically sortable.
    transaction_dt = np.sort(rng.integers(low=86400, high=max_dt, size=n_rows))
    day_index = transaction_dt // seconds_per_day

    # --- core numeric features -------------------------------------------------
    transaction_amt = np.round(rng.gamma(shape=2.0, scale=40.0, size=n_rows) + 1, 2)

    card1 = rng.integers(1000, 18000, size=n_rows)
    card2 = rng.integers(100, 600, size=n_rows).astype(float)
    card3 = rng.choice([150.0, 185.0, 100.0], size=n_rows)
    card5 = rng.choice([102.0, 117.0, 226.0, 137.0], size=n_rows)
    addr1 = rng.integers(100, 500, size=n_rows).astype(float)
    addr2 = rng.choice([87.0, 60.0, 96.0], size=n_rows)

    # C1-C14: "counting" features (e.g. # of addresses tied to a card) -- we
    # only materialize a handful for tractability; a velocity-style feature
    # C1 is the post-drift fraud driver.
    C1 = rng.poisson(lam=2.0, size=n_rows).astype(float)
    C2 = rng.poisson(lam=1.5, size=n_rows).astype(float)
    C_extra = {f"C{i}": rng.poisson(lam=1.0, size=n_rows).astype(float) for i in range(3, 15)}

    # D1-D15: "days since" features
    D1 = rng.exponential(scale=50, size=n_rows)
    D_extra = {f"D{i}": rng.exponential(scale=30, size=n_rows) for i in range(2, 16)}

    # M1-M9: match flags
    M_cols = {f"M{i}": _make_categorical(rng, n_rows, ["T", "F", np.nan], p=[0.45, 0.45, 0.10])
              for i in range(1, 10)}

    # A handful of V-columns (the real data has 339; we simulate 15 as pure
    # noise, enough to exercise dimensionality-driven code paths like
    # TOP_K_FEATURES_FOR_LIME without a 339-wide frame or a large file).
    V_cols = {f"V{i}": rng.normal(0, 1, size=n_rows).astype(np.float32) for i in range(1, 16)}

    ProductCD = _make_categorical(rng, n_rows, ["W", "C", "R", "H", "S"], p=[0.6, 0.15, 0.1, 0.1, 0.05])
    P_emaildomain = _make_categorical(
        rng, n_rows,
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", np.nan],
        p=[0.4, 0.2, 0.15, 0.15, 0.10],
    )
    R_emaildomain = _make_categorical(
        rng, n_rows,
        ["gmail.com", "yahoo.com", np.nan],
        p=[0.3, 0.2, 0.5],
    )

    # --- ground-truth fraud generating process with injected drift -------------
    pre_drift_mask = day_index < drift_day

    drift_feature_values = {"card1": card1}.get(drift_feature, card1)
    # "cohort A" = drift feature value in its lowest quartile
    cohort_a = drift_feature_values < np.quantile(drift_feature_values, 0.25)

    logit = np.full(n_rows, -3.6)  # base rate anchor, calibrated below
    logit += 0.9 * (np.log1p(transaction_amt) - np.log1p(transaction_amt).mean()) / np.log1p(transaction_amt).std()

    # Pre-drift regime: cohort_a on drift_feature strongly predicts fraud.
    logit_pre = logit + 2.6 * cohort_a.astype(float)
    # Post-drift regime: that relationship vanishes; C1 velocity becomes the driver.
    c1_z = (C1 - C1.mean()) / (C1.std() + 1e-6)
    logit_post = logit + 2.6 * (c1_z > 1.0).astype(float)

    final_logit = np.where(pre_drift_mask, logit_pre, logit_post)

    # Calibrate intercept via bisection so overall fraud_rate matches target.
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        rate = sigmoid(final_logit + mid).mean()
        if rate > fraud_rate:
            hi = mid
        else:
            lo = mid
    intercept_adj = (lo + hi) / 2
    p_fraud = sigmoid(final_logit + intercept_adj)
    isFraud = rng.binomial(1, p_fraud)

    transaction_id = np.arange(2987000, 2987000 + n_rows)

    tx = pd.DataFrame({
        "TransactionID": transaction_id,
        "isFraud": isFraud,
        "TransactionDT": transaction_dt,
        "TransactionAmt": transaction_amt,
        "ProductCD": ProductCD,
        "card1": card1, "card2": card2, "card3": card3,
        "card4": _make_categorical(rng, n_rows, ["visa", "mastercard", "amex", "discover"]),
        "card5": card5,
        "card6": _make_categorical(rng, n_rows, ["debit", "credit"]),
        "addr1": addr1, "addr2": addr2,
        "P_emaildomain": P_emaildomain,
        "R_emaildomain": R_emaildomain,
        "C1": C1, "C2": C2, **C_extra,
        "D1": D1, **D_extra,
        **M_cols,
        **V_cols,
    })

    # --- synthetic identity table (only a subset of rows have identity info,
    # exactly like the real IEEE-CIS challenge) ---------------------------------
    has_identity = rng.random(n_rows) < 0.24
    id_rows = tx.loc[has_identity, "TransactionID"].values
    n_id = len(id_rows)
    identity = pd.DataFrame({
        "TransactionID": id_rows,
        "id_01": rng.normal(0, 1, size=n_id),
        "id_02": rng.normal(100, 30, size=n_id),
        "DeviceType": _make_categorical(rng, n_id, ["mobile", "desktop", np.nan], p=[0.5, 0.4, 0.1]),
        "DeviceInfo": _make_categorical(rng, n_id, ["iOS Device", "Windows", "SM-G950F", np.nan],
                                         p=[0.3, 0.3, 0.2, 0.2]),
    })

    return tx, identity


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src import config

    tx, ident = generate_synthetic_ieee_cis(n_rows=15000)
    tx_path = config.TRANSACTION_FILE.replace("train_transaction", "SYNTHETIC_train_transaction")
    id_path = config.IDENTITY_FILE.replace("train_identity", "SYNTHETIC_train_identity")
    tx.to_csv(tx_path, index=False)
    ident.to_csv(id_path, index=False)
    print(f"Wrote synthetic transaction table -> {tx_path}  shape={tx.shape}")
    print(f"Wrote synthetic identity table    -> {id_path}  shape={ident.shape}")
    print(f"Overall fraud rate: {tx['isFraud'].mean():.4f}")
