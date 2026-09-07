"""
data_loader.py
===============
Loading, merging, and temporal windowing of the IEEE-CIS Fraud Detection
dataset (https://www.kaggle.com/c/ieee-fraud-detection).

Only `train_transaction.csv` and `train_identity.csv` are used: these are
the only IEEE-CIS files that carry the `isFraud` label, which is required
to (a) train the model and (b) evaluate AUC/F1 decay over time. The
competition's `test_*.csv` files are unlabeled and therefore cannot be used
for this study without violating requirement #2 (only the stated dataset).
"""

import logging
import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)


def load_raw_data(transaction_file: str = None, identity_file: str = None) -> pd.DataFrame:
    """Load and left-merge the transaction and identity tables.

    Returns a single DataFrame sorted ascending by TransactionDT (this
    ascending time order is the backbone of every downstream step -- the
    data must NEVER be shuffled after this point).
    """
    transaction_file = transaction_file or config.TRANSACTION_FILE
    identity_file = identity_file or config.IDENTITY_FILE

    if not os.path.exists(transaction_file):
        raise FileNotFoundError(
            f"Could not find {transaction_file}. Download the IEEE-CIS Fraud "
            f"Detection dataset from Kaggle "
            f"(https://www.kaggle.com/c/ieee-fraud-detection) and place "
            f"train_transaction.csv / train_identity.csv under {config.DATA_DIR}. "
            f"See README.md for exact instructions. "
            f"(For a dependency-free smoke test, run `python make_synthetic_data.py` "
            f"and point config.TRANSACTION_FILE/IDENTITY_FILE at the SYNTHETIC_* files.)"
        )

    logger.info("Loading transaction table from %s", transaction_file)
    tx = pd.read_csv(transaction_file)

    if identity_file and os.path.exists(identity_file):
        logger.info("Loading identity table from %s", identity_file)
        identity = pd.read_csv(identity_file)
        df = tx.merge(identity, on="TransactionID", how="left")
    else:
        logger.warning("Identity file not found at %s; proceeding with transaction-only features.",
                        identity_file)
        df = tx

    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    logger.info("Loaded merged frame: shape=%s, fraud_rate=%.4f, time span=%.1f days",
                df.shape, df["isFraud"].mean(),
                (df["TransactionDT"].max() - df["TransactionDT"].min()) / config.SECONDS_PER_DAY)
    return df


@dataclass
class Window:
    index: int
    start_day: float
    end_day: float
    df: pd.DataFrame


def create_sliding_windows(
    df: pd.DataFrame,
    window_size_days: int = None,
    step_size_days: int = None,
    min_rows: int = None,
    min_positives: int = None,
) -> List[Window]:
    """Slice `df` (already sorted by TransactionDT) into temporal windows.

    step_size_days == window_size_days  -> non-overlapping (default, and what
        the main experiments use, since overlapping windows would make
        successive drift measurements non-independent).
    step_size_days  < window_size_days  -> overlapping / finer temporal
        resolution, offered for sensitivity-analysis ablations only.
    """
    window_size_days = window_size_days or config.WINDOW_SIZE_DAYS
    step_size_days = step_size_days or config.STEP_SIZE_DAYS
    min_rows = min_rows if min_rows is not None else config.MIN_WINDOW_ROWS
    min_positives = min_positives if min_positives is not None else config.MIN_WINDOW_POSITIVES

    window_size_s = window_size_days * config.SECONDS_PER_DAY
    step_size_s = step_size_days * config.SECONDS_PER_DAY

    t_min, t_max = df["TransactionDT"].min(), df["TransactionDT"].max()

    windows = []
    idx = 0
    start = t_min
    while start < t_max:
        end = start + window_size_s
        mask = (df["TransactionDT"] >= start) & (df["TransactionDT"] < end)
        sub = df.loc[mask]

        if len(sub) >= min_rows and sub["isFraud"].sum() >= min_positives and sub["isFraud"].nunique() > 1:
            windows.append(Window(
                index=idx,
                start_day=(start - t_min) / config.SECONDS_PER_DAY,
                end_day=(end - t_min) / config.SECONDS_PER_DAY,
                df=sub.reset_index(drop=True),
            ))
            idx += 1
        else:
            logger.debug("Dropping degenerate window at start_day=%.1f (rows=%d, positives=%d)",
                         (start - t_min) / config.SECONDS_PER_DAY, len(sub), sub["isFraud"].sum())
        start += step_size_s

    if len(windows) < 3:
        raise ValueError(
            f"Only {len(windows)} usable windows were constructed. Widen "
            f"window_size_days or lower min_rows/min_positives in config.py."
        )

    logger.info("Constructed %d usable sliding windows (window=%dd, step=%dd)",
                len(windows), window_size_days, step_size_days)
    return windows


def split_train_test_within_window(window_df: pd.DataFrame, train_frac: float = None):
    """Time-ordered (never shuffled) split of a single window into a train
    slice (earliest `train_frac` of the window by TransactionDT) and a test
    slice (the remainder). This is what makes the "retrained" regime's
    evaluation genuinely out-of-sample rather than trivially in-sample.
    """
    train_frac = train_frac if train_frac is not None else config.TRAIN_FRAC_WITHIN_WINDOW
    df_sorted = window_df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    cut = max(1, int(len(df_sorted) * train_frac))
    train_df = df_sorted.iloc[:cut].reset_index(drop=True)
    test_df = df_sorted.iloc[cut:].reset_index(drop=True)
    return train_df, test_df
