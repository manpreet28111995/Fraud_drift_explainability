"""
feature_engineering.py
========================
Turns the raw merged IEEE-CIS frame into a model-ready (X, y) representation
that is *consistent across all sliding windows*, which is essential: SHAP/
LIME rankings are only comparable across windows if the feature matrix's
columns and categorical encodings mean the same thing in every window.

Design choice (explicitly noted for the paper): categorical vocabularies
(LabelEncoder classes) are fit ONCE on the full historical dataset before
windowing. This mirrors a production feature store with a periodically
refreshed category vocabulary rather than a fully online/incremental
encoder, and is a deliberate simplification we report as a limitation --
see README.md / paper Section on "Threats to Validity".
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

ID_COLS = ["TransactionID"]
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"

# Columns that are categorical / high-cardinality-but-discrete in IEEE-CIS.
BASE_CATEGORICAL_COLS = (
    ["ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
     "addr1", "addr2", "P_emaildomain", "R_emaildomain",
     "DeviceType", "DeviceInfo"]
    + [f"M{i}" for i in range(1, 10)]
    + [f"id_{i}" for i in range(12, 39)]  # categorical identity flags in real dataset
)


@dataclass
class FeatureSpec:
    """Frozen feature contract shared by every window and every seed."""
    numeric_cols: List[str]
    categorical_cols: List[str]
    all_cols: List[str]
    encoders: Dict[str, LabelEncoder] = field(default_factory=dict)


def build_feature_spec(df: pd.DataFrame) -> FeatureSpec:
    """Fit label encoders once, globally, and freeze the feature contract."""
    categorical_cols = [c for c in BASE_CATEGORICAL_COLS if c in df.columns]
    exclude = set(ID_COLS + [TARGET_COL, TIME_COL] + categorical_cols)
    numeric_cols = [c for c in df.columns if c not in exclude]

    encoders = {}
    for c in categorical_cols:
        le = LabelEncoder()
        vals = df[c].astype(str).fillna("missing")
        le.fit(np.append(vals.unique(), "__UNSEEN__"))
        encoders[c] = le

    logger.info("Feature spec: %d numeric, %d categorical columns",
                len(numeric_cols), len(categorical_cols))
    return FeatureSpec(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        all_cols=numeric_cols + categorical_cols,
        encoders=encoders,
    )


def _safe_transform(le: LabelEncoder, series: pd.Series) -> np.ndarray:
    vals = series.astype(str).fillna("missing")
    known = set(le.classes_)
    vals = vals.where(vals.isin(known), other="__UNSEEN__")
    return le.transform(vals)


def transform(df: pd.DataFrame, spec: FeatureSpec) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply the frozen feature contract to any window's raw DataFrame."""
    out = pd.DataFrame(index=df.index)

    for c in spec.numeric_cols:
        if c in df.columns:
            out[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            out[c] = np.nan
    out[spec.numeric_cols] = out[spec.numeric_cols].fillna(-999.0)

    # a couple of cheap, leakage-free temporal features derived from
    # TransactionDT (NOT the raw counter itself, which would just encode
    # row order and trivially "predict" via recency rather than any real
    # fraud signal).
    seconds_in_day = df[TIME_COL] % 86400
    out["hour_of_day"] = (seconds_in_day // 3600).astype(int)
    out["day_of_week"] = ((df[TIME_COL] // 86400) % 7).astype(int)

    for c in spec.categorical_cols:
        if c in df.columns:
            out[c] = _safe_transform(spec.encoders[c], df[c])
        else:
            out[c] = _safe_transform(spec.encoders[c], pd.Series(["missing"] * len(df), index=df.index))
        # IMPORTANT: keep as a plain integer dtype, NOT pandas `category`
        # dtype. If we cast to `category`, pandas would derive per-window
        # codes from whichever categories happen to appear in that window's
        # slice, silently breaking the global, seed-independent vocabulary
        # from `spec.encoders` that every window and every seed must share
        # for SHAP/LIME rankings to be comparable across time. LightGBM is
        # told which columns are categorical explicitly via the
        # `categorical_feature` argument at fit time (see model_utils.py)
        # and uses these raw integer codes directly as category ids.
        out[c] = out[c].astype("int32")

    y = df[TARGET_COL].astype(int).reset_index(drop=True)
    out = out.reset_index(drop=True)
    return out, y


def feature_names_with_time(spec: FeatureSpec) -> List[str]:
    return spec.numeric_cols + ["hour_of_day", "day_of_week"] + spec.categorical_cols
