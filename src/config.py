"""
config.py
=========
Central configuration for the "Temporal Faithfulness of Explanations Under
Concept Drift in Streaming Fraud" experiment pipeline.

Every tunable knob used anywhere in the pipeline lives here so that a run is
fully reproducible from a single file plus a seed list.
"""

import os
import multiprocessing

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TABLES_DIR = os.path.join(OUTPUT_DIR, "tables")

for _d in (DATA_DIR, OUTPUT_DIR, FIGURES_DIR, TABLES_DIR):
    os.makedirs(_d, exist_ok=True)

# Expected raw IEEE-CIS files (download from Kaggle, see README.md)
TRANSACTION_FILE = os.path.join(DATA_DIR, "train_transaction.csv")
IDENTITY_FILE = os.path.join(DATA_DIR, "train_identity.csv")

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
# The study requires >= 10 seeds. We use 12 so that after dropping any seed
# that fails a sanity check (e.g. a degenerate window with a single class)
# we still comfortably clear the >=10 requirement.
SEEDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# --------------------------------------------------------------------------
# Compute
# --------------------------------------------------------------------------
# Apple Silicon (M3 Pro) has 14 CPU cores. LightGBM / SHAP TreeExplainer /
# LIME all parallelize over CPU processes/threads; none of them benefit from
# the on-chip GPU (MPS), which is only exposed to PyTorch/TensorFlow/JAX and
# is deliberately NOT used here so the pipeline stays portable to any
# machine (incl. the CI-style Linux box this code is validated on).
N_JOBS = max(1, min(14, multiprocessing.cpu_count()))

# --------------------------------------------------------------------------
# Temporal windowing
# --------------------------------------------------------------------------
# TransactionDT in IEEE-CIS is seconds elapsed from an arbitrary reference
# point and spans ~182 days. We slide non-overlapping windows across it to
# emulate a streaming deployment.
SECONDS_PER_DAY = 86400
WINDOW_SIZE_DAYS = 14          # length of each streaming "batch"
STEP_SIZE_DAYS = 14            # 14 == non-overlapping; < WINDOW_SIZE_DAYS overlaps
MIN_WINDOW_ROWS = 500          # drop windows with fewer rows (too noisy)
MIN_WINDOW_POSITIVES = 5       # drop windows with too few fraud examples

BASELINE_WINDOW_INDEX = 0      # window used to train the "frozen" model
                                # and as the reference point for all drift metrics

# Within EVERY window we further split by time (never shuffled) into a
# train slice and a held-out test slice. This lets the "retrained" control
# regime be evaluated out-of-sample within its own window (rather than
# trivially in-sample), so its AUC/F1 decay curve is a meaningful
# counterfactual for the "frozen" regime's decay curve.
TRAIN_FRAC_WITHIN_WINDOW = 0.7

# --------------------------------------------------------------------------
# Modeling
# --------------------------------------------------------------------------
LGBM_PARAMS = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    min_child_samples=30,
    n_jobs=N_JOBS,
    verbosity=-1,
)

# Two deployment regimes are compared throughout:
#   "frozen"    -> model trained once on the baseline window, never retrained
#                  (the realistic decay scenario the paper is about)
#   "retrained" -> model retrained from scratch on every window
#                  (control condition: isolates data-distribution drift in
#                  the explanations from model-staleness drift)
MODEL_REGIMES = ["frozen", "retrained"]

CLASSIFICATION_THRESHOLD = 0.5  # for F1/precision/recall; PR/ROC-AUC are threshold-free

# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
# Restricting local-explanation work (LIME) to the top-K globally important
# features (per the baseline-window SHAP ranking) keeps runtime tractable
# given IEEE-CIS has ~430 raw features after merging identity+transaction.
# This mirrors common practice in explanation-stability literature
# (e.g. Alvarez-Melis & Jaakkola, 2018) and is reported as a design choice
# in the paper, not hidden as an implementation detail.
TOP_K_FEATURES_FOR_LIME = 50

SHAP_SUBSAMPLE_SIZE = 2000          # rows per window used for SHAP (speed)
LIME_N_INSTANCES_PER_WINDOW = 40    # instances sampled per window for LIME
LIME_N_REPEATS_FOR_STABILITY = 5    # repeated LIME fits per instance (perturbation stability)
LIME_NUM_SAMPLES = 2000             # perturbation samples LIME draws per explanation
RANKING_TOP_K = 10                  # top-k used for Jaccard-overlap style drift metrics

# --------------------------------------------------------------------------
# Drift / early-warning analysis
# --------------------------------------------------------------------------
MAX_LAG_WINDOWS = 6                  # max lead/lag (in #windows) searched for cross-correlation
N_PERMUTATIONS = 2000                # permutation test resamples for significance of lead-lag corr
PERFORMANCE_DROP_THRESHOLD = 0.03    # AUC drop (absolute) that defines "decay has started"
EXPLANATION_DRIFT_THRESHOLD = 0.30   # 1 - Spearman rho threshold that defines "explanation has drifted"

# --- Post-review robustness additions -------------------------------------
# On the real IEEE-CIS run, PERFORMANCE_DROP_THRESHOLD above was crossed at
# window 1 for every single one of the 12 seeds -- the earliest window that
# could possibly cross it. That "floor effect" means no signal could ever be
# measured as leading it, which makes any "lead time" computed against this
# fixed absolute threshold uninformative regardless of what the underlying
# data actually shows. Separately, EXPLANATION_DRIFT_THRESHOLD is a single
# absolute cutoff shared by SHAP and LIME even though the two drift signals
# live on very different absolute scales in practice (SHAP drift stayed
# below ~0.02 for the frozen model; LIME regularly exceeded 0.3-0.6) -- one
# shared absolute number cannot be a fair cutoff for both.
#
# RELATIVE_CROSSING_FRACTIONS fixes both problems at once: instead of an
# absolute cutoff, "has this signal crossed" is defined as "has it reached
# X% of its OWN observed peak over the study window" -- which adapts
# automatically to each signal's scale and is swept across several values
# (rather than picking one arbitrary number) so the lead-time conclusion's
# sensitivity to this choice is visible rather than hidden.
# See analysis.crossing_sensitivity_summary / drift_metrics.relative_crossing_index.
RELATIVE_CROSSING_FRACTIONS = [0.25, 0.5, 0.75]

# Much of the originally-reported cross-correlation "significance" turned
# out to be driven by both curves sharing a "jump early, then plateau"
# shape rather than genuine period-by-period coupling -- first-differencing
# removes that shared trend/level before correlating. A differenced series
# has one fewer point than its level counterpart, and is already short (13
# windows -> 12 differences), so extreme lags leave very few points to
# correlate; DIFFERENCED_MAX_LAG_WINDOWS is set lower than MAX_LAG_WINDOWS
# to keep those lag estimates from being computed on too few points.
# See analysis.correlation_robustness_summary / drift_metrics.differenced.
DIFFERENCED_MAX_LAG_WINDOWS = 4

RANDOM_STATE_DATA_SPLIT = 12345      # fixed, seed-independent, used only for any seed-agnostic step

# Note on window granularity: the real-data run's "decay" pattern looked
# like a single sharp jump between window 0 and window 1 followed by a
# relative plateau, rather than a gradual drift spread across all 13
# windows. Two things are worth checking as a follow-up (not implemented
# here, since they change the experimental design rather than fix a bug):
# (1) rerun with WINDOW_SIZE_DAYS = STEP_SIZE_DAYS = 7 to roughly double the
#     number of time points and reduce the small-n fragility of any
#     time-series claim; (2) rerun with a later BASELINE_WINDOW_INDEX (e.g.
#     2 or 3) to check whether the sharp early jump is a genuine early
#     regime change or an idiosyncrasy of using the very first 14 days of
#     the dataset as the reference point.
