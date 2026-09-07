# Temporal Faithfulness of Explanations Under Concept Drift in Streaming Fraud

Experiment code for the study: does SHAP (global) / LIME (local) explanation
drift over time act as an **early-warning signal** for fraud-model decay,
*before* AUC/F1 visibly drop? Dataset: [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection)
(`train_transaction.csv` + `train_identity.csv` only — these are the only
IEEE-CIS files with ground-truth labels).

---

## 1. What this code does

For each of `>= 10` random seeds:

1. Sorts all transactions chronologically by `TransactionDT` and slices them
   into non-overlapping sliding windows (default: 14 days each).
2. Trains a LightGBM fraud classifier once on the **earliest** window (the
   *frozen* model) and evaluates it, unmodified, on every later window —
   simulating a model that is deployed and never retrained.
3. As a **control regime**, also retrains a fresh model on each window's own
   data (*retrained* model) and evaluates it on a held-out slice of that
   same window. This isolates "explanation drift caused by a stale model"
   from "explanation drift that would happen anyway because the data
   changed, even for a fresh model."
4. At every window, for both regimes, computes:
   - **Performance**: ROC-AUC, PR-AUC, F1, precision, recall.
   - **SHAP** global feature-importance ranking (`shap.TreeExplainer`,
     exact Shapley values for the tree ensemble).
   - **LIME** local explanations for a sample of instances, including
     **perturbation stability** (re-running LIME 5x on the *same* instance
     to see how much of LIME's variability is just its own sampling noise
     vs. genuine drift).
   - Drift of both rankings relative to the baseline window (Spearman ρ,
     Kendall τ, top-k Jaccard overlap).
   - Per-feature Wasserstein distance between each window's raw feature
     distribution and the baseline window's (a model-free "background
     shift" measure).
5. Runs a **lead-lag analysis**: cross-correlates the explanation-drift
   signal against the performance-drop signal across a window of lags, with
   a permutation test for significance, plus a simple
   first-threshold-crossing comparison and a paired t-test across seeds.

All of this is orchestrated by `main.py`; see `outputs/tables/` and
`outputs/figures/` for what comes out.

## 2. Repository layout

```
src/
├── config.py                 All tunable parameters (seeds, window size, model
│                              hyperparameters, SHAP/LIME sample sizes, etc.)
├── data/
│   ├── data_loader.py        Load/merge IEEE-CIS CSVs, build sliding windows,
│   │                         within-window train/test split.
│   └── feature_engineering.py Build feature contract (label-encoded categoricals
│                              + numeric columns) shared by windows.
├── models/
│   ├── model_utils.py        LightGBM training, evaluation, threshold selection.
│   └── explainers.py         SHAP global importance; LIME local explanations
│                              + perturbation-stability instrumentation.
├── drift/
│   └── drift_metrics.py      Ranking-drift metrics, Wasserstein shift,
│                              cross-correlation / permutation test lead-lag analysis.
└── pipeline/
    ├── experiment_runner.py  Seed x window x regime loop; writes tables/results_*.parquet.
    └── analysis.py           Lead-lag aggregation, paired t-test, figures.
scripts/
└── make_synthetic_data.py    Generates schema-matched synthetic dataset.
main.py                       Single CLI entry point.
tests/test_pipeline_smoke.py  End-to-end smoke test on synthetic data.
```

## 3. Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Apple Silicon (M3 Pro) note**: LightGBM's macOS wheel needs the OpenMP
runtime. If `import lightgbm` fails with an OpenMP/`libomp` error:
```bash
brew install libomp
```
None of this pipeline uses the on-chip GPU (MPS) — LightGBM, SHAP's
TreeExplainer, and LIME are all CPU-bound and will use `os.cpu_count()`
(capped at 14) threads/processes automatically via `config.N_JOBS`.

## 4. Getting the data

This code uses **only** the IEEE-CIS Fraud Detection dataset, as required.
Kaggle requires an account and its terms of use accepted before download,
so this cannot be scripted from a sandboxed environment — download it
yourself:

1. Go to https://www.kaggle.com/c/ieee-fraud-detection/data (sign in,
   accept the competition rules).
2. Download `train_transaction.csv` and `train_identity.csv` (either the
   "Download All" zip, or individually, or via `kaggle competitions
   download -c ieee-fraud-detection` if you have the Kaggle CLI + API
   token configured).
3. Place both files in `data/`:
   ```
   data/train_transaction.csv
   data/train_identity.csv
   ```
   (`config.TRANSACTION_FILE` / `config.IDENTITY_FILE` point here by
   default; override with `--transaction-file/--identity-file` if you keep
   them elsewhere.)

Only the labeled `train_*` files are used. The competition's `test_*.csv`
files have no `isFraud` column and cannot be used for this study.

## 5. Running

**Step 0 — validate the pipeline works on your machine (no download needed,
~1-2 minutes):**
```bash
python scripts/make_synthetic_data.py
python -m pytest tests/test_pipeline_smoke.py -v
# or:
python tests/test_pipeline_smoke.py
```
This exercises the entire pipeline (data → features → model → SHAP → LIME
→ drift metrics → lead-lag stats → figures) on a small synthetic dataset
with a known, injected concept-drift point. It is a wiring/correctness
check, not a source of any reportable research numbers.

**Full run on real data, all `config.SEEDS` (12 by default, satisfying the
`>=10` requirement):**
```bash
python main.py
```

**Quick real-data test with fewer seeds first (recommended before the full
run):**
```bash
python main.py --seeds 0 1 2
```

**Re-run only the analysis/figures on already-computed results** (e.g.
after tweaking a plot):
```bash
python main.py --skip-experiments
```

Outputs:
- `outputs/tables/results_seed_<n>.parquet` — one seed's full results (saved
  incrementally, so a long run can be inspected/resumed manually mid-flight).
- `outputs/tables/results_all_seeds.{parquet,csv}` — combined table, one row
  per (seed, regime, window).
- `outputs/tables/leadlag_per_seed.csv` — per-seed lead-lag / crossing stats,
  now including the SHAP/LIME/Wasserstein level-vs-differenced correlation
  columns and the relative-threshold crossing sweep (see section 9 below).
- `outputs/tables/leadlag_summary.csv` — the ORIGINAL absolute-threshold,
  SHAP/LIME-only cross-seed aggregation + paired t-test, kept unchanged for
  continuity. **See section 9 for why this specific table should not be
  reported as the headline result on its own.**
- `outputs/tables/correlation_robustness_summary.csv` — level vs.
  differenced cross-correlation significance for all three signals; shows
  directly how much of any "significant" correlation survives trend removal.
- `outputs/tables/crossing_sensitivity_summary.csv` — the floor-effect-free,
  relative-threshold version of the lead-time analysis, swept across
  `config.RELATIVE_CROSSING_FRACTIONS`.
- `outputs/tables/regime_stability_comparison.csv` — the frozen-vs-retrained
  ranking-stability paired comparison; **this is the recommended headline
  result table** (see section 9).
- `outputs/figures/performance_decay.png`, `explanation_drift.png` —
  unchanged from the original analysis.
- `outputs/figures/lead_time_distribution.png` — unchanged (original
  absolute-threshold method, SHAP/LIME only).
- `outputs/figures/relative_lead_time_distribution.png` — new: the
  relative-threshold counterpart, SHAP/LIME/Wasserstein.
- `outputs/figures/regime_stability.png` — new: the recommended headline
  figure (frozen vs. retrained ranking stability).
- `outputs/figures/ccf_seed_<n>.png` — enhanced: now shows level AND
  differenced cross-correlation side by side, for SHAP, LIME, and
  Wasserstein together.

## 6. Runtime expectations (M3 Pro)

I benchmarked the two expensive steps directly (not guessed) on a
realistically-sized window (~13.5k held-out test rows, drawn from a
45k-row window — matching what a 14-day slice of the real ~590k-row
IEEE-CIS dataset looks like), single-threaded:

| Step | Measured cost (1 thread) | # calls in a full 12-seed run | Serial total |
|---|---|---|---|
| Model training (400 trees) | ~6.0 s | 156 (12 seeds × 13 windows, minus window-0 dedup) | ~15 min |
| SHAP (`TreeExplainer`, 2000-row subsample) | ~12.0 s | 312 (12 seeds × 26 SHAP calls/seed) | ~62 min |
| One LIME `explain_instance` (2000 samples, top-50 features) | ~0.25 s | ~62,400 (12 seeds × 5,200 LIME calls/seed) | ~4.3 hr |
| **Total, fully serial, single core** | | | **~5.5 hr** |

So **yes, left fully serial this would be too slow to iterate on**. Two
things claw that back:

1. **Seed-level parallelism (the default).** `experiment_runner.run_all_seeds`
   runs seeds concurrently via `joblib` rather than one after another —
   since each seed is a fully independent baseline-model-plus-sliding-
   window run, this is embarrassingly parallel and, unlike LightGBM's own
   internal thread-parallelism or LIME's per-instance parallelism, it keeps
   *every* core busy through *every* phase (training, SHAP, and LIME
   alike), not just the LIME inner loop. With `config.SEEDS` sized to 12 —
   close to an M3 Pro's CPU core count (11 or 12, depending on the exact
   variant; note the "14" in this machine's spec is its GPU core count,
   not CPU) — this is close to the ideal case: nearly all 12 seeds run
   truly at once, so the ~5.5 hour serial estimate above collapses toward
   **~30–45 minutes** for the full study (imperfect scaling from
   per-process overhead, the P-core/E-core split, and OS scheduling means
   it won't be a clean /12, but it should land well under an hour).
2. **A thread-oversubscription bug I found and fixed while checking this.**
   LightGBM bakes its own thread count into the trained model, and LIME's
   instance loop is itself parallelized — without care, each of N parallel
   LIME workers would *also* try to spawn LightGBM's full thread count for
   its `predict_proba` calls, oversubscribing the machine by roughly N².
   `explainers._explain_one_instance` now forces `n_jobs=1` on its local
   copy of the model before using it (safe: each worker process gets its
   own pickled copy). The same logic extends to seed-level parallelism: the
   total core budget (`config.N_JOBS`) is split between "how many seeds run
   at once" and "how many threads each seed gets," so total concurrency
   never exceeds the machine's core count — see the docstring on
   `run_all_seeds` for the exact split.

**Trade-off to know about**: seed-level parallelism means each of the
(up to 12) worker processes holds its own copy of the full windowed
dataset in memory. For the real IEEE-CIS data this is a few hundred MB
per worker, not gigabytes, but on a memory-constrained machine you can
fall back to sequential seeds (which still parallelizes LIME's inner
instance loop across cores) by setting `config.N_JOBS = 1` or passing
fewer seeds per `run_all_seeds` call — results are saved incrementally
per seed regardless, so partial runs are never wasted.

**Before committing to the full run**, time one seed on your machine and
extrapolate:
```bash
time python main.py --seeds 0
```
and to iterate faster while developing, the same knobs as before still
apply: lower `LIME_N_INSTANCES_PER_WINDOW` / `LIME_N_REPEATS_FOR_STABILITY`
/ `LIME_NUM_SAMPLES`, lower `TOP_K_FEATURES_FOR_LIME`, or widen
`WINDOW_SIZE_DAYS` (fewer, larger windows → fewer SHAP/LIME calls overall).

## 7. Design choices worth stating explicitly in the paper (Threats to Validity)

- **Global category vocabulary**: categorical columns are label-encoded
  once on the full historical dataset before windowing (see
  `feature_engineering.build_feature_spec`), not incrementally online. This
  mirrors a production feature store with a periodically refreshed
  vocabulary, but means genuinely novel categories appearing only in later
  windows are handled via an explicit `__UNSEEN__` bucket rather than
  triggering online vocabulary growth.
- **Reduced LIME feature surface**: LIME's local surrogate is fit only over
  the top-`TOP_K_FEATURES_FOR_LIME` (default 50) globally important
  features from the *baseline* window's SHAP ranking (see
  `explainers._make_reduced_predict_fn`); all other features are held fixed
  at the explained instance's true value when querying the real model. This
  keeps LIME tractable on IEEE-CIS's ~430 raw features and follows common
  practice in explanation-stability literature, but means features that
  only become important post-drift and were never in the baseline top-K are
  invisible to the LIME analysis (SHAP's analysis is unrestricted and does
  not have this limitation).
- **Frozen decision threshold**: the classification threshold for F1/
  precision/recall is chosen once via the F1-maximizing point on each
  model's own training split (`model_utils.find_best_threshold`) and then
  held fixed across all evaluation windows for that model — never re-tuned
  against a window's ground truth — so its decay is a genuine drift signal,
  not a re-calibration artifact.
- **Window size / step size** (default 14 days, non-overlapping) trades off
  the number of independent temporal observations against per-window
  sample size (and thus SHAP/LIME estimation noise); this is an obvious
  and easy sensitivity analysis to run by sweeping `WINDOW_SIZE_DAYS` /
  `STEP_SIZE_DAYS` and re-running.
- **LightGBM only**: the pipeline currently instruments one model family.
  `model_utils.py` is the only file that would need a second implementation
  to add an XGBoost/logistic-regression comparison arm.

## 8. Post-review robustness fixes (read this before writing up results)

A first full 12-seed run on the real IEEE-CIS data was audited before
write-up. Three problems were found in the original `analysis.py`, all
fixed by *adding* new analyses alongside the originals (nothing removed —
every previously-reported number is still reproduced exactly, see the
outputs list above for which file is which):

1. **The absolute performance threshold had a floor effect.**
   `config.PERFORMANCE_DROP_THRESHOLD = 0.03` was crossed at **window 1 for
   every single one of the 12 seeds** on the real data — the earliest
   window that could possibly cross it. No signal could ever be measured as
   "leading" a threshold that's already hit at the first opportunity every
   time, so the original `leadlag_summary.csv` "LIME lags performance by
   ~1.8 windows" result is largely an artifact of this miscalibration, not
   necessarily a real finding. Fix: `drift_metrics.relative_crossing_index`
   + `config.RELATIVE_CROSSING_FRACTIONS`, aggregated in
   `analysis.crossing_sensitivity_summary` — defines "crossed" relative to
   each signal's own observed range instead of a hand-picked absolute
   number, and sweeps the fraction rather than committing to one.

2. **Most of the reported cross-correlation "significance" was a
   trend-sharing artifact.** The original permutation test (12/12 seeds
   "significant" for both SHAP and LIME) treats each window as
   exchangeable, which does not account for both curves sharing a "jump
   early, then plateau" shape — two curves with that shape alone will show
   high correlation at many lags regardless of any real dynamic coupling.
   Re-running the same test on first-differenced series dropped
   significance to roughly 55-60% of seeds for SHAP and 25% (near chance)
   for LIME. Fix: `drift_metrics.differenced` + the `diff` rows in
   `analysis.correlation_robustness_summary` — report the differenced
   result as primary, the level result as context.

3. **No comparison against a plain, non-explanation-based drift monitor.**
   The per-window Wasserstein distance between each window's raw feature
   distribution and the baseline's (`mean_feature_wasserstein_shift`, only
   needs raw data, no model or explainer) was already being computed by
   `experiment_runner.py` but never used in the lead-lag analysis. It is
   now run through every analysis above as a third signal alongside SHAP
   and LIME, so the results directly show whether explanation-based
   monitoring adds anything beyond a much cheaper covariate-shift check.

Separately — and not a bug fix, but the single strongest and most robust
result found in the audit — is `analysis.regime_stability_comparison` /
`outputs/figures/regime_stability.png`: a frozen model's SHAP ranking
stayed within ρ ≈ 0.99 of its baseline across *every* window and seed,
while a model retrained from scratch on each window (evaluated the same
way) showed materially more ranking movement (ρ ≈ 0.75-0.80) — despite the
retrained model tracking ground truth *better*. Unlike every lead-lag
analysis above, this comparison requires no threshold choice at all, is
directly paired and seed-replicated, and survived every robustness check
run against it. **Consider making this the paper's headline finding**
("post-hoc explanations can look artificially stable under model
staleness, giving false reassurance exactly when monitoring matters most")
rather than the early-warning claim, which the fixes above show is not yet
well-supported on its own.

Two further things worth trying as follow-up experiments (not implemented
here since they change the experimental design rather than fix a bug —
see the comment in `config.py`'s windowing section): rerunning with
`WINDOW_SIZE_DAYS = STEP_SIZE_DAYS = 7` to get more time points and reduce
the small-n fragility of any time-series claim, and rerunning with a later
`BASELINE_WINDOW_INDEX` to check whether the sharp window-0-to-1 jump seen
in the real data is a genuine early regime change or an artifact of using
the very first 14 days as the reference point.

## 9. Extending

- Add a model family: implement `train_model`/`evaluate_model` variants and
  thread a `model_family` argument through `experiment_runner.py`.
- Add a metric: extend `model_utils.evaluate_model`'s returned dict; it
  flows automatically into `results_all_seeds.parquet` via the `perf_`
  prefix in `experiment_runner._flatten_row`.
- Add a drift metric: add a function to `drift_metrics.py` and call it
  alongside `ranking_drift_report` in `experiment_runner.run_single_seed`.
