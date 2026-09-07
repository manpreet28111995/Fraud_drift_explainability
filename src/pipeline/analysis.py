"""
analysis.py
=============
Consumes outputs/tables/results_all_seeds.parquet (produced by
experiment_runner.py) and answers the paper's central question:

    "Can SHAP/LIME ranking drift act as an early-warning signal for model
     decay BEFORE AUC/F1 drop?"

For each seed (frozen regime only -- the realistic deployment scenario):
  1. Build time series over the sliding windows for THREE signals:
       explanation_drift(t) = 1 - Spearman(rank_t, rank_baseline)  [SHAP, LIME]
       covariate_shift(t)   = mean per-feature Wasserstein distance vs.
                              baseline window                       [Wasserstein]
       performance_drop(t)  = AUC_baseline - AUC_t
     The Wasserstein signal is a cheap, model-free baseline: it answers
     whether explanation-based monitoring tells you anything a simple
     covariate-shift monitor wouldn't already tell you for free.
  2. Find the lag at which each drift/shift signal correlates most
     strongly with performance_drop(t + lag), computed on BOTH the raw
     level series and the first-differenced series (trend/level removed --
     see drift_metrics.differenced for why the differenced version is the
     more trustworthy one when both curves simply "jump early, then
     plateau"). Permutation-test each for significance.
  3. Compute "lead time" two ways: the original fixed ABSOLUTE-threshold
     crossing (config.PERFORMANCE_DROP_THRESHOLD /
     config.EXPLANATION_DRIFT_THRESHOLD, kept for continuity), and a
     RELATIVE-threshold crossing swept across
     config.RELATIVE_CROSSING_FRACTIONS (added after finding the absolute
     performance threshold was crossed at window 1 for every seed on the
     real data -- a floor effect that makes lead time against it
     uninformative, and that one shared absolute threshold cannot fairly
     compare signals living on very different scales).
  4. Separately, and without depending on any threshold at all: a paired
     comparison of ranking stability between the FROZEN and continuously-
     RETRAINED regimes (regime_stability_comparison) -- the most robust
     finding from the initial real-data run, and the recommended headline
     result (see regime_stability_comparison's docstring).

Every one of the above is written to outputs/tables/*.csv and
outputs/figures/*.png; nothing from the original analysis is removed, only
extended, so all originally-reported numbers remain reproducible alongside
the new, more robust versions.
"""

import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src import config
from src.drift import drift_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def load_results(path: str = None) -> pd.DataFrame:
    path = path or os.path.join(config.TABLES_DIR, "results_all_seeds.parquet")
    df = pd.read_parquet(path)
    return df.sort_values(["seed", "regime", "window"]).reset_index(drop=True)


def per_seed_leadlag(df: pd.DataFrame) -> pd.DataFrame:
    """Runs the lead-lag + threshold-crossing analysis independently for
    every seed, now covering three things the original version did not:

    1. ORIGINAL absolute-threshold crossing for SHAP/LIME (unchanged,
       columns `{shap,lime}_best_lag/_best_corr/_perm_pvalue/
       _explain_cross_window/_perf_cross_window/_lead_time_windows`) --
       kept exactly as before for continuity with earlier results.
    2. A third signal, `wasserstein` (the per-window mean Wasserstein
       distance of raw features vs. the baseline window -- already
       computed by experiment_runner.py, requires no model or explainer at
       all), run through the SAME analyses as SHAP/LIME. This answers the
       natural question: does explanation-based monitoring show you
       anything a much cheaper, model-free covariate-shift monitor
       wouldn't already show you?
    3. For all three signals: LEVEL vs. DIFFERENCED cross-correlation
       (`{signal}_{level,diff}_best_lag/_best_corr/_perm_pvalue`), and a
       RELATIVE-threshold crossing sweep across
       config.RELATIVE_CROSSING_FRACTIONS
       (`{signal}_relcross_f{fraction}[_lead_time]`,
       `perf_relcross_f{fraction}`). Both were added after finding that (a)
       the original absolute PERFORMANCE_DROP_THRESHOLD was crossed at
       window 1 for every seed on the real data (a floor effect that makes
       "lead time" against it uninformative), and (b) most of the
       originally-reported cross-correlation "significance" evaporated
       once the shared trend/level shape between the two curves was
       removed via differencing. See config.py's "Post-review robustness
       additions" comment for the full rationale.
    """
    records = []
    frozen = df[df["regime"] == "frozen"]

    for seed, g in frozen.groupby("seed"):
        g = g.sort_values("window")
        baseline_auc = g.loc[g["window"] == config.BASELINE_WINDOW_INDEX, "perf_roc_auc"].iloc[0]
        perf_drop = (baseline_auc - g["perf_roc_auc"]).values

        signals = {
            "shap": g["shap_spearman_drift"].values,
            "lime": g["lime_spearman_drift"].values,
            "wasserstein": g["mean_feature_wasserstein_shift"].values,
        }

        rec = dict(seed=seed, n_windows=len(g))

        # ---- 1. ORIGINAL absolute-threshold method (SHAP/LIME only, unchanged) ----
        for name in ("shap", "lime"):
            drift_signal = signals[name]
            lag, corr = drift_metrics.best_lead_lag(drift_signal, perf_drop, config.MAX_LAG_WINDOWS)
            pval = drift_metrics.permutation_test_lead_lag(
                drift_signal, perf_drop, lag, config.N_PERMUTATIONS, seed=seed
            )
            explain_cross = drift_metrics.first_crossing_index(
                drift_signal, config.EXPLANATION_DRIFT_THRESHOLD, "above"
            )
            perf_cross = drift_metrics.first_crossing_index(
                perf_drop, config.PERFORMANCE_DROP_THRESHOLD, "above"
            )
            lead_time = (
                (perf_cross - explain_cross)
                if (explain_cross != -1 and perf_cross != -1)
                else np.nan
            )
            rec.update({
                f"{name}_best_lag": lag,
                f"{name}_best_corr": corr,
                f"{name}_perm_pvalue": pval,
                f"{name}_explain_cross_window": explain_cross,
                f"{name}_perf_cross_window": perf_cross,
                f"{name}_lead_time_windows": lead_time,
            })

        # ---- 2. LEVEL vs. DIFFERENCED cross-correlation, all three signals ----
        for name, sig in signals.items():
            for series_type, sig_t, perf_t, max_lag in (
                ("level", sig, perf_drop, config.MAX_LAG_WINDOWS),
                ("diff", drift_metrics.differenced(sig), drift_metrics.differenced(perf_drop),
                 config.DIFFERENCED_MAX_LAG_WINDOWS),
            ):
                lag, corr = drift_metrics.best_lead_lag(sig_t, perf_t, max_lag)
                pval = drift_metrics.permutation_test_lead_lag(
                    sig_t, perf_t, lag, config.N_PERMUTATIONS, seed=seed
                )
                rec.update({
                    f"{name}_{series_type}_best_lag": lag,
                    f"{name}_{series_type}_best_corr": corr,
                    f"{name}_{series_type}_perm_pvalue": pval,
                })

        # ---- 3. RELATIVE-threshold crossing sweep, all three signals ----
        for fraction in config.RELATIVE_CROSSING_FRACTIONS:
            perf_cross_rel = drift_metrics.relative_crossing_index(perf_drop, fraction)
            rec[f"perf_relcross_f{fraction}"] = perf_cross_rel
            for name, sig in signals.items():
                explain_cross_rel = drift_metrics.relative_crossing_index(sig, fraction)
                lead_rel = (
                    (perf_cross_rel - explain_cross_rel)
                    if (explain_cross_rel != -1 and perf_cross_rel != -1)
                    else np.nan
                )
                rec[f"{name}_relcross_f{fraction}"] = explain_cross_rel
                rec[f"{name}_relcross_f{fraction}_lead_time"] = lead_rel

        records.append(rec)

    return pd.DataFrame(records)


def summarize_leadlag(per_seed: pd.DataFrame) -> dict:
    """Aggregate across seeds with a paired t-test: is the window at which
    explanation drift crosses threshold reliably EARLIER than the window at
    which performance drop crosses threshold?

    NOTE (post-review): this is the ORIGINAL absolute-threshold analysis,
    kept unchanged for continuity with earlier results. On the real
    IEEE-CIS data, PERFORMANCE_DROP_THRESHOLD was found to be crossed at
    window 1 for every single seed -- a floor effect that makes "lead time"
    computed against it largely uninformative regardless of the true
    underlying dynamics. Treat `crossing_sensitivity_summary` (relative,
    threshold-swept) and `correlation_robustness_summary` (level vs.
    differenced) below as the more trustworthy versions of this question,
    and this function's output as a specific, literal reproduction of the
    original fixed-threshold method for comparison.
    """
    summary = {}
    for name in ("shap", "lime"):
        valid = per_seed.dropna(subset=[f"{name}_lead_time_windows"])
        n = len(valid)
        if n >= 2:
            t_stat, p_val = stats.ttest_rel(
                valid[f"{name}_perf_cross_window"], valid[f"{name}_explain_cross_window"]
            )
        else:
            t_stat, p_val = np.nan, np.nan
        summary[name] = dict(
            n_seeds_with_both_crossings=n,
            n_seeds_total=len(per_seed),
            mean_lead_time_windows=valid[f"{name}_lead_time_windows"].mean() if n else np.nan,
            std_lead_time_windows=valid[f"{name}_lead_time_windows"].std() if n else np.nan,
            mean_best_lag=per_seed[f"{name}_best_lag"].mean(),
            mean_best_corr=per_seed[f"{name}_best_corr"].mean(),
            median_perm_pvalue=per_seed[f"{name}_perm_pvalue"].median(),
            frac_seeds_significant_p05=(per_seed[f"{name}_perm_pvalue"] < 0.05).mean(),
            paired_ttest_stat=t_stat,
            paired_ttest_pvalue=p_val,
        )
    return summary


def correlation_robustness_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the level-vs-differenced cross-correlation columns across
    seeds for all three signals (shap, lime, wasserstein), making explicit
    how much of any "significant" lead-lag correlation survives once the
    shared trend/level shape between the two curves is removed.

    A large drop in `frac_seeds_significant_p05` from the `level` row to
    the `diff` row for a given signal is evidence that the level-series
    result was substantially a trend-sharing artifact (both curves simply
    "jump early, then plateau") rather than genuine period-by-period
    coupling, and that the `diff` row is the more trustworthy number to
    report as evidence of an early-warning relationship.
    """
    rows = []
    for name in ("shap", "lime", "wasserstein"):
        for series_type in ("level", "diff"):
            lag_col = f"{name}_{series_type}_best_lag"
            corr_col = f"{name}_{series_type}_best_corr"
            p_col = f"{name}_{series_type}_perm_pvalue"
            valid_p = per_seed[p_col].dropna()
            rows.append(dict(
                signal=name,
                series_type=series_type,
                n_seeds=len(per_seed),
                n_seeds_valid=len(valid_p),
                mean_best_lag=per_seed[lag_col].mean(),
                mean_best_corr=per_seed[corr_col].mean(),
                median_perm_pvalue=valid_p.median() if len(valid_p) else np.nan,
                frac_seeds_significant_p05=(valid_p < 0.05).mean() if len(valid_p) else np.nan,
            ))
    return pd.DataFrame(rows)


def crossing_sensitivity_summary(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Aggregates the RELATIVE-threshold-crossing lead times across the
    fraction sweep (config.RELATIVE_CROSSING_FRACTIONS) for all three
    signals, with a paired t-test per (signal, fraction).

    This is the floor-effect-free counterpart to `summarize_leadlag`'s
    original absolute-threshold lead-time analysis, and sweeping the
    fraction (rather than picking one number) shows directly whether the
    lead/lag conclusion is sensitive to that choice.
    """
    rows = []
    for fraction in config.RELATIVE_CROSSING_FRACTIONS:
        perf_col = f"perf_relcross_f{fraction}"
        for name in ("shap", "lime", "wasserstein"):
            explain_col = f"{name}_relcross_f{fraction}"
            lead_col = f"{name}_relcross_f{fraction}_lead_time"
            valid = per_seed.dropna(subset=[lead_col])
            n = len(valid)
            if n >= 2:
                t_stat, p_val = stats.ttest_rel(valid[perf_col], valid[explain_col])
            else:
                t_stat, p_val = np.nan, np.nan
            rows.append(dict(
                signal=name,
                fraction=fraction,
                n_seeds_with_both_crossings=n,
                n_seeds_total=len(per_seed),
                mean_lead_time_windows=valid[lead_col].mean() if n else np.nan,
                std_lead_time_windows=valid[lead_col].std() if n else np.nan,
                paired_ttest_stat=t_stat,
                paired_ttest_pvalue=p_val,
            ))
    return pd.DataFrame(rows)


def regime_stability_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Paired comparison, across every (seed, window>0) pair, of ranking
    stability-vs-baseline between the FROZEN and continuously-RETRAINED
    regimes.

    This is the evidence for what post-review analysis of the real-data
    run found to be the strongest and most robust finding in the study: a
    frozen, silently-decaying model's SHAP/LIME ranking should NOT look
    MORE stable than a healthily-adapting retrained model's -- if it does,
    that is precisely the dangerous false-reassurance failure mode this
    paper investigates. Unlike every analysis above, this comparison does
    not depend on any arbitrary threshold choice at all: it is a direct
    paired test of ranking-stability-vs-baseline (Spearman rho and top-k
    Jaccard) between the two regimes.
    """
    d = df[df["window"] > 0]
    frozen = d[d["regime"] == "frozen"].sort_values(["seed", "window"]).reset_index(drop=True)
    retrained = d[d["regime"] == "retrained"].sort_values(["seed", "window"]).reset_index(drop=True)
    assert (frozen["seed"].values == retrained["seed"].values).all()
    assert (frozen["window"].values == retrained["window"].values).all()

    rows = []
    for metric in ["shap_spearman_rho", "lime_spearman_rho", "shap_topk_jaccard", "lime_topk_jaccard"]:
        diff = frozen[metric].values - retrained[metric].values
        t_stat, t_p = stats.ttest_rel(frozen[metric], retrained[metric])
        try:
            w_stat, w_p = stats.wilcoxon(diff)
        except ValueError:
            w_stat, w_p = np.nan, np.nan
        rows.append(dict(
            metric=metric,
            n_pairs=len(diff),
            frac_frozen_more_stable=float(np.mean(diff > 0)),
            mean_diff_frozen_minus_retrained=float(diff.mean()),
            std_diff=float(diff.std()),
            paired_ttest_stat=t_stat,
            paired_ttest_pvalue=t_p,
            wilcoxon_stat=w_stat,
            wilcoxon_pvalue=w_p,
        ))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _mean_std_by_window(df, regime, col):
    sub = df[df["regime"] == regime]
    g = sub.groupby("window")[col].agg(["mean", "std"]).reset_index()
    return g


def plot_performance_decay(df: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for regime, color in (("frozen", "tab:red"), ("retrained", "tab:blue")):
        g = _mean_std_by_window(df, regime, "perf_roc_auc")
        ax.plot(g["window"], g["mean"], marker="o", label=f"{regime} model", color=color)
        ax.fill_between(g["window"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2, color=color)
    ax.set_xlabel("Sliding window index (chronological)")
    ax.set_ylabel("ROC-AUC (mean ± std over seeds)")
    ax.set_title("Model performance over time: frozen vs. rolling-retrained")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_explanation_drift(df: pd.DataFrame, outpath: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, col, title in zip(axes, ["shap_spearman_drift", "lime_spearman_drift"],
                               ["SHAP global ranking drift", "LIME local ranking drift"]):
        for regime, color in (("frozen", "tab:red"), ("retrained", "tab:blue")):
            g = _mean_std_by_window(df, regime, col)
            ax.plot(g["window"], g["mean"], marker="o", label=f"{regime} model", color=color)
            ax.fill_between(g["window"], g["mean"] - g["std"], g["mean"] + g["std"], alpha=0.2, color=color)
        ax.set_xlabel("Sliding window index (chronological)")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("1 - Spearman(rank_t, rank_baseline)")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_lead_time_distribution(per_seed: pd.DataFrame, outpath: str):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    data = [per_seed["shap_lead_time_windows"].dropna().values,
            per_seed["lime_lead_time_windows"].dropna().values]
    labels = [f"SHAP (n={len(data[0])})", f"LIME (n={len(data[1])})"]
    bp = ax.boxplot([d if len(d) else [np.nan] for d in data], labels=labels, showmeans=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Lead time (windows) = perf-crossing - explanation-crossing\n(positive = explanation warns earlier)")
    ax.set_title(f"Early-warning lead time across {len(per_seed)} seeds")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_representative_ccf(df: pd.DataFrame, seed: int, outpath: str):
    """Cross-correlation of each drift/shift signal against performance
    drop, for one representative seed.

    Post-review update: now shows BOTH the level series (left panel) and
    the first-differenced series (right panel, shared trend/level removed)
    side by side, since the level version alone can substantially overstate
    significance when both curves share a "jump early, then plateau" shape
    (see correlation_robustness_summary for the seed-aggregated version of
    this same point). Also adds the Wasserstein covariate-shift signal
    alongside SHAP/LIME as a cheap-baseline comparison.
    """
    g = df[(df["regime"] == "frozen") & (df["seed"] == seed)].sort_values("window")
    baseline_auc = g.loc[g["window"] == config.BASELINE_WINDOW_INDEX, "perf_roc_auc"].iloc[0]
    perf_drop = (baseline_auc - g["perf_roc_auc"]).values

    signals = {
        "SHAP": (g["shap_spearman_drift"].values, "tab:purple"),
        "LIME": (g["lime_spearman_drift"].values, "tab:green"),
        "Wasserstein": (g["mean_feature_wasserstein_shift"].values, "tab:orange"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for label, (sig, color) in signals.items():
        ccf_level = drift_metrics.cross_correlation(sig, perf_drop, config.MAX_LAG_WINDOWS)
        axes[0].plot(ccf_level.index, ccf_level.values, marker="o", label=label, color=color)

        ccf_diff = drift_metrics.cross_correlation(
            drift_metrics.differenced(sig), drift_metrics.differenced(perf_drop),
            config.DIFFERENCED_MAX_LAG_WINDOWS,
        )
        axes[1].plot(ccf_diff.index, ccf_diff.values, marker="o", label=label, color=color)

    for ax, title in zip(axes, ["Level series\n(can overstate significance -- see correlation_robustness_summary)",
                                  "First-differenced series\n(shared trend removed; more trustworthy)"]):
        ax.axvline(0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("Lag (windows); positive = signal leads performance drop")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Cross-correlation")
    axes[0].legend()
    fig.suptitle(f"Explanation / covariate-shift drift vs. performance-drop cross-correlation (seed={seed})")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_relative_lead_time_distribution(per_seed: pd.DataFrame, outpath: str, fraction: float = None):
    """Boxplot of lead times under the RELATIVE-threshold crossing method
    (see crossing_sensitivity_summary) at one representative fraction,
    across all three signals (SHAP, LIME, Wasserstein) -- the floor-effect-
    free counterpart to plot_lead_time_distribution's absolute-threshold
    version."""
    fraction = fraction if fraction is not None else (
        config.RELATIVE_CROSSING_FRACTIONS[len(config.RELATIVE_CROSSING_FRACTIONS) // 2]
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cols = [f"{name}_relcross_f{fraction}_lead_time" for name in ("shap", "lime", "wasserstein")]
    data = [per_seed[c].dropna().values for c in cols]
    labels = [f"{name} (n={len(d)})" for name, d in zip(("SHAP", "LIME", "Wasserstein"), data)]
    ax.boxplot([d if len(d) else [np.nan] for d in data], labels=labels, showmeans=True)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("Lead time (windows) = perf-crossing - signal-crossing\n(positive = signal warns earlier)")
    ax.set_title(f"Relative-threshold (fraction={fraction}) early-warning lead time, {len(per_seed)} seeds")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_regime_stability(df: pd.DataFrame, outpath: str):
    """Boxplots of frozen vs. retrained SHAP/LIME rank-correlation-to-
    baseline, across all (seed, window>0) pairs -- the visual counterpart
    to regime_stability_comparison(), and the recommended headline figure
    given that comparison's robustness relative to the lead-lag analyses
    above."""
    d = df[df["window"] > 0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, col, title in zip(axes, ["shap_spearman_rho", "lime_spearman_rho"],
                               ["SHAP rank correlation to baseline", "LIME rank correlation to baseline"]):
        data = [d.loc[d["regime"] == "frozen", col].values,
                d.loc[d["regime"] == "retrained", col].values]
        ax.boxplot(data, labels=["frozen", "retrained"], showmeans=True)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("Spearman rho vs. baseline window")
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("A frozen, decaying model's explanation looks MORE stable\nthan a healthily-adapting retrained model's")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def run_full_analysis():
    df = load_results()
    per_seed = per_seed_leadlag(df)
    summary = summarize_leadlag(per_seed)

    per_seed_path = os.path.join(config.TABLES_DIR, "leadlag_per_seed.csv")
    per_seed.to_csv(per_seed_path, index=False)
    logger.info("Wrote per-seed lead-lag table -> %s", per_seed_path)

    summary_df = pd.DataFrame(summary).T
    summary_path = os.path.join(config.TABLES_DIR, "leadlag_summary.csv")
    summary_df.to_csv(summary_path)
    logger.info("Wrote aggregated summary (original absolute-threshold method) -> %s", summary_path)
    print("=== Original absolute-threshold lead-lag summary (kept for continuity) ===")
    print(summary_df.to_string())

    # --- Post-review robustness additions -----------------------------------
    corr_robust = correlation_robustness_summary(per_seed)
    corr_robust_path = os.path.join(config.TABLES_DIR, "correlation_robustness_summary.csv")
    corr_robust.to_csv(corr_robust_path, index=False)
    logger.info("Wrote correlation robustness summary (level vs. differenced) -> %s", corr_robust_path)
    print("\n=== Correlation robustness: level vs. differenced (trend-removed) ===")
    print(corr_robust.to_string(index=False))

    crossing_sens = crossing_sensitivity_summary(per_seed)
    crossing_sens_path = os.path.join(config.TABLES_DIR, "crossing_sensitivity_summary.csv")
    crossing_sens.to_csv(crossing_sens_path, index=False)
    logger.info("Wrote relative-threshold crossing sensitivity summary -> %s", crossing_sens_path)
    print("\n=== Relative-threshold crossing sensitivity (fraction sweep) ===")
    print(crossing_sens.to_string(index=False))

    regime_stability = regime_stability_comparison(df)
    regime_stability_path = os.path.join(config.TABLES_DIR, "regime_stability_comparison.csv")
    regime_stability.to_csv(regime_stability_path, index=False)
    logger.info("Wrote frozen-vs-retrained stability comparison -> %s", regime_stability_path)
    print("\n=== Frozen vs. retrained ranking stability ('false stability' evidence) ===")
    print(regime_stability.to_string(index=False))

    plot_performance_decay(df, os.path.join(config.FIGURES_DIR, "performance_decay.png"))
    plot_explanation_drift(df, os.path.join(config.FIGURES_DIR, "explanation_drift.png"))
    plot_lead_time_distribution(per_seed, os.path.join(config.FIGURES_DIR, "lead_time_distribution.png"))
    plot_relative_lead_time_distribution(
        per_seed, os.path.join(config.FIGURES_DIR, "relative_lead_time_distribution.png")
    )
    plot_regime_stability(df, os.path.join(config.FIGURES_DIR, "regime_stability.png"))
    representative_seed = int(per_seed["seed"].iloc[0])
    plot_representative_ccf(df, representative_seed,
                             os.path.join(config.FIGURES_DIR, f"ccf_seed_{representative_seed}.png"))
    logger.info("Wrote figures -> %s", config.FIGURES_DIR)

    return per_seed, summary_df


if __name__ == "__main__":
    run_full_analysis()
