#!/usr/bin/env python3
"""
Statistical analysis: paired t-test, Wilcoxon signed-rank test, 95% CI
between Baseline SDCNet and all contrastive methods across all data fractions.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import shapiro, ttest_rel, wilcoxon, norm
import warnings
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(BASE, "..")
RESULTS_DIR = os.path.join(ROOT, "results")

FRACTIONS = ["30pct", "40pct", "50pct", "60pct", "70pct", "100pct"]
METHODS = ["random", "random_shift", "semihard", "semihard_shift"]
STRATEGIES = ["LOSO", "LOGO"]  # Stratified has only 1 fold → no paired test possible


def load_results(fraction, method):
    """Load result CSV for a given fraction and method."""
    if method == "baseline":
        path = os.path.join(RESULTS_DIR, fraction, "baseline", "all_results_baseline.csv")
    else:
        path = os.path.join(RESULTS_DIR, fraction, method, f"all_results_{method}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df


def extract_fold_accs(df, strategy):
    """Extract per-fold Test_Acc for LOSO or LOGO."""
    sub = df[df["Strategy"] == strategy].copy()
    sub = sub.sort_values("Fold")
    return sub["Test_Acc"].values


def mean_ci95(values):
    """Return (mean, lower_95ci, upper_95ci) using normal approximation."""
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf(0.975, df=n - 1)
    return mean, mean - h, mean + h


def cohens_d(a, b):
    """Paired Cohen's d for matched samples."""
    diff = np.array(a) - np.array(b)
    return np.mean(diff) / (np.std(diff, ddof=1) + 1e-12)


def run_analysis():
    records = []

    for fraction in FRACTIONS:
        df_base = load_results(fraction, "baseline")
        if df_base is None:
            print(f"  [WARN] Missing baseline for {fraction}")
            continue

        for method in METHODS:
            df_method = load_results(fraction, method)
            if df_method is None:
                print(f"  [WARN] Missing {method} for {fraction}")
                continue

            for strategy in STRATEGIES:
                base_accs = extract_fold_accs(df_base, strategy)
                method_accs = extract_fold_accs(df_method, strategy)

                n = min(len(base_accs), len(method_accs))
                if n < 2:
                    continue
                base_accs = base_accs[:n]
                method_accs = method_accs[:n]

                diff = method_accs - base_accs
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1)
                se_diff = stats.sem(diff)
                ci_h = se_diff * stats.t.ppf(0.975, df=n - 1)
                ci_lo = mean_diff - ci_h
                ci_hi = mean_diff + ci_h

                # Normality test on differences (Shapiro-Wilk, valid for n<50)
                if n >= 3:
                    _, shapiro_p = shapiro(diff)
                    normal = shapiro_p > 0.05
                else:
                    shapiro_p = np.nan
                    normal = True  # assume normal for n<3

                # Paired t-test
                t_stat, t_pval = ttest_rel(method_accs, base_accs)

                # Wilcoxon signed-rank test (requires n >= 1 non-zero differences)
                nonzero_diff = diff[diff != 0]
                if len(nonzero_diff) >= 1:
                    try:
                        w_stat, w_pval = wilcoxon(diff, alternative='two-sided')
                    except Exception:
                        w_stat, w_pval = np.nan, np.nan
                else:
                    w_stat, w_pval = np.nan, np.nan

                d = cohens_d(method_accs, base_accs)

                mean_base, ci_lo_base, ci_hi_base = mean_ci95(base_accs)
                mean_method, ci_lo_method, ci_hi_method = mean_ci95(method_accs)

                records.append({
                    "Fraction": fraction,
                    "Method": method,
                    "Strategy": strategy,
                    "N_folds": n,
                    "Mean_Baseline": round(mean_base, 4),
                    "CI95_Baseline": f"[{ci_lo_base:.4f}, {ci_hi_base:.4f}]",
                    "Mean_Method": round(mean_method, 4),
                    "CI95_Method": f"[{ci_lo_method:.4f}, {ci_hi_method:.4f}]",
                    "Mean_Diff": round(mean_diff, 4),
                    "Std_Diff": round(std_diff, 4),
                    "CI95_Diff": f"[{ci_lo:.4f}, {ci_hi:.4f}]",
                    "t_stat": round(t_stat, 4),
                    "t_pval": round(t_pval, 6),
                    "Shapiro_p": round(shapiro_p, 4) if not np.isnan(shapiro_p) else "n/a",
                    "Normal_diff": normal,
                    "W_stat": round(w_stat, 4) if not np.isnan(w_stat) else "n/a",
                    "W_pval": round(w_pval, 6) if not np.isnan(w_pval) else "n/a",
                    "Cohens_d": round(d, 4),
                    "Sig_t (p<0.05)": "YES" if t_pval < 0.05 else "no",
                    "Sig_W (p<0.05)": ("YES" if (not isinstance(w_pval, str) and w_pval < 0.05) else "no"),
                })

    df_out = pd.DataFrame(records)
    out_path = os.path.join(ROOT, "results", "statistical_results.csv")
    df_out.to_csv(out_path, index=False)
    print(f"\nSaved full results to: {out_path}")
    return df_out


def print_summary(df):
    print("\n" + "=" * 100)
    print("STATISTICAL ANALYSIS — Paired t-test & Wilcoxon signed-rank test")
    print("Comparing each contrastive method vs Baseline SDCNet per fraction and strategy")
    print("=" * 100)

    for strategy in STRATEGIES:
        print(f"\n{'─'*100}")
        print(f"  STRATEGY: {strategy}")
        print(f"{'─'*100}")
        sub = df[df["Strategy"] == strategy]

        print(f"\n  {'Fraction':<10} {'Method':<18} {'N':<4} {'Base Acc':>10} {'Method Acc':>12} "
              f"{'Mean Δ':>9} {'95% CI Δ':>22} {'t-stat':>8} {'t-pval':>10} "
              f"{'Normal?':>8} {'W-pval':>10} {'Cohen d':>9} {'Sig?':>6}")
        print(f"  {'─'*170}")

        for _, row in sub.sort_values(["Fraction", "Method"]).iterrows():
            sig = "***" if (row["t_pval"] < 0.001) else ("**" if row["t_pval"] < 0.01
                  else ("*" if row["t_pval"] < 0.05 else "ns"))
            print(f"  {row['Fraction']:<10} {row['Method']:<18} {row['N_folds']:<4} "
                  f"{row['Mean_Baseline']:>10.4f} {row['Mean_Method']:>12.4f} "
                  f"{row['Mean_Diff']:>+9.4f} {row['CI95_Diff']:>22} "
                  f"{row['t_stat']:>8.3f} {row['t_pval']:>10.6f} "
                  f"{'Yes' if row['Normal_diff'] else 'No':>8} "
                  f"{str(row['W_pval']):>10} "
                  f"{row['Cohens_d']:>+9.3f} {sig:>6}")


def print_markdown_tables(df):
    """Print analysis as markdown-ready tables for insertion in RESULTS_ANALYSIS.md"""
    print("\n\n" + "=" * 100)
    print("MARKDOWN OUTPUT — Copy into RESULTS_ANALYSIS.md")
    print("=" * 100)

    for strategy in STRATEGIES:
        sub = df[df["Strategy"] == strategy]
        n_folds = sub["N_folds"].iloc[0] if len(sub) > 0 else "?"
        print(f"\n### {strategy} — N={n_folds} folds per comparison\n")
        print(f"| Fraction | Method | Mean Baseline | 95% CI (Baseline) | Mean Method | 95% CI (Method) | "
              f"Mean Δ | 95% CI (Δ) | t-stat | t p-value | Normal? | W p-value | Cohen's d | Sig |")
        print(f"|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for _, row in sub.sort_values(["Fraction", "Method"]).iterrows():
            sig = "***" if (row["t_pval"] < 0.001) else ("**" if row["t_pval"] < 0.01
                  else ("*" if row["t_pval"] < 0.05 else "ns"))
            norm_str = "Yes" if row["Normal_diff"] else "No"
            w_pval = f"{row['W_pval']:.6f}" if isinstance(row['W_pval'], float) else str(row['W_pval'])
            print(f"| {row['Fraction']} | {row['Method']} | {row['Mean_Baseline']:.4f} | {row['CI95_Baseline']} | "
                  f"{row['Mean_Method']:.4f} | {row['CI95_Method']} | "
                  f"{row['Mean_Diff']:+.4f} | {row['CI95_Diff']} | "
                  f"{row['t_stat']:.3f} | {row['t_pval']:.6f} | {norm_str} | {w_pval} | "
                  f"{row['Cohens_d']:+.3f} | {sig} |")


if __name__ == "__main__":
    print("Loading CSV files and running statistical tests...")
    df_results = run_analysis()
    print_summary(df_results)
    print_markdown_tables(df_results)
    print("\nDone.")
