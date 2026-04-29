#!/usr/bin/env python3
"""
Extract Accuracy, F1, Precision, Recall (macro avg) for baseline and random,
across all fractions and strategies (Stratified, LOSO, LOGO).
Values are averaged across folds within each strategy.
"""

import os
import re
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS_DIR = os.path.join(ROOT, "results")

FRACTIONS = ["30pct", "40pct", "50pct", "60pct", "70pct", "100pct"]
METHODS = {
    "baseline": ("all_results_baseline.csv", "classification_report_SDCNet"),
    "random":   ("all_results_random.csv",   "classification_report_Contrastive"),
}
STRATEGIES = ["Stratified", "LOSO", "LOGO"]


def parse_classification_report(filepath):
    """
    Parse a sklearn classification_report text file.
    Returns dict with macro avg precision, recall, f1-score.
    """
    with open(filepath, "r") as f:
        content = f.read()

    # Look for the macro avg line
    match = re.search(
        r"macro avg\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)",
        content
    )
    if match:
        return {
            "precision": float(match.group(1)),
            "recall":    float(match.group(2)),
            "f1":        float(match.group(3)),
        }
    return None


def load_fold_metrics(fraction, method_key):
    csv_name, report_prefix = METHODS[method_key]
    base_dir = os.path.join(RESULTS_DIR, fraction, method_key)
    csv_path = os.path.join(base_dir, csv_name)
    report_dir = os.path.join(base_dir, "classification_report")

    if not os.path.exists(csv_path):
        return None

    df = pd.read_csv(csv_path)
    records = []

    for _, row in df.iterrows():
        strategy = row["Strategy"]
        fold = row["Fold"]
        acc = row["Test_Acc"]
        f1_csv = row["Test_F1"]

        # Build report filename
        if strategy == "Stratified":
            report_fname = f"{report_prefix}_Stratified.txt"
        elif strategy == "LOSO":
            report_fname = f"{report_prefix}_LOSO_fold{fold}.txt"
        elif strategy == "LOGO":
            report_fname = f"{report_prefix}_LOGO_fold{fold}.txt"
        else:
            continue

        report_path = os.path.join(report_dir, report_fname)
        parsed = parse_classification_report(report_path) if os.path.exists(report_path) else None

        records.append({
            "strategy": strategy,
            "fold": fold,
            "accuracy": acc,
            "f1_csv": f1_csv,
            "precision": parsed["precision"] if parsed else np.nan,
            "recall":    parsed["recall"]    if parsed else np.nan,
            "f1_report": parsed["f1"]        if parsed else np.nan,
        })

    return pd.DataFrame(records)


def aggregate(df):
    """Average per strategy across folds."""
    out = {}
    for strategy in STRATEGIES:
        sub = df[df["strategy"] == strategy]
        if sub.empty:
            continue
        out[strategy] = {
            "accuracy":  sub["accuracy"].mean(),
            "f1":        sub["f1_csv"].mean(),
            "precision": sub["precision"].mean(),
            "recall":    sub["recall"].mean(),
            "n_folds":   len(sub),
        }
    return out


def main():
    rows = []
    for fraction in FRACTIONS:
        for method_key in ["baseline", "random"]:
            df = load_fold_metrics(fraction, method_key)
            if df is None:
                print(f"[WARN] Missing data: {fraction}/{method_key}")
                continue
            agg = aggregate(df)
            for strategy, vals in agg.items():
                rows.append({
                    "Fraction": fraction,
                    "Method": method_key,
                    "Strategy": strategy,
                    "N_folds": vals["n_folds"],
                    "Accuracy": round(vals["accuracy"], 4),
                    "F1_macro": round(vals["f1"], 4),
                    "Precision_macro": round(vals["precision"], 4),
                    "Recall_macro": round(vals["recall"], 4),
                })

    result_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Saved to: {out_path}\n")

    # Print formatted tables per strategy
    for strategy in STRATEGIES:
        sub = result_df[result_df["Strategy"] == strategy].copy()
        sub = sub.sort_values(["Fraction", "Method"])
        print("=" * 80)
        print(f"  {strategy}  (N folds: {sub['N_folds'].iloc[0] if len(sub)>0 else '?'})")
        print("=" * 80)
        print(f"  {'Fraction':<10} {'Method':<12} {'Accuracy':>10} {'Precision':>11} {'Recall':>9} {'F1':>9}")
        print(f"  {'-'*65}")
        for _, row in sub.iterrows():
            print(f"  {row['Fraction']:<10} {row['Method']:<12} "
                  f"{row['Accuracy']:>10.4f} {row['Precision_macro']:>11.4f} "
                  f"{row['Recall_macro']:>9.4f} {row['F1_macro']:>9.4f}")
        print()


if __name__ == "__main__":
    main()
