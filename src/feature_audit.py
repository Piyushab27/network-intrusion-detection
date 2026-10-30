"""
Per-feature audit of train versus test, plus relevance to the target.

For each rebuilt feature:
  - numeric: two-sample KS statistic (train vs test), missing rates
  - categorical: total variation distance between category frequencies
  - relevance: mutual information with Target on train (numeric features,
    30,000-row sample, fixed seed)
Only train labels are used for relevance. Test labels do not exist.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif

from train_intrusion_model import LABELS, load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("feature_audit.csv"))
    args = parser.parse_args()

    train, _ = load(args.data_dir)
    X = pd.read_pickle(args.cache_dir / "X_rebuilt.pkl")
    X_test = pd.read_pickle(args.cache_dir / "X_test_rebuilt.pkl")
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()

    rows = []
    for col in X.columns:
        train_missing, test_missing = X[col].isna().mean(), X_test[col].isna().mean()
        if isinstance(X[col].dtype, pd.CategoricalDtype):
            tr_freq = X[col].astype(str).value_counts(normalize=True)
            te_freq = X_test[col].astype(str).value_counts(normalize=True)
            cats = tr_freq.index.union(te_freq.index)
            tv = 0.5 * (tr_freq.reindex(cats, fill_value=0) - te_freq.reindex(cats, fill_value=0)).abs().sum()
            rows.append((col, "categorical", tv, np.nan, train_missing, test_missing))
        else:
            stat, pval = ks_2samp(X[col].dropna(), X_test[col].dropna())
            rows.append((col, "numeric", stat, pval, train_missing, test_missing))

    audit = pd.DataFrame(
        rows,
        columns=["feature", "type", "shift_stat", "p_value", "train_missing", "test_missing"],
    )

    numeric = [c for c in X.columns if not isinstance(X[c].dtype, pd.CategoricalDtype)]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(X), size=min(30000, len(X)), replace=False)
    mi = mutual_info_classif(
        X[numeric].iloc[idx].fillna(-1).to_numpy(), y[idx], random_state=42
    )
    audit = audit.merge(
        pd.DataFrame({"feature": numeric, "target_mi": mi}), on="feature", how="left"
    )
    audit.sort_values("shift_stat", ascending=False).to_csv(args.out, index=False)

    print("Top 25 features by train/test shift (KS for numeric, TV for categorical):")
    print(audit.sort_values("shift_stat", ascending=False).head(25).round(4).to_string(index=False))
    print("\nFeatures with missing values in either set:")
    print(audit[(audit.train_missing > 0) | (audit.test_missing > 0)].to_string(index=False) or "none")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
