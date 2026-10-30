"""
Adversarial validation: can a model tell train rows from test rows?

A high AUC means the test set is shifted from train. The out-of-fold
"test-likeness" of each train row is saved so later steps can weight or
hold out the train rows that look most like test.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, load, prepare


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)
    X_all = pd.concat([X, X_test], ignore_index=True)
    is_test = np.r_[np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)]

    oof = np.zeros(len(X_all))
    gain = pd.Series(0.0, index=X_all.columns)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (tr, va) in enumerate(skf.split(X_all, is_test), start=1):
        model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(X_all.iloc[tr], is_test[tr])
        oof[va] = model.predict_proba(X_all.iloc[va])[:, 1]
        gain += pd.Series(model.booster_.feature_importance("gain"), index=X_all.columns)
        print(f"fold {fold} done", flush=True)

    auc = roc_auc_score(is_test, oof)
    print(f"\nAdversarial AUC (train vs test): {auc:.4f}")
    print("\nTop features separating train from test (gain share):")
    print((gain / gain.sum()).sort_values(ascending=False).head(15).round(4).to_string())

    p_train = oof[: len(X)]
    np.save(args.cache_dir / "p_test_like.npy", p_train)
    print("\nTest-likeness of train rows (OOF P(test)):")
    print(pd.Series(p_train).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]).round(4).to_string())

    print("\nClass counts among train rows, by test-likeness:")
    for thr in [0.1, 0.25, 0.5]:
        mask = p_train >= thr
        counts = pd.Series(y[mask]).map(dict(enumerate(LABELS))).value_counts().to_dict()
        print(f"  P(test) >= {thr}: {mask.sum()} rows {counts}")


if __name__ == "__main__":
    main()
