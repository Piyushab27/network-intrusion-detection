"""
Reduce the train/test shift caused by flow-duration artifacts.

Train's per-second rates are dominated by flows shorter than a millisecond, so
they saturate at fixed values. Recomputing the rates with a 1 ms duration floor
puts train and test on a comparable scale. This script compares adversarial
AUC (how easily a model tells train from test) and in-distribution random CV
for the original and the rebuilt features.
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, load, make_model, prepare

DURATION_FLOOR = 1e-3
RATE_COLS = [
    "fwd_pkts_per_sec",
    "bwd_pkts_per_sec",
    "flow_pkts_per_sec",
    "payload_bytes_per_second",
]


def add_clipped_rates(X: pd.DataFrame) -> pd.DataFrame:
    X = X.drop(columns=[c for c in RATE_COLS if c in X.columns])
    duration = X["flow_duration"].clip(lower=DURATION_FLOOR)
    X["c_fwd_pps"] = X["fwd_pkts_tot"] / duration
    X["c_bwd_pps"] = X["bwd_pkts_tot"] / duration
    X["c_pps"] = (X["fwd_pkts_tot"] + X["bwd_pkts_tot"]) / duration
    X["c_payload_bps"] = X["flow_pkts_payload.tot"] / duration
    return X


def adversarial_auc(X: pd.DataFrame, X_test: pd.DataFrame):
    X_all = pd.concat([X, X_test], ignore_index=True)
    is_test = np.r_[np.zeros(len(X), dtype=int), np.ones(len(X_test), dtype=int)]
    oof = np.zeros(len(X_all))
    gain = pd.Series(0.0, index=X_all.columns)
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X_all, is_test):
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
    auc = roc_auc_score(is_test, oof)
    top = (gain / gain.sum()).sort_values(ascending=False).head(8).round(4)
    return auc, top, oof[: len(X)]


def random_cv(X: pd.DataFrame, y: np.ndarray, trees: int = 400):
    oof = np.zeros((len(X), len(LABELS)))
    for tr, va in StratifiedKFold(n_splits=5, shuffle=True, random_state=42).split(X, y):
        model = make_model(trees)
        model.fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])
    pred = oof.argmax(1)
    return f1_score(y, pred, average="macro"), f1_score(y, pred, average=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)
    Xn, Xn_test = add_clipped_rates(X), add_clipped_rates(X_test)

    auc_old, top_old, _ = adversarial_auc(X, X_test)
    print(f"Original features: adversarial AUC = {auc_old:.4f}")
    print(top_old.to_string(), flush=True)

    auc_new, top_new, p_new = adversarial_auc(Xn, Xn_test)
    print(f"\nRebuilt features:  adversarial AUC = {auc_new:.4f}")
    print(top_new.to_string(), flush=True)

    macro, per_class = random_cv(Xn, y)
    print(f"\nRebuilt features, random CV (400 trees): macro F1 = {macro:.4f}")
    print({LABELS[i]: round(v, 4) for i, v in enumerate(per_class)}, flush=True)

    Xn.to_pickle(args.cache_dir / "X_rebuilt.pkl")
    Xn_test.to_pickle(args.cache_dir / "X_test_rebuilt.pkl")
    np.save(args.cache_dir / "p_test_like_rebuilt.npy", p_new)
    print(f"\nSaved rebuilt features to {args.cache_dir}")


if __name__ == "__main__":
    main()
