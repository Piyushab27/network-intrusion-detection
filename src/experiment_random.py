"""
Random-CV experiment: blend LightGBM with ExtraTrees to improve Macro F1.

Each model's out-of-fold (OOF) and test probabilities are cached in --cache-dir,
so blend weights can be searched without retraining.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import (
    LABELS,
    N_FOLDS,
    RANDOM_STATE,
    load,
    make_model,
    prepare,
    tune_class_weights,
)

def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """ExtraTrees needs numeric input: category codes and no NaNs."""
    X = X.copy()
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].cat.codes
    return X.fillna(-1)


def make_et(n_trees: int) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=n_trees,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def oof_and_test(name, build, X, X_test, y, folds, cache_dir: Path):
    cache = cache_dir / f"{name}.npz"
    if cache.exists():
        data = np.load(cache)
        print(f"Loaded cached {name}")
        return data["oof"], data["test"]

    oof = np.zeros((len(X), len(LABELS)))
    test_proba = np.zeros((len(X_test), len(LABELS)))
    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = build()
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])
        test_proba += model.predict_proba(X_test) / N_FOLDS
        print(f"{name} fold {fold} done", flush=True)
    np.savez(cache, oof=oof, test=test_proba)
    return oof, test_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_blend.csv"))
    parser.add_argument("--lgb-trees", type=int, default=800)
    parser.add_argument("--et-trees", type=int, default=400)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X, y)
    )

    lgb_oof, lgb_test = oof_and_test(
        "lgbm", lambda: make_model(args.lgb_trees), X, X_test, y, folds, args.cache_dir
    )
    X_num, X_test_num = to_numeric(X), to_numeric(X_test)
    et_oof, et_test = oof_and_test(
        "extra_trees", lambda: make_et(args.et_trees), X_num, X_test_num, y, folds, args.cache_dir
    )

    print("\nBlend search (argmax OOF macro F1, w = weight on LightGBM):")
    best_w, best_f1 = 1.0, -1.0
    for w in np.linspace(0, 1, 11):
        f1 = f1_score(y, (w * lgb_oof + (1 - w) * et_oof).argmax(1), average="macro")
        print(f"  w={w:.1f}  macro F1={f1:.4f}")
        if f1 > best_f1:
            best_w, best_f1 = w, f1

    oof = best_w * lgb_oof + (1 - best_w) * et_oof
    test_proba = best_w * lgb_test + (1 - best_w) * et_test
    weights, tuned_f1 = tune_class_weights(oof, y)
    print(f"\nBest w={best_w:.1f}: argmax={best_f1:.4f} tuned={tuned_f1:.4f} weights={np.round(weights, 3)}")
    print("\n" + classification_report(y, (oof * weights).argmax(1), target_names=LABELS, digits=4))

    preds = (test_proba * weights).argmax(1)
    submission = pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in preds]})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(submission["Target"].value_counts())


if __name__ == "__main__":
    main()
