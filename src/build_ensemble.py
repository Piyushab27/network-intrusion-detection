"""
Pre-specified probability-average ensemble with nested evaluation.

Members are fixed in advance and are not chosen by searching subsets, so the
ensemble's CV score is not inflated by member selection. Every member's
out-of-fold (OOF) predictions come from the same random 5-fold split, so the
averaged OOF predictions are still out-of-fold.

Members:
  original_800   LightGBM, original features, 800 trees (the baseline model)
  original_pseudo LightGBM, original features, one pseudo round (the recipe of
                 submission_pseudo_1round.csv, public 0.84585)
  original_et    ExtraTrees, original features (new in this script)
  rebuilt_k0     LightGBM, rebuilt rate features
  pruned_k20     LightGBM, rebuilt features minus the first 20 shift-pruned features
  rebuilt_w_mod  LightGBM, rebuilt features with clipped domain-importance weights
  rebuilt_pseudo LightGBM, rebuilt features with in-fold pseudo-labelling
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load, prepare
from tune_candidates import fast_macro_f1, tune

MEMBERS = [
    "original_800",
    "original_pseudo",
    "original_et",
    "rebuilt_k0",
    "pruned_k20",
    "rebuilt_w_mod",
    "rebuilt_pseudo",
]


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].cat.codes
    return X.fillna(-1)


def make_et():
    return ExtraTreesClassifier(
        n_estimators=600,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def run_cv(X, X_test, y, folds):
    X_num, X_test_num = to_numeric(X), to_numeric(X_test)
    oof = np.zeros((len(X), len(LABELS)))
    test_proba = np.zeros((len(X_test), len(LABELS)))
    for tr, va in folds:
        model = make_et()
        model.fit(X_num.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X_num.iloc[va])
        test_proba += model.predict_proba(X_test_num) / N_FOLDS
    return oof, test_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    X_orig, X_test_orig, _ = prepare(train, test)
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y)), y
        )
    )

    et_path = args.cache_dir / "cv_original_et.npz"
    if et_path.exists():
        data = np.load(et_path)
        et_oof, et_test = data["oof"], data["test"]
    else:
        et_oof, et_test = run_cv(X_orig, X_test_orig, y, folds)
        np.savez(et_path, oof=et_oof, test=et_test)

    members = {}
    for name in MEMBERS:
        path = args.cache_dir / f"cv_{name}.npz"
        if name == "original_et":
            members[name] = {"oof": et_oof, "test": et_test}
        else:
            data = np.load(path)
            members[name] = {"oof": data["oof"], "test": data["test"]}

    print("Member scores (nested, each tuned on the other folds):")
    member_rows = []
    for name in MEMBERS:
        oof = members[name]["oof"]
        nested_pred = np.zeros(len(y), dtype=int)
        for tr, va in folds:
            w_k, _ = tune(oof[tr], y[tr])
            nested_pred[va] = (oof[va] * w_k).argmax(1)
        nested = fast_macro_f1(y, nested_pred)
        member_rows.append({"name": name, "argmax_oof": round(fast_macro_f1(y, oof.argmax(1)), 5), "nested": round(nested, 5)})
        print(f"  {name:<15} argmax={member_rows[-1]['argmax_oof']:.5f} nested={nested:.5f}", flush=True)

    oof_ens = np.mean([members[m]["oof"] for m in MEMBERS], axis=0)
    test_ens = np.mean([members[m]["test"] for m in MEMBERS], axis=0)

    argmax_f1 = fast_macro_f1(y, oof_ens.argmax(1))
    weights, tuned_f1 = tune(oof_ens, y)
    nested_pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        w_k, _ = tune(oof_ens[tr], y[tr])
        nested_pred[va] = (oof_ens[va] * w_k).argmax(1)
    nested_f1 = fast_macro_f1(y, nested_pred)

    test_pred = (test_ens * weights).argmax(1)
    out_path = args.out_dir / "tuned_final_ensemble.csv"
    pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in test_pred]}).to_csv(out_path, index=False)

    print(f"\nENSEMBLE argmax={argmax_f1:.5f} tuned={tuned_f1:.5f} nested={nested_f1:.5f} weights={np.round(weights, 3).tolist()}")
    print("test predicted share:", {LABELS[i]: round((test_pred == i).mean(), 4) for i in range(4)})
    print(f"Wrote {out_path.name}")


if __name__ == "__main__":
    main()
