"""
Build several candidate submissions on the shift-pruned features.

Candidates:
  pruned_lgbm    LightGBM (balanced) on the pruned features
  pruned_et      ExtraTrees (balanced) on the pruned features
  pruned_blend   0.4 LightGBM + 0.6 ExtraTrees
  pruned_pseudo  blend, then one pseudo-labelling round on confident test rows
  pruned_prior   blend, reweighted by test class priors estimated with EM

Each file is written to the output folder. Agreement between candidates is
printed so near-duplicates can be skipped on the public board.
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

from train_intrusion_model import LABELS, RANDOM_STATE, load, make_model

LGB_TREES = 800
ET_TREES = 600
BLEND_LGB = 0.4
PSEUDO_CONFIDENCE = 0.9


def to_numeric(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.select_dtypes("category").columns:
        X[col] = X[col].cat.codes
    return X.fillna(-1)


def fit_lgbm(X, y, X_eval):
    model = make_model(LGB_TREES)
    model.fit(X, y)
    return model.predict_proba(X_eval)


def fit_et(X, y, X_eval):
    model = ExtraTreesClassifier(
        n_estimators=ET_TREES,
        max_features="sqrt",
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    model.fit(to_numeric(X), y)
    return model.predict_proba(to_numeric(X_eval))


def em_priors(P: np.ndarray, train_prior: np.ndarray, iters: int = 100) -> np.ndarray:
    """Saerens et al. (2002): estimate test class priors from posteriors."""
    pi = train_prior.copy()
    for _ in range(iters):
        adjusted = P * (pi / train_prior)
        adjusted /= adjusted.sum(1, keepdims=True)
        pi = adjusted.mean(0)
    return pi


def write(path: Path, test_ids, proba):
    pred = proba.argmax(1)
    pd.DataFrame({"id": test_ids, "Target": [LABELS[i] for i in pred]}).to_csv(path, index=False)
    counts = np.bincount(pred, minlength=4)
    print(f"Wrote {path.name}: {dict(zip(LABELS, counts))}", flush=True)
    return pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dropped", required=True, help="Python list of pruned feature names")
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    X_full = pd.read_pickle(args.cache_dir / "X_rebuilt.pkl")
    X_test_full = pd.read_pickle(args.cache_dir / "X_test_rebuilt.pkl")
    dropped = set(ast.literal_eval(args.dropped))
    keep = [c for c in X_full.columns if c not in dropped]
    X, X_test = X_full[keep], X_test_full[keep]
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    print(f"Pruned feature count: {len(keep)}", flush=True)

    lgb_p = fit_lgbm(X, y, X_test)
    et_p = fit_et(X, y, X_test)
    blend_p = BLEND_LGB * lgb_p + (1 - BLEND_LGB) * et_p

    write(args.out_dir / "candidate_pruned_lgbm.csv", test["id"], lgb_p)
    write(args.out_dir / "candidate_pruned_et.csv", test["id"], et_p)
    write(args.out_dir / "candidate_pruned_blend.csv", test["id"], blend_p)

    confident = blend_p.max(1) >= PSEUDO_CONFIDENCE
    pseudo_y = blend_p.argmax(1)[confident]
    print(f"Pseudo-label round: adding {confident.sum()} confident test rows", flush=True)
    X_aug = pd.concat([X, X_test[confident]], ignore_index=True)
    y_aug = np.r_[y, pseudo_y]
    pseudo_p = fit_lgbm(X_aug, y_aug, X_test)
    pseudo_blend = BLEND_LGB * pseudo_p + (1 - BLEND_LGB) * fit_et(X_aug, y_aug, X_test)
    write(args.out_dir / "candidate_pruned_pseudo.csv", test["id"], pseudo_blend)

    train_prior = np.full(len(LABELS), 1 / len(LABELS))
    test_prior = em_priors(blend_p, train_prior)
    print(f"EM test prior estimate: {dict(zip(LABELS, test_prior.round(3)))}", flush=True)
    prior_p = blend_p * (test_prior / train_prior)
    write(args.out_dir / "candidate_pruned_prior.csv", test["id"], prior_p)

    names = ["pruned_lgbm", "pruned_et", "pruned_blend", "pruned_pseudo", "pruned_prior"]
    preds = {
        "pruned_lgbm": lgb_p.argmax(1),
        "pruned_et": et_p.argmax(1),
        "pruned_blend": blend_p.argmax(1),
        "pruned_pseudo": pseudo_blend.argmax(1),
        "pruned_prior": prior_p.argmax(1),
    }
    print("\nAgreement between candidates (%):")
    for a in names:
        print(a.ljust(15), " ".join(f"{(preds[a] == preds[b]).mean() * 100:6.2f}" for b in names))


if __name__ == "__main__":
    main()
