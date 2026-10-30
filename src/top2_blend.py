"""
Probability blend of the two best public recipes.

Averages the out-of-fold (OOF) and test probabilities of:
  original_800     tuned LightGBM (evaluate_candidates.py)
  original_pseudo  one-round pseudo-labelling recipe (pseudo_recipe_cv.py)
then tunes class multipliers on the averaged OOF and writes
submission_top2_blend.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load
from tune_candidates import fast_macro_f1, tune


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_top2_blend.csv"))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y)), y
        )
    )

    a = np.load(args.cache_dir / "cv_original_800.npz")
    b = np.load(args.cache_dir / "cv_original_pseudo.npz")
    oof = (a["oof"] + b["oof"]) / 2
    test_proba = (a["test"] + b["test"]) / 2

    weights, _ = tune(oof, y)
    nested_pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        w_k, _ = tune(oof[tr], y[tr])
        nested_pred[va] = (oof[va] * w_k).argmax(1)
    print(f"argmax OOF macro F1 {fast_macro_f1(y, oof.argmax(1)):.5f}, nested {fast_macro_f1(y, nested_pred):.5f}")

    pred = (test_proba * weights).argmax(1)
    pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in pred]}).to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
