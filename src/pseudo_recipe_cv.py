"""
Random 5-fold CV of the recipe behind submission_pseudo_1round.csv, the file
that scored 0.84585 on the public board.

Recipe (from pseudo_label.py):
  LightGBM, balanced, 800 trees, original features, argmax (no multipliers)
  one pseudo-labelling round: test rows with P >= 0.9 added with weight 1

Inside each fold, pseudo-labels come only from that fold's training model, so
validation labels never reach them. Test predictions average the fold models.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from tune_candidates import fast_macro_f1
from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load, make_model, prepare

TREES = 800
CONFIDENCE = 0.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    X, X_test, _ = prepare(train, test)
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y)), y
        )
    )

    oof = np.zeros((len(X), len(LABELS)))
    test_proba = np.zeros((len(X_test), len(LABELS)))
    fold_scores = []
    for fold, (tr, va) in enumerate(folds, start=1):
        model = make_model(TREES)
        model.fit(X.iloc[tr], y[tr])
        p_test = model.predict_proba(X_test)

        confident = p_test.max(1) >= CONFIDENCE
        X_aug = pd.concat([X.iloc[tr], X_test[confident]], ignore_index=True)
        y_aug = np.r_[y[tr], p_test.argmax(1)[confident]]
        model = make_model(TREES)
        model.fit(X_aug, y_aug)

        oof[va] = model.predict_proba(X.iloc[va])
        test_proba += model.predict_proba(X_test) / N_FOLDS
        fold_scores.append(fast_macro_f1(y[va], oof[va].argmax(1)))
        print(f"fold {fold}: added {confident.sum()} pseudo rows, macro F1 {fold_scores[-1]:.5f}", flush=True)

    np.savez(args.cache_dir / "cv_original_pseudo.npz", oof=oof, test=test_proba)

    pred = oof.argmax(1)
    print(f"\nPseudo recipe, argmax OOF macro F1 = {fast_macro_f1(y, pred):.5f}")
    print(f"fold mean {np.mean(fold_scores):.5f}, std {np.std(fold_scores):.5f}")
    print("test predicted share:", {LABELS[i]: round((test_proba.argmax(1) == i).mean(), 4) for i in range(4)})


if __name__ == "__main__":
    main()
