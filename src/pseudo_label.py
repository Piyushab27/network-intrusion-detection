"""
Pseudo-labelling for the shifted test set.

Train on the labelled train rows, label the test rows the model is most sure
about, add them to training, and refit. The labels come only from the model's
own predictions, never from the hidden test labels.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from train_intrusion_model import LABELS, load, make_model, prepare

TREES = 800
ROUNDS = 2
CONFIDENCE = 0.9


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_pseudo.csv"))
    parser.add_argument("--confidence", type=float, default=CONFIDENCE)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    args = parser.parse_args()

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)

    model = make_model(TREES)
    model.fit(X, y)
    proba = model.predict_proba(X_test)
    base_pred = proba.argmax(1)
    print(f"Round 0 (train only): test predicted counts {np.bincount(base_pred, minlength=4)}")

    for r in range(1, args.rounds + 1):
        pick = proba.max(1) >= args.confidence
        pseudo_y = proba.argmax(1)[pick]
        X_aug = pd.concat([X, X_test[pick]], ignore_index=True)
        y_aug = np.r_[y, pseudo_y]

        model = make_model(TREES)
        model.fit(X_aug, y_aug)
        proba = model.predict_proba(X_test)
        pred = proba.argmax(1)
        print(
            f"Round {r}: added {pick.sum()} confident test rows "
            f"{np.bincount(pseudo_y, minlength=4)}; "
            f"test predicted counts {np.bincount(pred, minlength=4)}; "
            f"changed vs round 0: {(pred != base_pred).sum()}",
            flush=True,
        )

    submission = pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in proba.argmax(1)]})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")


if __name__ == "__main__":
    main()
