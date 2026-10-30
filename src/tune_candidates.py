"""
Apply the same class-multiplier tuning to every CV candidate, so they compare
like for like with the uploaded baseline (which used tuned multipliers).

For each candidate, from its cached out-of-fold (OOF) and test probabilities:
  - argmax OOF Macro F1
  - tuned OOF Macro F1 (multipliers fitted on all OOF rows, in-sample)
  - nested OOF Macro F1: each fold's multipliers are fitted on the other four
    folds' OOF rows and applied to that fold only (no in-sample optimism)
  - a tuned test submission written to the output folder

The baseline should reproduce submission_random_cv.csv byte for byte.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load

NAMES = [
    "original_800",
    "rebuilt_k0",
    "pruned_k5",
    "pruned_k10",
    "pruned_k15",
    "pruned_k20",
    "pruned_k25",
    "rebuilt_w_mod",
    "rebuilt_pseudo",
]


def fast_macro_f1(y, pred, k=4):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return f1.mean()


def tune(proba, y, rounds=3):
    """Same coordinate search as train_intrusion_model.tune_class_weights."""
    weights = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(-1.5, 1.5, 31))
    best = fast_macro_f1(y, (proba * weights).argmax(1))
    for _ in range(rounds):
        for c in range(proba.shape[1]):
            for g in grid:
                trial = weights.copy()
                trial[c] = g
                score = fast_macro_f1(y, (proba * trial).argmax(1))
                if score > best:
                    best, weights = score, trial
    return weights, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    folds = list(
        StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(
            np.zeros(len(y)), y
        )
    )

    rows = []
    for name in NAMES:
        data = np.load(args.cache_dir / f"cv_{name}.npz")
        oof, test_proba = data["oof"], data["test"]
        argmax_f1 = fast_macro_f1(y, oof.argmax(1))
        weights, tuned_f1 = tune(oof, y)

        nested_pred = np.zeros(len(y), dtype=int)
        for tr, va in folds:
            w_k, _ = tune(oof[tr], y[tr])
            nested_pred[va] = (oof[va] * w_k).argmax(1)
        nested_f1 = fast_macro_f1(y, nested_pred)

        test_pred = (test_proba * weights).argmax(1)
        out_path = args.out_dir / f"tuned_{name}.csv"
        pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in test_pred]}).to_csv(
            out_path, index=False
        )
        rows.append(
            {
                "name": name,
                "argmax_oof": round(argmax_f1, 5),
                "tuned_oof_insample": round(tuned_f1, 5),
                "tuned_oof_nested": round(nested_f1, 5),
                "weights": np.round(weights, 3).tolist(),
                **{f"test_{LABELS[i]}": round((test_pred == i).mean(), 4) for i in range(4)},
            }
        )
        print(f"{name:<15} argmax={argmax_f1:.5f} tuned={tuned_f1:.5f} nested={nested_f1:.5f} weights={np.round(weights, 3).tolist()}", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "tuned_candidate_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
