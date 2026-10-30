"""
Shift-invariant feature pruning.

Repeatedly drop the feature that most separates train from test (highest
adversarial gain) until a model can no longer tell them apart well. Then check
random CV on the remaining features and write a candidate submission.
"""

import argparse
from pathlib import Path

import pandas as pd

from feature_shift import adversarial_auc, random_cv
from train_intrusion_model import LABELS, load, make_model

TARGET_AUC = 0.6
MAX_DROPS = 25
TREES = 800


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_shift_pruned.csv"))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    X = pd.read_pickle(args.cache_dir / "X_rebuilt.pkl")
    X_test = pd.read_pickle(args.cache_dir / "X_test_rebuilt.pkl")
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()

    kept = list(X.columns)
    dropped = []
    for _ in range(MAX_DROPS):
        auc, top, _ = adversarial_auc(X[kept], X_test[kept])
        print(f"features={len(kept)} adversarial AUC={auc:.4f} next drop={top.index[0]} ({top.iloc[0]:.3f})", flush=True)
        if auc <= TARGET_AUC:
            break
        dropped.append(top.index[0])
        kept.remove(top.index[0])

    print(f"\nDropped {len(dropped)} features: {dropped}")
    macro, per_class = random_cv(X[kept], y)
    print(f"Pruned features, random CV: macro F1 = {macro:.4f}")
    print({LABELS[i]: round(v, 4) for i, v in enumerate(per_class)}, flush=True)

    model = make_model(TREES)
    model.fit(X[kept], y)
    pred = model.predict_proba(X_test[kept]).argmax(1)
    submission = pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in pred]})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(submission["Target"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
