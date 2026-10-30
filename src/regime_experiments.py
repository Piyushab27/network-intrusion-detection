"""
Regime check: does a feature set still work when flows look like the test set?

Almost all train flows last under 1 ms, while about 37% of test rows last 1 s
or longer. This script trains on the short train flows and scores on the long
ones, then compares feature sets. The best set is used for a final model
trained on all train rows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from train_intrusion_model import LABELS, load, make_model, prepare

LONG_FLOW_SECONDS = 1.0
TREES = 400
PORT_COLS = ["id.orig_p", "id.resp_p", "resp_port_is_wellknown"]


def time_cols(columns):
    keys = ("iat", "active", "idle", "per_sec", "bulk_rate")
    return [
        c
        for c in columns
        if any(k in c for k in keys) or c in ("flow_duration", "payload_bytes_per_second")
    ]


def configs(columns):
    time = time_cols(columns)
    return {
        "all_features": [],
        "no_time_features": time,
        "no_time_no_ports": time + PORT_COLS,
    }


def fit_predict(X, y, X_eval, drop):
    keep = [c for c in X.columns if c not in drop]
    model = make_model(TREES)
    model.fit(X[keep], y)
    return model.predict_proba(X_eval[keep]).argmax(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission_regime.csv"))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)
    long_mask = X["flow_duration"].to_numpy() >= LONG_FLOW_SECONDS
    print(f"Long-flow train rows (holdout): {long_mask.sum()} of {len(X)}")
    print(f"Holdout class counts: {dict(zip(LABELS, np.bincount(y[long_mask], minlength=4)))}")

    results = {}
    for name, drop in configs(X.columns).items():
        pred = fit_predict(X[~long_mask], y[~long_mask], X[long_mask], drop)
        truth = y[long_mask]
        per_class = f1_score(truth, pred, labels=range(4), average=None, zero_division=0)
        results[name] = per_class.mean()
        print(
            f"{name:<20} holdout macro F1={per_class.mean():.4f} "
            + " ".join(f"{LABELS[i][:5]}={per_class[i]:.3f}" for i in range(4))
            + f" acc={accuracy_score(truth, pred):.4f} dropped={len(drop)}",
            flush=True,
        )
        print(f"{'':<20} predicted counts on holdout: {dict(zip(LABELS, np.bincount(pred, minlength=4)))}")

    best = max(results, key=results.get)
    print(f"\nBest on long-flow holdout: {best} ({results[best]:.4f})")

    drop = configs(X.columns)[best]
    test_pred = fit_predict(X, y, X_test, drop)
    submission = pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in test_pred]})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(submission["Target"].value_counts(normalize=True).round(3).to_string())


if __name__ == "__main__":
    main()
