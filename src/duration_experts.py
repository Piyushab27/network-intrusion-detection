"""
Duration-aware analysis and one hypothesis: per-regime expert models.

Regimes (raw flow_duration, seconds): short < 1 ms, mid 1 ms to 1 s, long >= 1 s.

Part 1 (analysis, no training): train class counts per regime, out-of-fold
(OOF) per-class F1 per regime for the cached global models, and predicted test
shares per regime for the existing submissions.

Part 2 (hypothesis): train one LightGBM per regime on that regime's rows only,
using the same random 5-fold split, and compare against the global model.
  experts_all   every regime uses its own expert
  hybrid_short  the short-flow expert, the global model elsewhere
Test probabilities are cached; no submission is written here.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load, make_model, prepare
from tune_candidates import fast_macro_f1, tune

REGIMES = ["short", "mid", "long"]


def regime_of(duration: pd.Series) -> np.ndarray:
    return np.select([duration < 1e-3, duration < 1.0], ["short", "mid"], "long")


def per_class_f1(y, pred, k=4):
    cm = np.bincount(y * k + pred, minlength=k * k).reshape(k, k)
    tp = np.diag(cm).astype(float)
    denom = 2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp)
    return np.where(denom > 0, 2 * tp / np.maximum(denom, 1), np.nan)


def full_proba(model, X):
    """Expand predict_proba to all four classes when a regime lacks a class."""
    out = np.zeros((len(X), len(LABELS)))
    out[:, model.classes_] = model.predict_proba(X)
    return out


def nested(oof, y, folds):
    pred = np.zeros(len(y), dtype=int)
    for tr, va in folds:
        w, _ = tune(oof[tr], y[tr])
        pred[va] = (oof[va] * w).argmax(1)
    return pred


def report(name, y, pred, reg):
    print(f"\n{name}: overall macro F1 {fast_macro_f1(y, pred):.5f}")
    for r in REGIMES:
        m = reg == r
        f1 = per_class_f1(y[m], pred[m])
        print(f"  {r:<5} n={m.sum():>6}  " + "  ".join(f"{LABELS[i][:5]}={f1[i]:.3f}" for i in range(4)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    X, X_test, _ = prepare(train, test)
    reg, reg_test = regime_of(train["flow_duration"]), regime_of(test["flow_duration"])
    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(np.zeros(len(y)), y))

    print("== Train class counts by regime")
    print(pd.crosstab(reg, train["Target"]))
    print("\n== Test rows by regime:", pd.Series(reg_test).value_counts().to_dict())

    print("\n== Existing submissions: predicted share (%) within each test regime")
    for f in ["submission_pseudo_1round.csv", "submission_random_cv.csv", "submission_top2_blend.csv"]:
        s = pd.read_csv(f)
        tab = (pd.crosstab(reg_test, s["Target"], normalize="index") * 100).round(1)
        print(f"-- {f}\n{tab}")

    glob = np.load(args.cache_dir / "cv_original_800.npz")
    g_oof, g_test = glob["oof"], glob["test"]
    report("GLOBAL (tuned LightGBM, nested multipliers)", y, nested(g_oof, y, folds), reg)

    e_oof = np.zeros_like(g_oof)
    e_test = np.zeros_like(g_test)
    for r in REGIMES:
        idx = np.where(reg == r)[0]
        te_idx = np.where(reg_test == r)[0]
        for tr, va in folds:
            tr_r, va_r = np.intersect1d(tr, idx), np.intersect1d(va, idx)
            model = make_model(800)
            model.fit(X.iloc[tr_r], y[tr_r])
            e_oof[va_r] = full_proba(model, X.iloc[va_r])
            e_test[te_idx] += full_proba(model, X_test.iloc[te_idx]) / N_FOLDS
        print(f"expert {r} done", flush=True)

    report("EXPERTS_ALL (nested multipliers)", y, nested(e_oof, y, folds), reg)

    h_oof, h_test = g_oof.copy(), g_test.copy()
    h_oof[reg == "short"] = e_oof[reg == "short"]
    h_test[reg_test == "short"] = e_test[reg_test == "short"]
    report("HYBRID_SHORT (nested multipliers)", y, nested(h_oof, y, folds), reg)

    for name, oof, tp in [("experts_all", e_oof, e_test), ("hybrid_short", h_oof, h_test)]:
        w, _ = tune(oof, y)
        pred = (tp * w).argmax(1)
        print(f"\n{name} test share %:", {LABELS[i]: round((pred == i).mean() * 100, 1) for i in range(4)})
        np.savez(args.cache_dir / f"cv_{name}.npz", oof=oof, test=tp)


if __name__ == "__main__":
    main()
