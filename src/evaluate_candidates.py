"""
Random 5-fold CV evaluation of candidate feature sets and training variants.

Every candidate uses the same folds: StratifiedKFold(5, shuffle=True,
random_state=42) on the train labels. Importance weights come from features
only (the train-versus-test classifier). Pseudo-labels are created inside each
fold from that fold's own model, so validation labels never reach them.

Candidates:
  original_800       original features, 800 trees (baseline reproduction)
  rebuilt_k0         rebuilt features, all kept, 400 trees
  pruned_k5/10/15/20/25
                     rebuilt features with the first k pruned features removed
  rebuilt_w_mod      rebuilt features, importance weights clipped to [0.1, 10]
  rebuilt_pseudo     rebuilt features, one in-fold pseudo-labelling round
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

from train_intrusion_model import LABELS, N_FOLDS, RANDOM_STATE, load, make_model, prepare

PSEUDO_CONFIDENCE = 0.9
PSEUDO_WEIGHT = 0.5


def fit_predict(X_tr, y_tr, w_tr, eval_sets, trees):
    model = make_model(trees)
    model.fit(X_tr, y_tr, sample_weight=w_tr)
    return [model.predict_proba(X_e) for X_e in eval_sets]


def run_fold(X_tr, y_tr, w_tr, X_va, X_te, trees, pseudo):
    p_va, p_te = fit_predict(X_tr, y_tr, w_tr, [X_va, X_te], trees)
    if pseudo:
        confident = p_te.max(1) >= PSEUDO_CONFIDENCE
        X_aug = pd.concat([X_tr, X_te[confident]], ignore_index=True)
        y_aug = np.r_[y_tr, p_te.argmax(1)[confident]]
        w_aug = np.r_[w_tr, np.full(confident.sum(), PSEUDO_WEIGHT)]
        p_va, p_te = fit_predict(X_aug, y_aug, w_aug, [X_va, X_te], trees)
    return p_va, p_te


def importance_weights(p_train, n_train, n_test, clip):
    odds = p_train / np.clip(1 - p_train, 1e-6, None)
    w = np.clip(odds * (n_train / n_test), clip[0], clip[1])
    return w / w.mean()


def summarise(name, y, oof, folds, test_proba, weights=None):
    pred = oof.argmax(1)
    labels = list(range(len(LABELS)))
    macro = f1_score(y, pred, average="macro", labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels)
    fold_scores = [f1_score(y[va], pred[va], average="macro", labels=labels) for _, va in folds]
    cm = confusion_matrix(y, pred, labels=labels)
    test_pred = test_proba.argmax(1)
    result = {
        "name": name,
        "macro_f1": round(macro, 5),
        "fold_mean": round(float(np.mean(fold_scores)), 5),
        "fold_std": round(float(np.std(fold_scores)), 5),
        **{f"f1_{LABELS[i]}": round(f1[i], 5) for i in labels},
        **{f"prec_{LABELS[i]}": round(precision[i], 5) for i in labels},
        **{f"rec_{LABELS[i]}": round(recall[i], 5) for i in labels},
        **{f"test_share_{LABELS[i]}": round((test_pred == i).mean(), 4) for i in labels},
    }
    if weights is not None:
        result["ess"] = round(weights.sum() ** 2 / (weights ** 2).sum(), 1)
    print(f"\n== {name}: macro F1 {macro:.5f}  folds {np.round(fold_scores, 5).tolist()}")
    print("   per-class F1 :", {LABELS[i]: round(f1[i], 4) for i in labels})
    print("   precision    :", {LABELS[i]: round(precision[i], 4) for i in labels})
    print("   recall       :", {LABELS[i]: round(recall[i], 4) for i in labels})
    print("   confusion (rows=true, cols=pred):\n", cm)
    print("   test predicted share:", {LABELS[i]: round((test_pred == i).mean(), 4) for i in labels})
    if weights is not None:
        print(f"   effective sample size: {result['ess']} of {len(weights)}")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dropped", required=True, help="Python list, in prune order")
    parser.add_argument("--configs", nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("candidate_cv_results.csv"))
    args = parser.parse_args()

    train, test = load(args.data_dir)
    y = train["Target"].map({label: i for i, label in enumerate(LABELS)}).to_numpy()
    X_orig, X_test_orig, _ = prepare(train, test)
    X_reb = pd.read_pickle(args.cache_dir / "X_rebuilt.pkl")
    X_test_reb = pd.read_pickle(args.cache_dir / "X_test_rebuilt.pkl")
    p_domain = np.load(args.cache_dir / "p_test_like_rebuilt.npy")
    dropped = ast.literal_eval(args.dropped)

    folds = list(StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE).split(X_reb, y))

    def spec(name):
        if name == "original_800":
            return X_orig, X_test_orig, None, 800, False
        if name == "rebuilt_k0":
            return X_reb, X_test_reb, None, 400, False
        if name.startswith("pruned_k"):
            k = int(name.removeprefix("pruned_k"))
            keep = [c for c in X_reb.columns if c not in dropped[:k]]
            return X_reb[keep], X_test_reb[keep], None, 400, False
        if name == "rebuilt_w_mod":
            return X_reb, X_test_reb, "mod", 400, False
        if name == "rebuilt_pseudo":
            return X_reb, X_test_reb, None, 400, True
        raise ValueError(f"unknown config {name}")

    results = []
    for name in args.configs:
        X, X_test, weight_mode, trees, pseudo = spec(name)
        weights = None
        if weight_mode == "mod":
            weights = importance_weights(p_domain, len(X), len(X_test), clip=(0.1, 10.0))
        oof = np.zeros((len(X), len(LABELS)))
        test_proba = np.zeros((len(X_test), len(LABELS)))
        for fold, (tr, va) in enumerate(folds, start=1):
            w_tr = weights[tr] if weights is not None else np.ones(len(tr))
            p_va, p_te = run_fold(X.iloc[tr], y[tr], w_tr, X.iloc[va], X_test, trees, pseudo)
            oof[va] = p_va
            test_proba += p_te / N_FOLDS
            print(f"  {name} fold {fold} done", flush=True)
        results.append(summarise(name, y, oof, folds, test_proba, weights))
        np.savez(args.cache_dir / f"cv_{name}.npz", oof=oof, test=test_proba)

    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
