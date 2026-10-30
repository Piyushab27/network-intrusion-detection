"""
Network intrusion classification (Macro F1).

Pipeline:
  1. Load train/test, encode categorical columns (proto, service).
  2. Add flow-ratio features that help separate traffic profiles.
  3. Stratified 5-fold LightGBM with balanced class weights.
  4. Tune per-class probability multipliers on out-of-fold (OOF) predictions
     to maximise Macro F1 (rare classes matter equally).
  5. Average fold models on the test set and write submission.csv.

Usage:
  python train_intrusion_model.py --data-dir /path/to/csvs --out submission.csv
"""

import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_FOLDS = 5
CAT_COLS = ["proto", "service"]
LABELS = ["Availability", "Benign", "Reconnaissance", "Spoofing"]
N_TREES = 400  # fixed count for grouped CV; early stopping on the held-out fold would leak


def profile_groups(df: pd.DataFrame, n_clusters: int = 30) -> np.ndarray:
    """Unsupervised traffic-profile groups: k-means clusters of flow statistics.

    The dataset has no profile column. Grouping by protocol/service/port was
    tried first, but one port (21) holds 94% of the Availability rows, so
    holding it out leaves no Availability in training. Clustering on the flow
    statistics (no labels used) gives groups that look like distinct behaviour
    profiles, so holding whole clusters out tests generalisation to unseen
    behaviour as the competition describes.
    """
    num = df.select_dtypes("number").drop(columns=["id.orig_p", "id.resp_p"], errors="ignore")
    signed_log = np.sign(num) * np.log1p(np.abs(num))
    scaled = StandardScaler().fit_transform(signed_log.fillna(0).to_numpy())
    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=3, batch_size=4096
    )
    return kmeans.fit_predict(scaled)


def load(data_dir: Path):
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    return train, test


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1.0
    # Forward/backward asymmetry: attacks such as floods are often one-sided.
    df["pkt_asym"] = (df["fwd_pkts_tot"] - df["bwd_pkts_tot"]) / (
        df["fwd_pkts_tot"] + df["bwd_pkts_tot"] + eps
    )
    df["data_pkt_asym"] = (df["fwd_data_pkts_tot"] - df["bwd_data_pkts_tot"]) / (
        df["fwd_data_pkts_tot"] + df["bwd_data_pkts_tot"] + eps
    )
    # Byte asymmetry from payload totals.
    df["payload_asym"] = (df["fwd_pkts_payload.tot"] - df["bwd_pkts_payload.tot"]) / (
        df["fwd_pkts_payload.tot"] + df["bwd_pkts_payload.tot"] + eps
    )
    # Mean packet size proxies.
    df["fwd_bytes_per_pkt"] = df["fwd_pkts_payload.tot"] / (df["fwd_pkts_tot"] + eps)
    df["bwd_bytes_per_pkt"] = df["bwd_pkts_payload.tot"] / (df["bwd_pkts_tot"] + eps)
    # Handshake / reset pressure: SYN and RST without ACK are typical of scans.
    df["syn_ack_ratio"] = df["flow_SYN_flag_count"] / (df["flow_ACK_flag_count"] + eps)
    df["rst_per_pkt"] = df["flow_RST_flag_count"] / (
        df["fwd_pkts_tot"] + df["bwd_pkts_tot"] + eps
    )
    # Well-known vs ephemeral destination port (scans hit low ports).
    df["resp_port_is_wellknown"] = (df["id.resp_p"] < 1024).astype(int)
    return df


def prepare(train: pd.DataFrame, test: pd.DataFrame):
    feats_train = train.drop(columns=["id", "Target"])
    feats_test = test.drop(columns=["id"])

    # Shared categorical encoding so train and test use identical codes.
    for col in CAT_COLS:
        cats = pd.Categorical(
            pd.concat([feats_train[col], feats_test[col]]).astype(str)
        ).categories
        feats_train[col] = pd.Categorical(feats_train[col].astype(str), categories=cats)
        feats_test[col] = pd.Categorical(feats_test[col].astype(str), categories=cats)

    feats_train = add_features(feats_train)
    feats_test = add_features(feats_test)

    # Replace infinities produced by ratio features.
    feats_train = feats_train.replace([np.inf, -np.inf], np.nan)
    feats_test = feats_test.replace([np.inf, -np.inf], np.nan)

    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    y = train["Target"].map(label_to_idx).to_numpy()
    return feats_train, feats_test, y


def make_model(n_estimators: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=63,
        min_child_samples=10,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )


def tune_class_weights(proba: np.ndarray, y: np.ndarray, rounds: int = 3):
    """Coordinate search over per-class multipliers to maximise Macro F1."""
    weights = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(-1.5, 1.5, 31))
    best = f1_score(y, (proba * weights).argmax(1), average="macro")
    for _ in range(rounds):
        for c in range(proba.shape[1]):
            for g in grid:
                trial = weights.copy()
                trial[c] = g
                score = f1_score(y, (proba * trial).argmax(1), average="macro")
                if score > best:
                    best, weights = score, trial
    return weights, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("submission.csv"))
    parser.add_argument(
        "--cv",
        choices=["random", "group"],
        default="random",
        help="random: stratified folds (optimistic). group: hold out whole profiles.",
    )
    parser.add_argument(
        "--trees",
        type=int,
        default=None,
        help="Trees per fold. Defaults to 400 for group CV and 800 for random CV.",
    )
    args = parser.parse_args()

    train, test = load(args.data_dir)
    X, X_test, y = prepare(train, test)
    print(f"Train {X.shape}, test {X_test.shape}, CV={args.cv}")

    if args.cv == "group":
        groups = profile_groups(train)
        print(f"Profile groups in train: {len(np.unique(groups))}")
        splitter = StratifiedGroupKFold(
            n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE
        )
        folds = list(splitter.split(X, y, groups))
        n_trees = args.trees or N_TREES
    else:
        splitter = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        folds = list(splitter.split(X, y))
        n_trees = args.trees or 800

    oof = np.zeros((len(X), len(LABELS)))
    test_proba = np.zeros((len(X_test), len(LABELS)))

    for fold, (tr_idx, va_idx) in enumerate(folds, start=1):
        model = make_model(n_trees)
        model.fit(X.iloc[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict_proba(X.iloc[va_idx])
        test_proba += model.predict_proba(X_test) / N_FOLDS
        present = np.unique(y[va_idx])
        fold_f1 = f1_score(y[va_idx], oof[va_idx].argmax(1), labels=present, average="macro")
        print(f"Fold {fold}: macro F1 = {fold_f1:.4f} (val rows {len(va_idx)}, classes {len(present)})")

    base_f1 = f1_score(y, oof.argmax(1), average="macro")
    weights, tuned_f1 = tune_class_weights(oof, y)
    print(f"\nOOF macro F1 (argmax):  {base_f1:.4f}")
    print(f"OOF macro F1 (tuned):   {tuned_f1:.4f}  weights={np.round(weights, 3)}")
    print("\n" + classification_report(y, (oof * weights).argmax(1), target_names=LABELS, digits=4))

    preds = (test_proba * weights).argmax(1)
    submission = pd.DataFrame({"id": test["id"], "Target": [LABELS[i] for i in preds]})
    submission.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(submission)} rows")
    print(submission["Target"].value_counts())


if __name__ == "__main__":
    main()
