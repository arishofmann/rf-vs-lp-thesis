"""Train/Val/Test scores for LP, RF, kNN on the forest-class subset, all four scenarios.

Usage:
  python -u train_eval.py \
      --base-dir ./embeddings \
      --output ./results/results_train_eval.csv
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=UserWarning)  # sklearn convergence/MultiOutput noise

FOREST_INDICES = [3, 6, 12, 17]

SCENARIOS = {
    "FI_FI": ("finland", "finland"),
    "FI_PT": ("finland", "portugal"),
    "PT_PT": ("portugal", "portugal"),
    "PT_FI": ("portugal", "finland"),
}

SEED = 42


def load_split(base_dir: Path, country: str, split: str):
    d = base_dir / country / split
    X = torch.load(d / "embeddings.pt", weights_only=True).numpy()
    y = torch.load(d / "labels.pt", weights_only=True).numpy()
    return X, y


def filter_forest(X, y):
    y_forest = y[:, FOREST_INDICES]
    mask = y_forest.sum(axis=1) > 0
    return X[mask], y_forest[mask]


def compute_metrics(y_true, y_prob):
    num_classes = y_true.shape[1]
    y_pred = (y_prob >= 0.5).astype(int)
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cols = [i for i in range(num_classes) if y_true[:, i].sum() > 0]
    mAP = float(average_precision_score(
        y_true[:, cols], y_prob[:, cols], average="macro"
    )) if cols else 0.0
    return f1, mAP


class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def train_lp(X_train, y_train, X_val, y_val, num_classes,
             max_epochs=50, lr=1e-3, patience=7):
    torch.manual_seed(SEED)
    model = LinearProbe(X_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).float()
    X_v = torch.from_numpy(X_val).float()
    y_v = torch.from_numpy(y_val).float()
    batch_size = min(256, len(X_tr))

    best_val_loss = float("inf")
    no_improve = 0
    best_state = None

    model.train()
    for _ in range(max_epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            loss = criterion(model(X_tr[idx]), y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()

    def predict(X):
        with torch.no_grad():
            return torch.sigmoid(model(torch.from_numpy(X).float())).numpy()

    return predict


def train_rf(X_train, y_train):
    rf = MultiOutputClassifier(
        RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
        n_jobs=1,
    )
    rf.fit(X_train, y_train.astype(int))

    def predict(X):
        num_classes = y_train.shape[1]
        probs = np.zeros((len(X), num_classes), dtype=np.float32)
        for i, est in enumerate(rf.estimators_):
            p = est.predict_proba(X)
            if p.shape[1] == 2:
                probs[:, i] = p[:, 1]
            elif p.shape[1] == 1:
                probs[:, i] = p[:, 0] if est.classes_[0] == 1 else 0.0
        return probs

    return predict


def train_knn(X_train, y_train):
    knn = MultiOutputClassifier(
        KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
        n_jobs=1,
    )
    knn.fit(X_train, y_train.astype(int))

    def predict(X):
        num_classes = y_train.shape[1]
        probs = np.zeros((len(X), num_classes), dtype=np.float32)
        for i, est in enumerate(knn.estimators_):
            p = est.predict_proba(X)
            if p.shape[1] == 2:
                probs[:, i] = p[:, 1]
            elif p.shape[1] == 1:
                probs[:, i] = p[:, 0] if est.classes_[0] == 1 else 0.0
        return probs

    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
                        default="./embeddings")
    parser.add_argument("--output",
                        default="./results/results_train_eval.csv")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    num_classes = len(FOREST_INDICES)

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        print(f"\n[{sc_name}] train={train_c}, test={test_c}")

        X_tr_raw, y_tr_raw = load_split(base_dir, train_c, "train")
        X_val_raw, y_val_raw = load_split(base_dir, train_c, "validation")
        X_te_raw, y_te_raw = load_split(base_dir, test_c, "test")

        X_tr, y_tr = filter_forest(X_tr_raw, y_tr_raw)
        X_val, y_val = filter_forest(X_val_raw, y_val_raw)
        X_te, y_te = filter_forest(X_te_raw, y_te_raw)

        print(f"  Forest patches — train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_te)}")

        decoders = {
            "LP": train_lp(X_tr, y_tr, X_val, y_val, num_classes),
            "RF": train_rf(X_tr, y_tr),
            "kNN": train_knn(X_tr, y_tr),
        }

        for dec_name, predict_fn in decoders.items():
            train_f1, train_mAP = compute_metrics(y_tr, predict_fn(X_tr))
            val_f1, val_mAP = compute_metrics(y_val, predict_fn(X_val))
            test_f1, test_mAP = compute_metrics(y_te, predict_fn(X_te))

            rows.append({
                "scenario": sc_name,
                "decoder": dec_name,
                "train_f1": round(train_f1, 4),
                "train_mAP": round(train_mAP, 4),
                "val_f1": round(val_f1, 4),
                "val_mAP": round(val_mAP, 4),
                "test_f1": round(test_f1, 4),
                "test_mAP": round(test_mAP, 4),
            })

            print(f"  {dec_name:4s}  "
                  f"train: F1={train_f1:.3f} mAP={train_mAP:.3f}  "
                  f"val:   F1={val_f1:.3f} mAP={val_mAP:.3f}  "
                  f"test:  F1={test_f1:.3f} mAP={test_mAP:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
