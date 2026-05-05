"""E5 ablation: Linear Probing / Random Forest on Gaussian Random Projection (50d).

This script extends run_all_experiments_v2.py with the E5 ablation. The shared
infrastructure (data loading, decoder training, metrics) is duplicated rather
than imported to keep this ablation runnable as a standalone script. Output:
results/results_experiment_E5.{csv,json}.

Usage:
    python run_random_projection_ablation.py
"""

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings("ignore", category=UserWarning)

ALL_CLASSES = [
    "Agro-forestry areas", "Arable land", "Beaches, dunes, sands",
    "Broad-leaved forest", "Coastal wetlands", "Complex cultivation patterns",
    "Coniferous forest", "Industrial or commercial units", "Inland waters",
    "Inland wetlands",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Marine waters", "Mixed forest",
    "Moors, heathland and sclerophyllous vegetation",
    "Natural grassland and sparsely vegetated areas", "Pastures",
    "Permanent crops", "Transitional woodland, shrub", "Urban fabric",
]

FOREST_INDICES = [3, 6, 12, 17]
FOREST_NAMES = [ALL_CLASSES[i] for i in FOREST_INDICES]

FRACTIONS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
SEEDS = [42, 123, 456, 789, 1024]

SCENARIOS = {
    "FI_FI": ("finland", "finland"),
    "FI_PT": ("finland", "portugal"),
    "PT_PT": ("portugal", "portugal"),
    "PT_FI": ("portugal", "finland"),
}

_cache: dict[str, tuple] = {}


def load_split(base_dir: Path, country: str, split: str):
    key = f"{country}/{split}"
    if key not in _cache:
        d = base_dir / country / split
        X = torch.load(d / "embeddings.pt", weights_only=True).numpy()
        y = torch.load(d / "labels.pt", weights_only=True).numpy()
        _cache[key] = (X, y)
    return _cache[key]


def subsample_stratified(X, y, fraction, seed):
    """Stratified subsampling: guarantees >=1 positive per class, fills rest uniformly."""
    if fraction >= 1.0:
        return X, y
    rng = np.random.RandomState(seed)
    n = len(X)
    k = max(1, int(n * fraction))

    selected = set()
    num_classes = y.shape[1]

    # Phase 1: guarantee >=1 positive per class (randomise class priority)
    class_order = rng.permutation(num_classes)
    for cls in class_order:
        if len(selected) >= k:
            break
        pos = np.where(y[:, cls] == 1)[0]
        if len(pos) == 0:
            continue
        # Check if already covered by a previously selected sample
        if any(y[s, cls] == 1 for s in selected):
            continue
        chosen = rng.choice(pos)
        selected.add(int(chosen))

    # Phase 2: fill remaining budget uniformly from unselected samples
    remaining = k - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(n), list(selected))
        extra = rng.choice(pool, min(remaining, len(pool)), replace=False)
        selected.update(extra.tolist())

    idx = np.array(sorted(selected))
    return X[idx], y[idx]


def filter_forest(X, y):
    y_forest = y[:, FOREST_INDICES]
    mask = y_forest.sum(axis=1) > 0
    return X[mask], y_forest[mask]


def compute_metrics(y_true, y_prob, class_names=None):
    num_classes = y_true.shape[1]
    y_pred = (y_prob >= 0.5).astype(int)

    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0).tolist()

    cols_with_pos = [i for i in range(num_classes) if y_true[:, i].sum() > 0]
    if len(cols_with_pos) > 0:
        mAP = float(average_precision_score(
            y_true[:, cols_with_pos], y_prob[:, cols_with_pos], average="macro"
        ))
    else:
        mAP = 0.0

    auroc_per_class = []
    for i in range(num_classes):
        try:
            if y_true[:, i].sum() == 0 or y_true[:, i].sum() == len(y_true):
                auroc_per_class.append(float("nan"))
            else:
                auroc_per_class.append(float(roc_auc_score(y_true[:, i], y_prob[:, i])))
        except ValueError:
            auroc_per_class.append(float("nan"))

    return {
        "f1_macro": f1_macro,
        "mAP": mAP,
        "f1_per_class": f1_per_class,
        "auroc_per_class": auroc_per_class,
    }


class LinearProbe(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.linear(x)


def _train_lp_model(X_train, y_train, X_val, y_val, num_classes, seed,
                    max_epochs=50, lr=1e-3, patience=7):
    """Train LP model and return it (without evaluation)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = LinearProbe(X_train.shape[1], num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    X_tr = torch.from_numpy(X_train).float()
    y_tr = torch.from_numpy(y_train).float()
    X_v = torch.from_numpy(X_val).float()
    y_v = torch.from_numpy(y_val).float()
    batch_size = min(256, len(X_tr))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_state = None

    model.train()
    for epoch in range(max_epochs):
        perm = torch.randperm(len(X_tr))
        for i in range(0, len(X_tr), batch_size):
            idx = perm[i:i + batch_size]
            logits = model(X_tr[idx])
            loss = criterion(logits, y_tr[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(X_v), y_v).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    return model


def train_lp(X_train, y_train, X_val, y_val, X_test, y_test, num_classes, seed,
             max_epochs=50, lr=1e-3, patience=7):
    model = _train_lp_model(X_train, y_train, X_val, y_val, num_classes, seed,
                            max_epochs, lr, patience)
    with torch.no_grad():
        probs = torch.sigmoid(model(torch.from_numpy(X_test).float())).numpy()
    return compute_metrics(y_test, probs)


def _rf_predict_proba(rf, X_test, num_classes):
    probs = np.zeros((len(X_test), num_classes), dtype=np.float32)
    for i, est in enumerate(rf.estimators_):
        p = est.predict_proba(X_test)
        if p.shape[1] == 2:
            probs[:, i] = p[:, 1]
        elif p.shape[1] == 1:
            probs[:, i] = p[:, 0] if est.classes_[0] == 1 else 0.0
        else:
            probs[:, i] = p[:, 1]
    return probs


def train_rf(X_train, y_train, X_test, y_test, num_classes, seed,
             n_estimators=100, class_weight=None, max_features="sqrt"):
    rf = MultiOutputClassifier(
        RandomForestClassifier(
            n_estimators=n_estimators,
            class_weight=class_weight,
            max_features=max_features,
            random_state=seed,
            n_jobs=-1,
        ),
        n_jobs=1,
    )
    rf.fit(X_train, y_train.astype(int))
    probs = _rf_predict_proba(rf, X_test, num_classes)
    return compute_metrics(y_test, probs)


def _knn_predict_proba(knn, X_test, num_classes):
    """Extract probabilities from MultiOutputClassifier-wrapped kNN."""
    probs = np.zeros((len(X_test), num_classes), dtype=np.float32)
    for i, est in enumerate(knn.estimators_):
        p = est.predict_proba(X_test)
        if p.shape[1] == 2:
            probs[:, i] = p[:, 1]
        elif p.shape[1] == 1:
            probs[:, i] = p[:, 0] if est.classes_[0] == 1 else 0.0
        else:
            probs[:, i] = p[:, 1]
    return probs


def train_knn(X_train, y_train, X_test, y_test, num_classes, seed,
              n_neighbors=10):
    knn = MultiOutputClassifier(
        KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=-1),
        n_jobs=1,
    )
    knn.fit(X_train, y_train.astype(int))
    probs = _knn_predict_proba(knn, X_test, num_classes)
    return probs, compute_metrics(y_test, probs)


def _rf_native_predict_proba(rf, X_test, num_classes):
    """Extract probabilities from sklearn's native multi-output RF."""
    proba_list = rf.predict_proba(X_test)
    probs = np.zeros((len(X_test), num_classes), dtype=np.float32)
    for i in range(num_classes):
        p = proba_list[i]
        if p.shape[1] == 2:
            probs[:, i] = p[:, 1]
        elif p.shape[1] == 1:
            probs[:, i] = p[:, 0] if rf.classes_[i][0] == 1 else 0.0
        else:
            probs[:, i] = p[:, 1]
    return probs


def train_rf_native(X_train, y_train, X_test, y_test, num_classes, seed,
                    n_estimators=100, class_weight=None, max_features="sqrt"):
    """Train RF with native multi-output (shared tree structure, no MultiOutputClassifier)."""
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight=class_weight,
        max_features=max_features,
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train.astype(int))
    probs = _rf_native_predict_proba(rf, X_test, num_classes)
    return compute_metrics(y_test, probs)


def run_diagnostics(base_dir, output_dir):
    """Save exact split sizes, class distributions, and software versions."""
    print("\n=== DATA DIAGNOSTICS ===")

    diag = {
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "sklearn": __import__("sklearn").__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
        },
        "splits": {},
        "class_distributions": {},
    }

    for country in ["finland", "portugal"]:
        diag["splits"][country] = {}
        for split in ["train", "validation", "test"]:
            X, y = load_split(base_dir, country, split)
            n_samples = len(X)
            embedding_dim = X.shape[1]
            pos_per_class = y.sum(axis=0).astype(int).tolist()

            diag["splits"][country][split] = {
                "n_samples": n_samples,
                "embedding_dim": embedding_dim,
                "pos_per_class": dict(zip(ALL_CLASSES, pos_per_class)),
            }

            print(f"  {country}/{split}: {n_samples} samples, {embedding_dim}-dim embeddings")

        # Class distribution for train split
        _, y_train = load_split(base_dir, country, "train")
        freq = y_train.mean(axis=0)
        abs_count = y_train.sum(axis=0).astype(int)
        diag["class_distributions"][country] = {}
        for i, name in enumerate(ALL_CLASSES):
            diag["class_distributions"][country][name] = {
                "absolute": int(abs_count[i]),
                "relative": round(float(freq[i]), 6),
            }

    # Print class distribution table
    print(f"\n  {'Class':<55s} {'FI abs':>8s} {'FI %':>8s} {'PT abs':>8s} {'PT %':>8s}")
    print("  " + "-" * 87)
    for i, name in enumerate(ALL_CLASSES):
        fi = diag["class_distributions"]["finland"][name]
        pt = diag["class_distributions"]["portugal"][name]
        print(f"  {name:<55s} {fi['absolute']:>8d} {fi['relative']:>8.4f} "
              f"{pt['absolute']:>8d} {pt['relative']:>8.4f}")

    print(f"\n  Software versions:")
    for k, v in diag["software"].items():
        print(f"    {k}: {v}")

    _save_json(diag, output_dir / "diagnostics.json")
    return diag


def experiment_0(base_dir, output_dir):
    """Pre-study: LP vs RF vs RF-bal vs kNN on full 19 classes, all scenarios, 5 seeds."""
    print("\n=== EXPERIMENT 0: pre-study (Full 19 Classes) ===")

    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_val, y_val = load_split(base_dir, train_c, "validation")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            lp = train_lp(X_tr, y_tr, X_val, y_val, X_te, y_te, num_classes, seed)
            rf = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed)
            rf_bal = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed,
                              class_weight="balanced")
            _, knn = train_knn(X_tr, y_tr, X_te, y_te, num_classes, seed)

            row = {
                "experiment": "0_vorstudie", "scenario": sc_name, "seed": seed,
                "n_train": len(X_tr), "n_test": len(X_te),
                "lp_f1": lp["f1_macro"], "lp_mAP": lp["mAP"],
                "rf_f1": rf["f1_macro"], "rf_mAP": rf["mAP"],
                "rf_bal_f1": rf_bal["f1_macro"], "rf_bal_mAP": rf_bal["mAP"],
                "knn_f1": knn["f1_macro"], "knn_mAP": knn["mAP"],
                "lp_auroc_per_class": lp["auroc_per_class"],
                "rf_auroc_per_class": rf["auroc_per_class"],
                "rf_bal_auroc_per_class": rf_bal["auroc_per_class"],
                "knn_auroc_per_class": knn["auroc_per_class"],
                "lp_f1_per_class": lp["f1_per_class"],
                "rf_f1_per_class": rf["f1_per_class"],
                "rf_bal_f1_per_class": rf_bal["f1_per_class"],
                "knn_f1_per_class": knn["f1_per_class"],
                "class_names": ALL_CLASSES,
            }
            results.append(row)

            print(f"  seed={seed}  LP={lp['f1_macro']:.3f}  RF={rf['f1_macro']:.3f}  "
                  f"RF-bal={rf_bal['f1_macro']:.3f}  kNN={knn['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_0.json")
    _save_summary_csv(results, "0_vorstudie", output_dir / "results_experiment_0.csv")
    return results


def experiment_A(base_dir, output_dir):
    print("\n=== EXPERIMENT A: Forest Class Subset ===")

    results = []
    num_classes = len(FOREST_INDICES)

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr_full, y_tr_full = load_split(base_dir, train_c, "train")
        X_val_full, y_val_full = load_split(base_dir, train_c, "validation")
        X_te_full, y_te_full = load_split(base_dir, test_c, "test")

        X_tr, y_tr = filter_forest(X_tr_full, y_tr_full)
        X_val, y_val = filter_forest(X_val_full, y_val_full)
        X_te, y_te = filter_forest(X_te_full, y_te_full)

        print(f"\n--- {sc_name} (train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}) ---")
        for i, name in enumerate(FOREST_NAMES):
            print(f"  {name}: train={int(y_tr[:, i].sum())}, test={int(y_te[:, i].sum())}")

        for seed in SEEDS:
            lp = train_lp(X_tr, y_tr, X_val, y_val, X_te, y_te, num_classes, seed)
            rf = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed)
            rf_bal = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed, class_weight="balanced")
            _, knn = train_knn(X_tr, y_tr, X_te, y_te, num_classes, seed)

            row = {
                "experiment": "A_forest", "scenario": sc_name, "seed": seed,
                "n_train": len(X_tr), "n_test": len(X_te),
                "lp_f1": lp["f1_macro"], "lp_mAP": lp["mAP"],
                "rf_f1": rf["f1_macro"], "rf_mAP": rf["mAP"],
                "rf_bal_f1": rf_bal["f1_macro"], "rf_bal_mAP": rf_bal["mAP"],
                "knn_f1": knn["f1_macro"], "knn_mAP": knn["mAP"],
                "lp_f1_per_class": lp["f1_per_class"],
                "rf_f1_per_class": rf["f1_per_class"],
                "rf_bal_f1_per_class": rf_bal["f1_per_class"],
                "knn_f1_per_class": knn["f1_per_class"],
                "lp_auroc_per_class": lp["auroc_per_class"],
                "rf_auroc_per_class": rf["auroc_per_class"],
                "rf_bal_auroc_per_class": rf_bal["auroc_per_class"],
                "knn_auroc_per_class": knn["auroc_per_class"],
                "class_names": FOREST_NAMES,
            }
            results.append(row)

            print(f"  seed={seed}  LP={lp['f1_macro']:.3f}  RF={rf['f1_macro']:.3f}  "
                  f"RF-bal={rf_bal['f1_macro']:.3f}  kNN={knn['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_A.json")
    _save_summary_csv(results, "A_forest", output_dir / "results_experiment_A.csv")
    return results


def experiment_B(base_dir, output_dir):
    """Sample efficiency with STRATIFIED subsampling (v2 change)."""
    print("\n=== EXPERIMENT B: Sample Efficiency (Stratified Subsampling) ===")

    results = []
    scenarios_b = {k: v for k, v in SCENARIOS.items() if k in ("FI_FI", "FI_PT")}

    for sc_name, (train_c, test_c) in scenarios_b.items():
        X_tr_full, y_tr_full = load_split(base_dir, train_c, "train")
        X_val, y_val = load_split(base_dir, train_c, "validation")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr_full.shape[1]

        print(f"\n--- {sc_name}: {train_c}->{test_c} "
              f"(full_train={len(X_tr_full)}, test={len(X_te)}) ---")

        for frac in FRACTIONS:
            for seed in SEEDS:
                X_sub, y_sub = subsample_stratified(X_tr_full, y_tr_full, frac, seed)
                n = len(X_sub)
                pos_per_class = y_sub.sum(axis=0).astype(int).tolist()
                classes_with_zero = sum(1 for p in pos_per_class if p == 0)

                t0 = time.time()
                lp = train_lp(X_sub, y_sub, X_val, y_val, X_te, y_te, num_classes, seed)
                lp_t = time.time() - t0

                t0 = time.time()
                rf = train_rf(X_sub, y_sub, X_te, y_te, num_classes, seed)
                rf_t = time.time() - t0

                t0 = time.time()
                rf_bal = train_rf(X_sub, y_sub, X_te, y_te, num_classes, seed,
                                  class_weight="balanced")
                rfb_t = time.time() - t0

                t0 = time.time()
                _, knn = train_knn(X_sub, y_sub, X_te, y_te, num_classes, seed)
                knn_t = time.time() - t0

                row = {
                    "experiment": "B_sample_eff", "scenario": sc_name,
                    "fraction": frac, "n_train": n, "seed": seed,
                    "classes_with_zero_positives": classes_with_zero,
                    "pos_per_class": pos_per_class,
                    "lp_f1": lp["f1_macro"], "lp_mAP": lp["mAP"], "lp_time": round(lp_t, 1),
                    "rf_f1": rf["f1_macro"], "rf_mAP": rf["mAP"], "rf_time": round(rf_t, 1),
                    "rf_bal_f1": rf_bal["f1_macro"], "rf_bal_mAP": rf_bal["mAP"],
                    "rf_bal_time": round(rfb_t, 1),
                    "knn_f1": knn["f1_macro"], "knn_mAP": knn["mAP"],
                    "knn_time": round(knn_t, 1),
                    "lp_f1_per_class": lp["f1_per_class"],
                    "rf_f1_per_class": rf["f1_per_class"],
                    "lp_auroc_per_class": lp["auroc_per_class"],
                    "rf_auroc_per_class": rf["auroc_per_class"],
                }
                results.append(row)

                print(f"  frac={frac:.2f} n={n:>6d} seed={seed} zero_cls={classes_with_zero}  "
                      f"LP={lp['f1_macro']:.3f}({lp_t:.0f}s) "
                      f"RF={rf['f1_macro']:.3f}({rf_t:.0f}s) "
                      f"RF-b={rf_bal['f1_macro']:.3f} "
                      f"kNN={knn['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_B.json")
    _save_summary_csv(results, "B_sample_eff", output_dir / "results_experiment_B.csv")
    return results


def experiment_C(base_dir, output_dir):
    """RF optimisation with extended variants (v2: added PCA-50, PCA-200, mtry-200)."""
    print("\n=== EXPERIMENT C: RF Optimisation (Extended) ===")

    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            rf_std = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed)

            rf_bal = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed,
                              class_weight="balanced")

            rf_mtry100 = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed,
                                   max_features=100)

            rf_mtry200 = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed,
                                   max_features=200)

            pca_results = {}
            for n_comp in [50, 100, 200]:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr)
                X_te_pca = pca.transform(X_te)
                variance_explained = float(pca.explained_variance_ratio_.sum())
                pca_results[n_comp] = {
                    "metrics": train_rf(X_tr_pca, y_tr, X_te_pca, y_te, num_classes, seed),
                    "variance_explained": variance_explained,
                }

            row = {
                "experiment": "C_rf_opt", "scenario": sc_name, "seed": seed,
                "rf_std_f1": rf_std["f1_macro"], "rf_std_mAP": rf_std["mAP"],
                "rf_bal_f1": rf_bal["f1_macro"], "rf_bal_mAP": rf_bal["mAP"],
                "rf_mtry100_f1": rf_mtry100["f1_macro"], "rf_mtry100_mAP": rf_mtry100["mAP"],
                "rf_mtry200_f1": rf_mtry200["f1_macro"], "rf_mtry200_mAP": rf_mtry200["mAP"],
                "rf_pca50_f1": pca_results[50]["metrics"]["f1_macro"],
                "rf_pca50_mAP": pca_results[50]["metrics"]["mAP"],
                "rf_pca50_var": pca_results[50]["variance_explained"],
                "rf_pca100_f1": pca_results[100]["metrics"]["f1_macro"],
                "rf_pca100_mAP": pca_results[100]["metrics"]["mAP"],
                "rf_pca100_var": pca_results[100]["variance_explained"],
                "rf_pca200_f1": pca_results[200]["metrics"]["f1_macro"],
                "rf_pca200_mAP": pca_results[200]["metrics"]["mAP"],
                "rf_pca200_var": pca_results[200]["variance_explained"],
                "rf_std_f1_per_class": rf_std["f1_per_class"],
                "rf_bal_f1_per_class": rf_bal["f1_per_class"],
                "rf_mtry100_f1_per_class": rf_mtry100["f1_per_class"],
                "rf_mtry200_f1_per_class": rf_mtry200["f1_per_class"],
                "rf_pca50_f1_per_class": pca_results[50]["metrics"]["f1_per_class"],
                "rf_pca100_f1_per_class": pca_results[100]["metrics"]["f1_per_class"],
                "rf_pca200_f1_per_class": pca_results[200]["metrics"]["f1_per_class"],
            }
            results.append(row)

            print(f"  seed={seed}  std={rf_std['f1_macro']:.3f}  bal={rf_bal['f1_macro']:.3f}  "
                  f"mtry100={rf_mtry100['f1_macro']:.3f}  mtry200={rf_mtry200['f1_macro']:.3f}  "
                  f"pca50={pca_results[50]['metrics']['f1_macro']:.3f}  "
                  f"pca100={pca_results[100]['metrics']['f1_macro']:.3f}  "
                  f"pca200={pca_results[200]['metrics']['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_C.json")
    _save_csv_C(results, output_dir / "results_experiment_C.csv")
    return results


def experiment_D(base_dir, output_dir):
    print("\n=== EXPERIMENT D: Per-Class Domain Shift Analysis ===")

    _, y_fi = load_split(base_dir, "finland", "train")
    _, y_pt = load_split(base_dir, "portugal", "train")

    freq_fi = y_fi.mean(axis=0)
    freq_pt = y_pt.mean(axis=0)
    freq_diff = np.abs(freq_fi - freq_pt)

    eps = 1e-10
    chi2_dist = np.sum((freq_fi - freq_pt) ** 2 / (freq_fi + freq_pt + eps))

    print(f"\nClass frequency comparison (train splits):")
    print(f"  Chi-square distance: {chi2_dist:.4f}")
    print(f"  {'Class':<60s} {'FI freq':>8s} {'PT freq':>8s} {'|diff|':>8s}")
    for i, name in enumerate(ALL_CLASSES):
        print(f"  {name:<60s} {freq_fi[i]:>8.4f} {freq_pt[i]:>8.4f} {freq_diff[i]:>8.4f}")

    num_classes = y_fi.shape[1]

    all_f1s = {}
    all_aurocs = {}

    DECODERS_D = ["LP", "RF_bal", "RF_std", "kNN", "RF_pca50"]

    for decoder_name in DECODERS_D:
        for sc_name, (train_c, test_c) in SCENARIOS.items():
            X_tr, y_tr = load_split(base_dir, train_c, "train")
            X_val, y_val = load_split(base_dir, train_c, "validation")
            X_te, y_te = load_split(base_dir, test_c, "test")

            f1s_per_seed = []
            aurocs_per_seed = []

            for seed in SEEDS:
                if decoder_name == "LP":
                    m = train_lp(X_tr, y_tr, X_val, y_val, X_te, y_te, num_classes, seed)
                elif decoder_name == "RF_bal":
                    m = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed,
                                 class_weight="balanced")
                elif decoder_name == "RF_std":
                    m = train_rf(X_tr, y_tr, X_te, y_te, num_classes, seed)
                elif decoder_name == "kNN":
                    _, m = train_knn(X_tr, y_tr, X_te, y_te, num_classes, seed)
                elif decoder_name == "RF_pca50":
                    pca = PCA(n_components=50, random_state=seed)
                    X_tr_pca = pca.fit_transform(X_tr)
                    X_te_pca = pca.transform(X_te)
                    m = train_rf(X_tr_pca, y_tr, X_te_pca, y_te, num_classes, seed)
                f1s_per_seed.append(m["f1_per_class"])
                aurocs_per_seed.append(m["auroc_per_class"])

            key = f"{decoder_name}_{sc_name}"
            all_f1s[key] = np.array(f1s_per_seed)
            all_aurocs[key] = np.array(aurocs_per_seed)

            mean_f1 = np.mean(f1s_per_seed, axis=0)
            print(f"  {key}: F1-macro={np.mean(mean_f1):.3f}")

    drop_analysis = []
    for decoder_name in DECODERS_D:
        fi_id = np.mean(all_f1s[f"{decoder_name}_FI_FI"], axis=0)
        fi_ood = np.mean(all_f1s[f"{decoder_name}_FI_PT"], axis=0)
        pt_id = np.mean(all_f1s[f"{decoder_name}_PT_PT"], axis=0)
        pt_ood = np.mean(all_f1s[f"{decoder_name}_PT_FI"], axis=0)

        fi_id_std = np.std(all_f1s[f"{decoder_name}_FI_FI"], axis=0)
        fi_ood_std = np.std(all_f1s[f"{decoder_name}_FI_PT"], axis=0)
        pt_id_std = np.std(all_f1s[f"{decoder_name}_PT_PT"], axis=0)
        pt_ood_std = np.std(all_f1s[f"{decoder_name}_PT_FI"], axis=0)

        drop_fi = fi_id - fi_ood
        drop_pt = pt_id - pt_ood

        for i, name in enumerate(ALL_CLASSES):
            drop_analysis.append({
                "decoder": decoder_name, "class": name, "class_idx": i,
                "fi_id_f1": float(fi_id[i]), "fi_id_f1_std": float(fi_id_std[i]),
                "fi_ood_f1": float(fi_ood[i]), "fi_ood_f1_std": float(fi_ood_std[i]),
                "drop_fi": float(drop_fi[i]),
                "pt_id_f1": float(pt_id[i]), "pt_id_f1_std": float(pt_id_std[i]),
                "pt_ood_f1": float(pt_ood[i]), "pt_ood_f1_std": float(pt_ood_std[i]),
                "drop_pt": float(drop_pt[i]),
                "freq_fi": float(freq_fi[i]), "freq_pt": float(freq_pt[i]),
                "freq_diff": float(freq_diff[i]),
            })

    print(f"\nDomain shift drop (FI->FI minus FI->PT), per class:")
    header = f"  {'Class':<40s}"
    for dn in DECODERS_D:
        header += f" {dn:>9s}"
    header += f" {'freq_diff':>10s}"
    print(header)
    for i, name in enumerate(ALL_CLASSES):
        line = f"  {name:<40s}"
        for dn in DECODERS_D:
            drop = [d for d in drop_analysis if d["class_idx"] == i and d["decoder"] == dn][0]["drop_fi"]
            line += f" {drop:>+9.3f}"
        line += f" {freq_diff[i]:>10.4f}"
        print(line)

    for decoder_name in DECODERS_D:
        drops = [d["drop_fi"] for d in drop_analysis if d["decoder"] == decoder_name]
        freqs_train = [d["freq_fi"] for d in drop_analysis if d["decoder"] == decoder_name]
        diffs = [d["freq_diff"] for d in drop_analysis if d["decoder"] == decoder_name]

        valid = [(dr, fr, di) for dr, fr, di in zip(drops, freqs_train, diffs)
                 if not np.isnan(dr)]
        if len(valid) >= 3:
            dr_arr = [v[0] for v in valid]
            fr_arr = [v[1] for v in valid]
            di_arr = [v[2] for v in valid]

            r_freq, p_freq = pearsonr(dr_arr, fr_arr)
            r_diff, p_diff = pearsonr(dr_arr, di_arr)
            print(f"\n  {decoder_name} correlations:")
            print(f"    Drop vs train_freq:  r={r_freq:.3f}, p={p_freq:.4f}")
            print(f"    Drop vs freq_diff:   r={r_diff:.3f}, p={p_diff:.4f}")

    result = {
        "chi2_distance": float(chi2_dist),
        "freq_fi": freq_fi.tolist(),
        "freq_pt": freq_pt.tolist(),
        "drop_analysis": drop_analysis,
        "all_f1s": {k: v.tolist() for k, v in all_f1s.items()},
        "all_aurocs": {k: v.tolist() for k, v in all_aurocs.items()},
    }
    _save_json(result, output_dir / "results_experiment_D.json")

    df_drop = pd.DataFrame(drop_analysis)
    df_drop.to_csv(output_dir / "results_experiment_D.csv", index=False)
    print(f"\nSaved -> {output_dir / 'results_experiment_D.csv'}")

    return [result]


def experiment_E1(base_dir, output_dir):
    """Native multi-output RF (shared tree structure, no Binary Relevance wrapper)."""
    print("\n=== EXPERIMENT E1: Native Multi-Output RF ===")

    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            rf_native = train_rf_native(X_tr, y_tr, X_te, y_te, num_classes, seed)

            rf_native_bal = train_rf_native(X_tr, y_tr, X_te, y_te, num_classes, seed,
                                            class_weight="balanced")

            pca = PCA(n_components=50, random_state=seed)
            X_tr_pca = pca.fit_transform(X_tr)
            X_te_pca = pca.transform(X_te)
            variance_explained = float(pca.explained_variance_ratio_.sum())
            rf_native_pca50 = train_rf_native(X_tr_pca, y_tr, X_te_pca, y_te,
                                               num_classes, seed)

            row = {
                "experiment": "E1_native_rf", "scenario": sc_name, "seed": seed,
                "rf_native_f1": rf_native["f1_macro"],
                "rf_native_mAP": rf_native["mAP"],
                "rf_native_bal_f1": rf_native_bal["f1_macro"],
                "rf_native_bal_mAP": rf_native_bal["mAP"],
                "rf_native_pca50_f1": rf_native_pca50["f1_macro"],
                "rf_native_pca50_mAP": rf_native_pca50["mAP"],
                "rf_native_pca50_var": variance_explained,
                "rf_native_f1_per_class": rf_native["f1_per_class"],
                "rf_native_bal_f1_per_class": rf_native_bal["f1_per_class"],
                "rf_native_pca50_f1_per_class": rf_native_pca50["f1_per_class"],
            }
            results.append(row)

            print(f"  seed={seed}  native={rf_native['f1_macro']:.3f}  "
                  f"native_bal={rf_native_bal['f1_macro']:.3f}  "
                  f"native_pca50={rf_native_pca50['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_E1.json")
    _save_csv_E1(results, output_dir / "results_experiment_E1.csv")
    return results


def experiment_E3(base_dir, output_dir):
    """LP on PCA-reduced embeddings (50/100/200 components)."""
    print("\n=== EXPERIMENT E3: LP on PCA-reduced Embeddings ===")

    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_val, y_val = load_split(base_dir, train_c, "validation")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            pca_results = {}
            for n_comp in [50, 100, 200]:
                pca = PCA(n_components=n_comp, random_state=seed)
                X_tr_pca = pca.fit_transform(X_tr)
                X_val_pca = pca.transform(X_val)
                X_te_pca = pca.transform(X_te)
                variance_explained = float(pca.explained_variance_ratio_.sum())

                m = train_lp(X_tr_pca, y_tr, X_val_pca, y_val, X_te_pca, y_te,
                             num_classes, seed)
                pca_results[n_comp] = {
                    "metrics": m,
                    "variance_explained": variance_explained,
                }

            row = {
                "experiment": "E3_lp_pca", "scenario": sc_name, "seed": seed,
                "lp_pca50_f1": pca_results[50]["metrics"]["f1_macro"],
                "lp_pca50_mAP": pca_results[50]["metrics"]["mAP"],
                "lp_pca50_var": pca_results[50]["variance_explained"],
                "lp_pca100_f1": pca_results[100]["metrics"]["f1_macro"],
                "lp_pca100_mAP": pca_results[100]["metrics"]["mAP"],
                "lp_pca100_var": pca_results[100]["variance_explained"],
                "lp_pca200_f1": pca_results[200]["metrics"]["f1_macro"],
                "lp_pca200_mAP": pca_results[200]["metrics"]["mAP"],
                "lp_pca200_var": pca_results[200]["variance_explained"],
                "lp_pca50_f1_per_class": pca_results[50]["metrics"]["f1_per_class"],
                "lp_pca100_f1_per_class": pca_results[100]["metrics"]["f1_per_class"],
                "lp_pca200_f1_per_class": pca_results[200]["metrics"]["f1_per_class"],
            }
            results.append(row)

            print(f"  seed={seed}  pca50={pca_results[50]['metrics']['f1_macro']:.3f}  "
                  f"pca100={pca_results[100]['metrics']['f1_macro']:.3f}  "
                  f"pca200={pca_results[200]['metrics']['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_E3.json")
    _save_csv_E3(results, output_dir / "results_experiment_E3.csv")
    return results


def experiment_E5(base_dir, output_dir):
    """E5: Gaussian Random Projection (50d). Isometric to PCA-50 (E3); tests
    whether PCA gain is dimension- or correlation-driven."""
    print("\n=== EXPERIMENT E5: Random Projection (50d) — LP and RF ===")

    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_val, y_val = load_split(base_dir, train_c, "validation")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            n_comp = 50
            rp = GaussianRandomProjection(n_components=n_comp, random_state=seed)
            X_tr_rp = rp.fit_transform(X_tr)
            X_val_rp = rp.transform(X_val)
            X_te_rp = rp.transform(X_te)

            lp_m = train_lp(X_tr_rp, y_tr, X_val_rp, y_val, X_te_rp, y_te,
                            num_classes, seed)
            rf_m = train_rf(X_tr_rp, y_tr, X_te_rp, y_te, num_classes, seed)

            row = {
                "experiment": "E5_rp", "scenario": sc_name, "seed": seed,
                "lp_rp50_f1": lp_m["f1_macro"],
                "lp_rp50_mAP": lp_m["mAP"],
                "rf_rp50_f1": rf_m["f1_macro"],
                "rf_rp50_mAP": rf_m["mAP"],
                "lp_rp50_f1_per_class": lp_m["f1_per_class"],
                "rf_rp50_f1_per_class": rf_m["f1_per_class"],
            }
            results.append(row)

            print(f"  seed={seed}  LP_rp50={lp_m['f1_macro']:.3f}  "
                  f"RF_rp50={rf_m['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_E5.json")
    _save_csv_E5(results, output_dir / "results_experiment_E5.csv")
    return results


def experiment_E4(base_dir, output_dir):
    """Threshold sweep: F1-macro at thresholds 0.1-0.9 for all decoders at 100% training."""
    print("\n=== EXPERIMENT E4: Threshold Sweep ===")

    THRESHOLDS = [round(t, 1) for t in np.arange(0.1, 1.0, 0.1)]
    results = []

    for sc_name, (train_c, test_c) in SCENARIOS.items():
        X_tr, y_tr = load_split(base_dir, train_c, "train")
        X_val, y_val = load_split(base_dir, train_c, "validation")
        X_te, y_te = load_split(base_dir, test_c, "test")
        num_classes = y_tr.shape[1]

        print(f"\n--- {sc_name} (train={len(X_tr)}, test={len(X_te)}) ---")

        for seed in SEEDS:
            probs_by_decoder = {}

            # LP
            model = _train_lp_model(X_tr, y_tr, X_val, y_val, num_classes, seed)
            with torch.no_grad():
                probs_by_decoder["lp"] = torch.sigmoid(
                    model(torch.from_numpy(X_te).float())).numpy()

            # RF std (Binary Relevance)
            rf = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
                n_jobs=1)
            rf.fit(X_tr, y_tr.astype(int))
            probs_by_decoder["rf"] = _rf_predict_proba(rf, X_te, num_classes)

            # RF balanced (Binary Relevance)
            rf_bal = MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, random_state=seed,
                                       class_weight="balanced", n_jobs=-1),
                n_jobs=1)
            rf_bal.fit(X_tr, y_tr.astype(int))
            probs_by_decoder["rf_bal"] = _rf_predict_proba(rf_bal, X_te, num_classes)

            # kNN
            knn = MultiOutputClassifier(
                KNeighborsClassifier(n_neighbors=10, n_jobs=-1), n_jobs=1)
            knn.fit(X_tr, y_tr.astype(int))
            probs_by_decoder["knn"] = _knn_predict_proba(knn, X_te, num_classes)

            # RF native (shared tree structure, no Binary Relevance)
            rf_nat = RandomForestClassifier(
                n_estimators=100, random_state=seed, n_jobs=-1)
            rf_nat.fit(X_tr, y_tr.astype(int))
            probs_by_decoder["rf_native"] = _rf_native_predict_proba(
                rf_nat, X_te, num_classes)

            # Save probs as .npy for post-hoc analysis
            probs_dir = output_dir / "probs_E4"
            probs_dir.mkdir(parents=True, exist_ok=True)
            for dec_name, probs in probs_by_decoder.items():
                np.save(probs_dir / f"{sc_name}_{dec_name}_seed{seed}.npy", probs)

            # Sweep thresholds
            for dec_name, probs in probs_by_decoder.items():
                for thresh in THRESHOLDS:
                    y_pred = (probs >= thresh).astype(int)
                    f1 = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
                    results.append({
                        "scenario": sc_name, "seed": seed,
                        "decoder": dec_name, "threshold": thresh,
                        "f1_macro": f1,
                    })

            # Print optimal thresholds for this seed
            for dec_name in probs_by_decoder:
                rows = [r for r in results
                        if r["scenario"] == sc_name and r["seed"] == seed
                        and r["decoder"] == dec_name]
                best = max(rows, key=lambda r: r["f1_macro"])
                print(f"  seed={seed}  {dec_name}: best_thresh={best['threshold']:.1f}  "
                      f"f1={best['f1_macro']:.3f}")

    _save_json(results, output_dir / "results_experiment_E4.json")
    pd.DataFrame(results).to_csv(output_dir / "results_experiment_E4.csv", index=False)
    print(f"\nSaved -> {output_dir / 'results_experiment_E4.csv'}")
    return results


def _save_json(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved -> {path}")


def _save_summary_csv(results, experiment_name, path):
    rows_csv = []

    for sc in SCENARIOS:
        matched = [r for r in results if r["scenario"] == sc]
        if not matched:
            continue

        fracs = sorted(set(r.get("fraction", 1.0) for r in matched))

        for frac in fracs:
            sub = [r for r in matched if r.get("fraction", 1.0) == frac]
            if not sub:
                continue
            n = sub[0].get("n_train", "")

            for decoder in ["lp", "rf", "rf_bal", "knn"]:
                f1_key = f"{decoder}_f1"
                map_key = f"{decoder}_mAP"
                if f1_key not in sub[0]:
                    continue
                f1s = [r[f1_key] for r in sub]
                maps = [r[map_key] for r in sub]
                rows_csv.append({
                    "experiment": experiment_name, "scenario": sc,
                    "fraction": frac, "n_train": n, "decoder": decoder,
                    "f1_mean": f"{np.mean(f1s):.4f}", "f1_std": f"{np.std(f1s):.4f}",
                    "mAP_mean": f"{np.mean(maps):.4f}", "mAP_std": f"{np.std(maps):.4f}",
                })

    df = pd.DataFrame(rows_csv)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def _save_csv_C(results, path):
    rows_csv = []
    for sc in SCENARIOS:
        matched = [r for r in results if r["scenario"] == sc]
        if not matched:
            continue
        for variant in ["rf_std", "rf_bal", "rf_mtry100", "rf_mtry200",
                        "rf_pca50", "rf_pca100", "rf_pca200"]:
            f1_key = f"{variant}_f1"
            map_key = f"{variant}_mAP"
            if f1_key not in matched[0]:
                continue
            f1s = [r[f1_key] for r in matched]
            maps = [r[map_key] for r in matched]
            row = {
                "scenario": sc, "variant": variant,
                "f1_mean": f"{np.mean(f1s):.4f}", "f1_std": f"{np.std(f1s):.4f}",
                "mAP_mean": f"{np.mean(maps):.4f}", "mAP_std": f"{np.std(maps):.4f}",
            }
            # Add variance explained for PCA variants
            var_key = f"{variant}_var"
            if var_key in matched[0]:
                row["variance_explained"] = f"{np.mean([r[var_key] for r in matched]):.4f}"
            rows_csv.append(row)
    df = pd.DataFrame(rows_csv)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def _save_csv_E1(results, path):
    rows_csv = []
    for sc in SCENARIOS:
        matched = [r for r in results if r["scenario"] == sc]
        if not matched:
            continue
        for variant in ["rf_native", "rf_native_bal", "rf_native_pca50"]:
            f1_key = f"{variant}_f1"
            map_key = f"{variant}_mAP"
            if f1_key not in matched[0]:
                continue
            f1s = [r[f1_key] for r in matched]
            maps = [r[map_key] for r in matched]
            row = {
                "scenario": sc, "variant": variant,
                "f1_mean": f"{np.mean(f1s):.4f}", "f1_std": f"{np.std(f1s):.4f}",
                "mAP_mean": f"{np.mean(maps):.4f}", "mAP_std": f"{np.std(maps):.4f}",
            }
            var_key = f"{variant}_var"
            if var_key in matched[0]:
                row["variance_explained"] = f"{np.mean([r[var_key] for r in matched]):.4f}"
            rows_csv.append(row)
    df = pd.DataFrame(rows_csv)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def _save_csv_E3(results, path):
    rows_csv = []
    for sc in SCENARIOS:
        matched = [r for r in results if r["scenario"] == sc]
        if not matched:
            continue
        for variant in ["lp_pca50", "lp_pca100", "lp_pca200"]:
            f1_key = f"{variant}_f1"
            map_key = f"{variant}_mAP"
            if f1_key not in matched[0]:
                continue
            f1s = [r[f1_key] for r in matched]
            maps = [r[map_key] for r in matched]
            row = {
                "scenario": sc, "variant": variant,
                "f1_mean": f"{np.mean(f1s):.4f}", "f1_std": f"{np.std(f1s):.4f}",
                "mAP_mean": f"{np.mean(maps):.4f}", "mAP_std": f"{np.std(maps):.4f}",
            }
            var_key = f"{variant}_var"
            if var_key in matched[0]:
                row["variance_explained"] = f"{np.mean([r[var_key] for r in matched]):.4f}"
            rows_csv.append(row)
    df = pd.DataFrame(rows_csv)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def _save_csv_E5(results, path):
    rows_csv = []
    for sc in SCENARIOS:
        matched = [r for r in results if r["scenario"] == sc]
        if not matched:
            continue
        for variant in ["lp_rp50", "rf_rp50"]:
            f1_key = f"{variant}_f1"
            map_key = f"{variant}_mAP"
            if f1_key not in matched[0]:
                continue
            f1s = [r[f1_key] for r in matched]
            maps = [r[map_key] for r in matched]
            row = {
                "scenario": sc, "variant": variant,
                "f1_mean": f"{np.mean(f1s):.4f}", "f1_std": f"{np.std(f1s):.4f}",
                "mAP_mean": f"{np.mean(maps):.4f}", "mAP_std": f"{np.std(maps):.4f}",
            }
            rows_csv.append(row)
    df = pd.DataFrame(rows_csv)
    df.to_csv(path, index=False)
    print(f"Saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir",
                        default="./embeddings")
    parser.add_argument("--output-dir",
                        default="./results")
    parser.add_argument("--experiments", nargs="+",
                        default=["E5"],
                        help="Which experiments to run (0, A, B, C, D, E1, E3, E4, E5)")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always run diagnostics first
    run_diagnostics(base_dir, output_dir)

    all_results = {}

    if "0" in args.experiments:
        all_results["0"] = experiment_0(base_dir, output_dir)
    if "A" in args.experiments:
        all_results["A"] = experiment_A(base_dir, output_dir)
    if "B" in args.experiments:
        all_results["B"] = experiment_B(base_dir, output_dir)
    if "C" in args.experiments:
        all_results["C"] = experiment_C(base_dir, output_dir)
    if "D" in args.experiments:
        all_results["D"] = experiment_D(base_dir, output_dir)
    if "E1" in args.experiments:
        all_results["E1"] = experiment_E1(base_dir, output_dir)
    if "E3" in args.experiments:
        all_results["E3"] = experiment_E3(base_dir, output_dir)
    if "E4" in args.experiments:
        all_results["E4"] = experiment_E4(base_dir, output_dir)
    if "E5" in args.experiments:
        all_results["E5"] = experiment_E5(base_dir, output_dir)

    _save_json(all_results, output_dir / "results_all_experiments.json")
    print("\n=== ALL DONE ===")


if __name__ == "__main__":
    main()
