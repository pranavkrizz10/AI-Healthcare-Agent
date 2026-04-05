"""
train.py
========
Training pipeline for the large real-encoded symptom-disease dataset
(773 diseases, 377 symptoms, ~246,000 rows).

WHY THIS DATASET IS DIFFERENT FROM THE SMALL KAGGLE ONE
--------------------------------------------------------
The previous Kaggle dataset was synthetic (rules-based, ~120 identical
rows per disease), so any split gave 100% accuracy via memorisation.

THIS dataset is real-encoded with genuine variation:
  - Multiple different patients per disease with different symptom subsets
  - Rows are already distinct partial-symptom observations
  - A standard 80/20 stratified split is correct and meaningful here

TRAINING STRATEGY
-----------------
With 773 classes and ~246k rows, the priority is:
  1. Honest evaluation -- stratified split ensures every disease is in both sets
  2. Neural network implementation -- use a deep feed-forward MLP classifier
  3. Practicality -- keep training manageable on a normal laptop

Primary model: Deep Neural Network (MLPClassifier)
Also available: Bernoulli Naive Bayes (fast baseline)
Optional: Random Forest (comparison baseline)

Usage
-----
    python src/train.py                  # trains DNN + NB + RF, saves best
    python src/train.py --model dnn      # Deep Neural Network only
    python src/train.py --model nb       # Bernoulli Naive Bayes only
    python src/train.py --model rf       # Random Forest baseline only
"""

import os
import time
import argparse
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -- Paths ---------------------------------------------------------------------
PROCESSED_PATH = os.path.join("data", "processed_data.csv")
MODEL_SAVE_PATH = os.path.join("models", "disease_classifier.pkl")

os.makedirs("models", exist_ok=True)

# Minimum rows a disease must have to be included in training.
# Diseases with fewer rows than this can't be reliably split 80/20.
MIN_ROWS_PER_CLASS = 5


# -- Data Loading --------------------------------------------------------------
def load_data(path: str = PROCESSED_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed data not found at '{path}'.\n"
            "Run python src/preprocess.py first."
        )
    print(f"[train] Loading '{path}' ...")
    t0 = time.time()
    df = pd.read_csv(path, low_memory=False)
    print(
        f"[train] Loaded  {df.shape[0]:,} rows x {df.shape[1]} columns ({time.time()-t0:.1f}s)")

    counts = df["disease"].value_counts()
    rare = counts[counts < MIN_ROWS_PER_CLASS].index
    if len(rare):
        df = df[~df["disease"].isin(rare)]
        print(
            f"[train] Dropped {len(rare)} diseases with <{MIN_ROWS_PER_CLASS} rows. "
            f"Remaining: {df['disease'].nunique()} diseases, {len(df):,} rows."
        )

    feature_cols = [c for c in df.columns if c != "disease"]
    return df, feature_cols


# -- Split ---------------------------------------------------------------------
def split_data(df: pd.DataFrame, feature_cols: list, test_size: float = 0.2):
    """Stratified 80/20 split for real distinct patient observations."""
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["disease"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    print(
        f"[train] Split  --> train={len(X_train):,}  test={len(X_test):,}  classes={len(le.classes_)}")

    train_set = set(map(tuple, X_train[:5000]))
    sample_test = X_test[:1000]
    leaking = sum(1 for row in sample_test if tuple(row) in train_set)
    leak_pct = (leaking / len(sample_test) * 100) if len(sample_test) else 0.0
    if leak_pct > 50:
        print(
            f"[train] WARNING: ~{leak_pct:.0f}% of sampled test rows match train rows. Dataset may have high duplication.")
    else:
        print(
            f"[train] Leakage check (sample): ~{leak_pct:.0f}% overlap -- looks fine.")

    return X_train, X_test, y_train, y_test, le


# -- Model Definitions ---------------------------------------------------------
def get_models(choice: str = "all") -> dict:
    """
    DNN / MLPClassifier : Required neural-network implementation. Deep feed-forward network.
    BernoulliNB         : Fast baseline for binary features.
    RandomForest        : Optional comparison baseline.
    """
    catalogue = {
        "dnn": (
            "Deep Neural Network",
            MLPClassifier(
                hidden_layer_sizes=(512, 256, 128),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                batch_size=256,
                learning_rate_init=1e-3,
                max_iter=80,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=8,
                random_state=42,
                verbose=True,
            ),
        ),
        "nb": (
            "Bernoulli Naive Bayes",
            BernoulliNB(
                alpha=1.0,
                binarize=0.5,
            ),
        ),
        "rf": (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=250,
                min_samples_leaf=2,
                max_features="sqrt",
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
                verbose=0,
            ),
        ),
    }

    if choice == "all":
        return {k: catalogue[k] for k in ("dnn", "nb", "rf")}
    if choice not in catalogue:
        raise ValueError(
            f"Unknown model '{choice}'. Available: {list(catalogue.keys())}")
    return {choice: catalogue[choice]}


# -- Evaluation ----------------------------------------------------------------
def evaluate(model, X_test, y_test, le, model_name: str) -> dict:
    t0 = time.time()
    y_pred = model.predict(X_test)
    elapsed = time.time() - t0

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"\n{'=' * 60}")
    print(f"  {model_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy        : {acc * 100:.2f}%")
    print(
        f"  F1 (weighted)   : {f1 * 100:.2f}%   <- main metric (handles imbalance)")
    print(
        f"  F1 (macro)      : {f1_macro * 100:.2f}%   <- unweighted across all classes")
    print(f"  Inference time  : {elapsed:.2f}s on {len(X_test):,} samples")

    report = classification_report(
        y_test,
        y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )
    per_class = {
        label: stats["f1-score"]
        for label, stats in report.items()
        if label not in ("accuracy", "macro avg", "weighted avg")
    }
    worst = sorted(per_class.items(), key=lambda x: x[1])[:10]
    print("\n  10 hardest-to-predict diseases (lowest F1):")
    for disease, score in worst:
        print(f"    {disease:<45} F1={score:.2f}")

    return {
        "name": model_name,
        "accuracy": acc,
        "f1_weighted": f1,
        "f1_macro": f1_macro,
    }


# -- Main Training Pipeline ----------------------------------------------------
def train_all(choice: str = "all"):
    df, feature_cols = load_data()
    X_train, X_test, y_train, y_test, le = split_data(df, feature_cols)

    models = get_models(choice)
    results = []
    best_model = None
    best_score = -1.0
    best_name = ""

    for _, (name, clf) in models.items():
        print(f"\n[train] Fitting {name} ...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        elapsed = time.time() - t0
        print(f"[train] Training time : {elapsed:.1f}s")

        metrics = evaluate(clf, X_test, y_test, le, name)
        results.append(metrics)

        if metrics["f1_weighted"] > best_score:
            best_score = metrics["f1_weighted"]
            best_model = clf
            best_name = name

    bundle = {
        "model": best_model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "model_name": best_name,
        "f1_weighted": best_score,
        "n_classes": len(le.classes_),
    }
    joblib.dump(bundle, MODEL_SAVE_PATH, compress=3)
    print(
        f"\n[train] Best model  : {best_name}  (weighted F1={best_score * 100:.2f}%)")
    print(f"[train] Saved       : {MODEL_SAVE_PATH}")
    print("\n[train] Neural-network implementation is now enabled.")

    return bundle, results


# -- Entry Point ---------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train disease classifier on the large encoded symptom dataset."
    )
    parser.add_argument(
        "--model",
        default="all",
        choices=["dnn", "nb", "rf", "all"],
        help="Which model to train (default: all).",
    )
    args = parser.parse_args()
    train_all(args.model)
