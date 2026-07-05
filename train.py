"""
Train the fraud-detection SVM model end to end and save deployment artifacts.

Pipeline: Load -> Clean -> Impute -> Risk-Label -> Feature Engineer -> Split
          -> SMOTE (train only) -> Scale (train only) -> Train SVM -> Evaluate -> Save

SVM (RBF kernel) is used because, in the original model comparison against
Random Forest and KNN, it gave the best Recall — the metric prioritized here
since missing an actual fraud case is costlier than a false alarm.

Usage:
    python train.py --data data/unlabeled_messy_transactions.csv --out models/
"""

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.pipeline import (
    RAW_NUMERIC_COLS,
    add_risk_label,
    clean_raw_data,
    engineer_features,
    impute_missing,
)

try:
    from imblearn.over_sampling import SMOTE

    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False


def main(data_path: str, out_dir: str, test_size: float, random_state: int):
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading {data_path} ...")
    df_raw = pd.read_csv(data_path)
    print(f"Shape: {df_raw.shape}")

    print("Cleaning ...")
    df, had_invalid_time = clean_raw_data(df_raw)

    print("Imputing missing numeric values (KNN) ...")
    df, imputer = impute_missing(df, RAW_NUMERIC_COLS)

    print("Building risk-based proxy fraud label ...")
    df, is_fraud = add_risk_label(df, had_invalid_time, random_state=random_state)
    print(f"Fraud rate: {is_fraud.mean() * 100:.1f}% ({is_fraud.sum()} / {len(is_fraud)})")

    # Save the exact percentile thresholds used for High_Amount / High_Relative_Amount,
    # so a single live transaction can be scored consistently at inference time.
    amount_75 = float(df["Transaction_Amount"].quantile(0.75))
    relative_75 = float(df["Relative_Amount"].quantile(0.75))

    print("Engineering features ...")
    df_features = engineer_features(df, had_invalid_time)
    df_features["is_fraud"] = is_fraud

    X = df_features.drop(columns=["is_fraud"])
    y = df_features["is_fraud"]
    feature_columns = list(X.columns)

    print("Splitting (stratified 70/30) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if SMOTE_AVAILABLE:
        print("Applying SMOTE to training data only ...")
        sm = SMOTE(random_state=random_state)
        X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        class_weight = None
    else:
        print("imblearn not available - falling back to class_weight='balanced'")
        X_train_res, y_train_res = X_train, y_train
        class_weight = "balanced"

    print("Scaling (fit on train only) ...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    print("Training SVM (RBF kernel, C=10) ...")
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              random_state=random_state, class_weight=class_weight)
    svm.fit(X_train_scaled, y_train_res)

    y_pred = svm.predict(X_test_scaled)
    y_prob = svm.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    print("\n" + "=" * 50)
    print("  TEST SET RESULTS - SVM (RBF, C=10)")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k:<10}: {v:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"], zero_division=0))

    print(f"Saving artifacts to {out_dir}/ ...")
    joblib.dump(svm, os.path.join(out_dir, "svm_model.joblib"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(imputer, os.path.join(out_dir, "imputer.joblib"))

    metadata = {
        "feature_columns": feature_columns,
        "thresholds": {"amount_75": amount_75, "relative_75": relative_75},
        "test_metrics": metrics,
        "model": "SVC(kernel='rbf', C=10, gamma='scale')",
        "trained_on_rows": int(len(df_raw)),
        "fraud_rate_pct": float(round(is_fraud.mean() * 100, 2)),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done. Artifacts: svm_model.joblib, scaler.joblib, imputer.joblib, metadata.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=str, default="data/unlabeled_messy_transactions.csv")
    parser.add_argument("--out", type=str, default="models")
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    main(args.data, args.out, args.test_size, args.random_state)
