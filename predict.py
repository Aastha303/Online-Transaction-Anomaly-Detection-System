"""
Score transactions for fraud risk using the trained SVM model.

Examples:
    # Single transaction via CLI flags
    python predict.py --amount 85000 --user-median 12000 --type UPI \
        --device Mobile --location Chennai --time 02:15

    # Batch scoring from a CSV with the same raw columns as training
    python predict.py --csv data/new_transactions.csv --out scored.csv
"""

import argparse
import json
import os

import joblib
import pandas as pd

from src.pipeline import (
    RAW_NUMERIC_COLS,
    clean_raw_data,
    engineer_features,
    prepare_single_transaction,
)


def load_artifacts(model_dir: str):
    svm = joblib.load(os.path.join(model_dir, "svm_model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    imputer = joblib.load(os.path.join(model_dir, "imputer.joblib"))
    with open(os.path.join(model_dir, "metadata.json")) as f:
        metadata = json.load(f)
    return svm, scaler, imputer, metadata


def predict_single(transaction: dict, model_dir: str = "models"):
    svm, scaler, _imputer, metadata = load_artifacts(model_dir)
    X = prepare_single_transaction(transaction, metadata["feature_columns"], metadata["thresholds"])
    X_scaled = scaler.transform(X)
    prob = svm.predict_proba(X_scaled)[0, 1]
    pred = int(prob >= 0.5)
    return pred, prob


def predict_batch(df_raw: pd.DataFrame, model_dir: str = "models") -> pd.DataFrame:
    """Score a batch of raw transactions (same columns as the training CSV,
    including User_ID so per-user relative amount can be computed)."""
    svm, scaler, imputer, metadata = load_artifacts(model_dir)

    df, had_invalid_time = clean_raw_data(df_raw)

    # Reuse the fitted training imputer (transform only - no leakage / no refitting).
    present_cols = [c for c in RAW_NUMERIC_COLS if c in df.columns]
    df[present_cols] = imputer.transform(df[present_cols])

    # Fill any still-invalid times with midday as a neutral default.
    from datetime import time as dtime
    df["Time_Cleaned"] = df["Time_Cleaned"].fillna(dtime(12, 0))

    user_median = df.groupby("User_ID")["Transaction_Amount"].transform("median")
    df["Relative_Amount"] = df["Transaction_Amount"] / (user_median + 1)
    thresholds = metadata["thresholds"]
    df["High_Amount"] = (df["Transaction_Amount"] > thresholds["amount_75"]).astype(int)
    df["High_Relative_Amount"] = (df["Relative_Amount"] > thresholds["relative_75"]).astype(int)
    df["Unknown_Location_Flag"] = (df["Location"] == "Unknown").astype(int)
    df.drop(columns=["Balance_Before", "Balance_After"], inplace=True, errors="ignore")

    df_features = engineer_features(df, had_invalid_time)
    X = df_features.reindex(columns=metadata["feature_columns"], fill_value=0)
    X_scaled = scaler.transform(X)

    probs = svm.predict_proba(X_scaled)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = df_raw.copy()
    result["Fraud_Probability"] = probs.round(4)
    result["Prediction"] = ["Fraud / Suspicious" if p else "Not Fraud" for p in preds]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=str, default="models")

    # Single transaction mode
    parser.add_argument("--amount", type=float)
    parser.add_argument("--user-median", type=float, default=None,
                         help="This user's typical (historical median) transaction amount")
    parser.add_argument("--type", type=str, choices=["UPI", "Credit Card", "Debit Card", "Net Banking"])
    parser.add_argument("--device", type=str, choices=["Mobile", "Desktop", "Laptop", "Unknown"])
    parser.add_argument("--location", type=str, default="Unknown")
    parser.add_argument("--time", type=str, help="HH:MM, 24-hour")

    # Batch mode
    parser.add_argument("--csv", type=str, help="Path to a CSV of raw transactions to score")
    parser.add_argument("--out", type=str, default="scored_transactions.csv")

    args = parser.parse_args()

    if args.csv:
        df_raw = pd.read_csv(args.csv)
        result = predict_batch(df_raw, args.model_dir)
        result.to_csv(args.out, index=False)
        print(f"Scored {len(result)} transactions -> {args.out}")
        print(result[["Prediction", "Fraud_Probability"]].value_counts(["Prediction"]))
    elif args.amount is not None:
        transaction = {
            "Transaction_Amount": args.amount,
            "User_Median_Amount": args.user_median if args.user_median is not None else args.amount,
            "Transaction_Type": args.type or "Debit Card",
            "Device_Type": args.device or "Unknown",
            "Location": args.location,
            "Transaction_Time": args.time or "12:00",
        }
        pred, prob = predict_single(transaction, args.model_dir)
        label = "FRAUD / SUSPICIOUS" if pred else "NOT FRAUD"
        print(f"Transaction: {transaction}")
        print(f"\nPrediction: {label}")
        print(f"Fraud probability: {prob:.4f}")
    else:
        parser.error("Provide either --csv for batch scoring or --amount for a single transaction.")
