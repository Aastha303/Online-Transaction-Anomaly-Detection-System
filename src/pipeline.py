"""
Shared preprocessing / feature-engineering logic for the fraud-detection pipeline.

This module is imported by both `train.py` (fitting) and `predict.py` / `app.py`
(inference), so training and serving can never drift apart. It mirrors the logic
from the original research notebook (`notebooks/online_transactions_anomaly_detection2.ipynb`)
but is reorganized into reusable, testable functions.

Pipeline stages:
    1. clean_raw_data       -> fix amount/time/category columns
    2. impute_missing       -> KNN-impute numeric columns
    3. add_risk_label       -> build the risk-based proxy `is_fraud` label (TRAINING ONLY)
    4. engineer_features    -> derive model-ready numeric/categorical features
    5. align_columns        -> one-hot encode + align to the training feature set (INFERENCE)
"""

from __future__ import annotations

from datetime import time as dtime
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer

RAW_NUMERIC_COLS = ["Transaction_Amount", "Balance_Before", "Balance_After"]
CATEGORICAL_COLS = ["Transaction_Type", "Device_Type", "Location"]
RISKY_TRANSACTION_TYPES = ["Credit Card", "UPI"]
RISK_THRESHOLD = 3.5


def clean_time(value) -> Optional[dtime]:
    """Parse an 'HH:MM' string into a datetime.time, or None if invalid (e.g. '25:61')."""
    if pd.isna(value):
        return None
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
        if 0 <= h <= 23 and 0 <= m <= 59:
            return dtime(h, m)
        return None
    except ValueError:
        return None


def clean_raw_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Standardize raw columns: strip currency symbols from amount, validate time,
    fill categorical NaNs with 'Unknown'.

    Returns
    -------
    df : cleaned dataframe (Time_Cleaned added, Transaction_Time kept for reference)
    had_invalid_time : boolean Series, True where the original time could not be parsed
    """
    df = df_raw.copy()

    df["Transaction_Amount"] = (
        df["Transaction_Amount"]
        .astype(str)
        .str.replace("₹", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df["Transaction_Amount"] = pd.to_numeric(df["Transaction_Amount"], errors="coerce")

    df["Time_Cleaned"] = df["Transaction_Time"].apply(clean_time)
    had_invalid_time = df["Time_Cleaned"].isna()

    for col in ["Device_Type", "Location"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    return df, had_invalid_time


def impute_missing(df: pd.DataFrame, numeric_cols: list[str] = RAW_NUMERIC_COLS) -> tuple[pd.DataFrame, KNNImputer]:
    """KNN-impute the given numeric columns (fit on this data) and backfill invalid times
    with the modal (hour, minute)."""
    df = df.copy()
    present_cols = [c for c in numeric_cols if c in df.columns]

    imputer = KNNImputer(n_neighbors=5)
    df[present_cols] = imputer.fit_transform(df[present_cols])

    if df["Time_Cleaned"].notna().any():
        common_hour = df["Time_Cleaned"].dropna().apply(lambda x: x.hour).mode()[0]
        common_min = df["Time_Cleaned"].dropna().apply(lambda x: x.minute).mode()[0]
    else:
        common_hour, common_min = 12, 0
    df["Time_Cleaned"] = df["Time_Cleaned"].fillna(dtime(common_hour, common_min))

    return df, imputer


def add_risk_label(
    df: pd.DataFrame, had_invalid_time: pd.Series, random_state: int = 42
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build the risk-based proxy fraud label. TRAINING ONLY — a real deployment
    would replace this with confirmed fraud outcomes once they're available.

    Also attaches the intermediate risk-signal columns to `df` so they can be
    reused as model features later.
    """
    df = df.copy()

    user_median = df.groupby("User_ID")["Transaction_Amount"].transform("median")
    df["Relative_Amount"] = df["Transaction_Amount"] / (user_median + 1)

    amount_75 = df["Transaction_Amount"].quantile(0.75)
    relative_75 = df["Relative_Amount"].quantile(0.75)
    hour_arr = df["Time_Cleaned"].apply(lambda x: x.hour if pd.notna(x) else 12)

    df["High_Amount"] = (df["Transaction_Amount"] > amount_75).astype(int)
    df["High_Relative_Amount"] = (df["Relative_Amount"] > relative_75).astype(int)
    df["Unknown_Location_Flag"] = (df["Location"] == "Unknown").astype(int)
    late_night_signal = ((hour_arr >= 1) & (hour_arr <= 4)).astype(int)
    risky_type_signal = df["Transaction_Type"].isin(RISKY_TRANSACTION_TYPES).astype(int)
    mobile_signal = (df["Device_Type"] == "Mobile").astype(int)

    risk_score = (
        2.0 * df["High_Relative_Amount"]
        + 2.0 * df["High_Amount"]
        + 1.5 * df["Unknown_Location_Flag"]
        + 1.5 * had_invalid_time.astype(int).values
        + 1.0 * late_night_signal.values
        + 1.0 * risky_type_signal.values
        + 1.0 * mobile_signal.values
    )

    is_fraud = (risk_score >= RISK_THRESHOLD).astype(int).to_numpy().copy()

    rng = np.random.RandomState(random_state)
    noise_idx = rng.choice(np.arange(len(is_fraud)), size=int(len(is_fraud) * 0.05), replace=False)
    is_fraud[noise_idx] = 1 - is_fraud[noise_idx]

    df.drop(columns=["Balance_Before", "Balance_After"], inplace=True, errors="ignore")
    return df, is_fraud


def engineer_features(df: pd.DataFrame, had_invalid_time: pd.Series) -> pd.DataFrame:
    """Derive time/amount features and drop identifier/raw columns not used by the model.
    Assumes risk-flag columns (High_Amount, etc.) already exist — call add_risk_label first
    during training, or compute the same flags for a single live transaction before this step."""
    df = df.copy()

    df["Transaction_Hour"] = df["Time_Cleaned"].apply(lambda x: x.hour if pd.notna(x) else 0)
    df["Transaction_Minute"] = df["Time_Cleaned"].apply(lambda x: x.minute if pd.notna(x) else 0)
    df["Late_Night"] = ((df["Transaction_Hour"] >= 1) & (df["Transaction_Hour"] <= 4)).astype(int)
    df["Log_Amount"] = np.log1p(df["Transaction_Amount"])
    df["Had_Invalid_Time"] = had_invalid_time.astype(int).values

    df.drop(
        columns=["Transaction_ID", "User_ID", "Transaction_Time", "Time_Cleaned"],
        inplace=True,
        errors="ignore",
    )
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=True)
    return df


def align_columns(df_encoded: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Reindex a one-hot-encoded dataframe to exactly match the training feature columns
    (missing columns filled with 0, extra columns dropped, order preserved)."""
    return df_encoded.reindex(columns=feature_columns, fill_value=0)


def prepare_single_transaction(transaction, feature_columns, thresholds):
    """
    Build a model-ready single-row dataframe from a raw transaction dict, e.g.:

        {
            "Transaction_Amount": 85000,
            "User_Median_Amount": 12000,   # caller's own historical median for this user
            "Transaction_Type": "UPI",
            "Device_Type": "Mobile",
            "Location": "Chennai",
            "Transaction_Time": "02:15",
        }

    `User_Median_Amount` substitutes for the per-user median that training computes
    from history, since a single live transaction has no history of its own.

    `thresholds` must contain 'amount_75' and 'relative_75', the percentile cutoffs
    saved from the training set (a single row has no distribution of its own to
    compute percentiles from).
    """
    row = pd.DataFrame([transaction])
    row["Time_Cleaned"] = row["Transaction_Time"].apply(clean_time)
    had_invalid_time = row["Time_Cleaned"].isna()
    row["Time_Cleaned"] = row["Time_Cleaned"].fillna(dtime(12, 0))

    for col in ["Device_Type", "Location"]:
        if col in row.columns:
            row[col] = row[col].fillna("Unknown")

    user_median = row["User_Median_Amount"] if "User_Median_Amount" in row.columns else row["Transaction_Amount"]
    row["Relative_Amount"] = row["Transaction_Amount"] / (user_median + 1)
    row.drop(columns=["User_Median_Amount"], inplace=True, errors="ignore")

    row["Transaction_Hour"] = row["Time_Cleaned"].apply(lambda x: x.hour if pd.notna(x) else 0)
    row["Transaction_Minute"] = row["Time_Cleaned"].apply(lambda x: x.minute if pd.notna(x) else 0)
    row["Late_Night"] = ((row["Transaction_Hour"] >= 1) & (row["Transaction_Hour"] <= 4)).astype(int)
    row["Log_Amount"] = np.log1p(row["Transaction_Amount"])
    row["Had_Invalid_Time"] = had_invalid_time.astype(int).values
    row["Unknown_Location_Flag"] = (row["Location"] == "Unknown").astype(int)
    row["High_Amount"] = (row["Transaction_Amount"] > thresholds["amount_75"]).astype(int)
    row["High_Relative_Amount"] = (row["Relative_Amount"] > thresholds["relative_75"]).astype(int)

    row.drop(columns=["Transaction_ID", "User_ID", "Transaction_Time", "Time_Cleaned"],
              inplace=True, errors="ignore")
    row_encoded = pd.get_dummies(row, columns=CATEGORICAL_COLS, drop_first=True)
    return align_columns(row_encoded, feature_columns)
