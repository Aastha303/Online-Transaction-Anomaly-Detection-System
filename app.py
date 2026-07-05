"""
Streamlit demo app for the online-transaction fraud-detection SVM model.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud by pointing it at this file in
your GitHub repo (see README.md).
"""

import json
import os

import joblib
import pandas as pd
import streamlit as st

from src.pipeline import RAW_NUMERIC_COLS, clean_raw_data, engineer_features, prepare_single_transaction

MODEL_DIR = "models"

st.set_page_config(page_title="Fraud Detection (SVM)", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_artifacts():
    svm = joblib.load(os.path.join(MODEL_DIR, "svm_model.joblib"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    imputer = joblib.load(os.path.join(MODEL_DIR, "imputer.joblib"))
    with open(os.path.join(MODEL_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return svm, scaler, imputer, metadata


st.title("🛡️ Online Transaction Fraud Detector")
st.caption("SVM (RBF kernel) classifier trained on a risk-based proxy fraud label — see README for methodology and limitations.")

if not os.path.exists(os.path.join(MODEL_DIR, "svm_model.joblib")):
    st.error(
        "No trained model found in `models/`. Run `python train.py` first "
        "(after `python generate_sample_data.py` if you don't have your own CSV)."
    )
    st.stop()

svm, scaler, imputer, metadata = load_artifacts()

with st.expander("Model info", expanded=False):
    st.json(metadata["test_metrics"])
    st.caption(f"Trained on {metadata['trained_on_rows']} transactions · proxy fraud rate {metadata['fraud_rate_pct']}%")

tab_single, tab_batch = st.tabs(["Score a single transaction", "Score a CSV batch"])

with tab_single:
    st.subheader("Enter transaction details")
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction amount (₹)", min_value=0.0, value=2500.0, step=100.0)
        user_median = st.number_input(
            "This user's typical amount (₹)", min_value=0.0, value=2200.0, step=100.0,
            help="Historical median spend for this user. Leave equal to the amount above if unknown."
        )
        tx_type = st.selectbox("Transaction type", ["UPI", "Credit Card", "Debit Card", "Net Banking"])
    with col2:
        device = st.selectbox("Device type", ["Mobile", "Desktop", "Laptop", "Unknown"])
        location = st.text_input("Location", value="Delhi")
        tx_time = st.text_input("Transaction time (HH:MM, 24h)", value="14:20")

    if st.button("Check transaction", type="primary"):
        transaction = {
            "Transaction_Amount": amount,
            "User_Median_Amount": user_median,
            "Transaction_Type": tx_type,
            "Device_Type": device,
            "Location": location or "Unknown",
            "Transaction_Time": tx_time,
        }
        X = prepare_single_transaction(transaction, metadata["feature_columns"], metadata["thresholds"])
        X_scaled = scaler.transform(X)
        prob = svm.predict_proba(X_scaled)[0, 1]
        pred = prob >= 0.5

        if pred:
            st.error(f"⚠️ Flagged as **Fraud / Suspicious** — probability {prob:.1%}")
        else:
            st.success(f"✅ **Not Fraud** — fraud probability {prob:.1%}")
        st.progress(min(float(prob), 1.0))

with tab_batch:
    st.subheader("Upload a CSV of raw transactions")
    st.caption(
        "Expected columns: Transaction_ID, User_ID, Transaction_Amount, Transaction_Type, "
        "Device_Type, Location, Transaction_Time (Balance_Before/After optional)."
    )
    uploaded = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded is not None:
        df_raw = pd.read_csv(uploaded)
        st.write(f"Loaded {len(df_raw)} transactions.")

        df, had_invalid_time = clean_raw_data(df_raw)
        present_cols = [c for c in RAW_NUMERIC_COLS if c in df.columns]
        df[present_cols] = imputer.transform(df[present_cols])

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
        preds = probs >= 0.5

        result = df_raw.copy()
        result["Fraud_Probability"] = probs.round(4)
        result["Prediction"] = ["Fraud / Suspicious" if p else "Not Fraud" for p in preds]

        st.write(f"Flagged **{int(preds.sum())}** of {len(result)} transactions as suspicious.")
        st.dataframe(result.sort_values("Fraud_Probability", ascending=False), use_container_width=True)

        st.download_button(
            "Download scored CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "⚠️ The fraud label used to train this model is a **risk-based proxy**, not confirmed "
    "bank fraud data. This is a portfolio/demo project — a production system would need "
    "verified fraud outcomes and ongoing monitoring."
)
