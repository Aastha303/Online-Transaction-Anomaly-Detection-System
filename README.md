# 🛡️ Online Transaction Anomaly Detection System

An end-to-end Machine Learning application that detects potentially fraudulent online transactions using a Support Vector Machine (SVM). The project includes data preprocessing, feature engineering, model training, batch and single-transaction prediction, and an interactive Streamlit web application.

> **Note:** This project uses a **risk-based proxy fraud label** generated from transaction patterns for demonstration purposes. It is intended as a portfolio project and not for production banking systems.

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://online-transaction-anomaly-detection-system-iydn5oc2bgfappw5jw.streamlit.app/

🔗 **GitHub Repository:** https://github.com/Aastha303/Online-Transaction-Anomaly-Detection-System

---

# 📌 Project Overview

Online payment fraud has become one of the major challenges in digital banking and e-commerce. This project demonstrates how machine learning can identify suspicious transactions by analyzing transaction amount, device information, transaction type, location, and transaction timing.

The system performs:

- Data cleaning
- Missing value handling
- Feature engineering
- Fraud risk labeling
- Model training
- Real-time prediction
- Batch CSV prediction
- Interactive Streamlit dashboard

---

# ✨ Features

- ✅ End-to-end ML pipeline
- ✅ Automated preprocessing
- ✅ Missing value imputation using KNN
- ✅ Feature engineering
- ✅ SVM-based fraud detection
- ✅ Single transaction prediction
- ✅ Batch CSV prediction
- ✅ Download prediction results
- ✅ Interactive Streamlit web application
- ✅ Deployment-ready project structure

---

# 🛠️ Tech Stack

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit

### Machine Learning
- Support Vector Machine (RBF Kernel)

### Deployment
- Streamlit Community Cloud

---

# 📂 Project Structure

```
Online-Transaction-Anomaly-Detection-System/

│
├── app.py                     # Streamlit application
├── train.py                   # Model training pipeline
├── predict.py                 # Prediction pipeline
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   ├── __init__.py
│   └── pipeline.py            # Shared preprocessing & feature engineering
│
├── data/
│   └── unlabeled_messy_transactions.csv
│
├── models/
│   ├── svm_model.joblib
│   ├── scaler.joblib
│   ├── imputer.joblib
│   └── metadata.json
│
└── notebooks/
    └── online_transactions_anomaly_detection.ipynb
```

---

# ⚙️ Machine Learning Pipeline

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Missing Value Imputation
      │
      ▼
Risk Label Generation
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
SMOTE
      │
      ▼
Standard Scaling
      │
      ▼
Support Vector Machine
      │
      ▼
Prediction
```

---

# 📊 Feature Engineering

The following engineered features are created during preprocessing:

- Relative Transaction Amount
- High Amount Flag
- High Relative Amount Flag
- Unknown Location Flag
- Late Night Transaction Flag
- Invalid Time Flag
- Log Transaction Amount
- Transaction Hour
- Transaction Minute

---

# 🤖 Model

Algorithm Used:

- Support Vector Machine (RBF Kernel)

Why SVM?

- Strong performance on non-linear classification
- Effective on medium-sized datasets
- Good generalization ability
- High recall for suspicious transaction detection

---

# 📈 Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score

Evaluation metrics are automatically saved in:

```
models/metadata.json
```

---

# 🧪 Running the Project

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/Online-Transaction-Anomaly-Detection-System.git

cd Online-Transaction-Anomaly-Detection-System
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Train Model

```bash
python train.py
```

This generates:

- svm_model.joblib
- scaler.joblib
- imputer.joblib
- metadata.json

---

## 4. Launch Streamlit App

```bash
streamlit run app.py
```

---

# 📥 Batch Prediction

The application allows users to upload a CSV file containing multiple transactions.

Output:

- Fraud Probability
- Prediction Label

Users can download the scored CSV directly from the application.

---

# 📌 Dataset

This project uses a **synthetic transaction dataset** created specifically for this portfolio project.

The dataset simulates online financial transactions with intentionally introduced:

- Missing values
- Invalid timestamps
- Noisy formatting
- Different transaction types
- Multiple device types
- Multiple locations

to demonstrate a realistic preprocessing and anomaly detection workflow.

---

# ⚠️ Limitations

- The fraud labels are **proxy labels** generated using predefined risk rules.
- The dataset is synthetic and intended for educational purposes.
- A production fraud detection system would require verified historical fraud labels and continuous model monitoring.

---

# 🔮 Future Improvements

- Deep Learning Models
- XGBoost Comparison
- Explainable AI (SHAP)
- Real-time API Deployment
- Database Integration
- User Authentication
- Model Monitoring Dashboard

---

# 👩‍💻 Author

**Aastha**

AI & Machine Learning Student

---

# ⭐ If you found this project useful, consider giving it a Star!
