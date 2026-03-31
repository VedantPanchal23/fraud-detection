# Credit Card Fraud Detection System
### ML InnovateX Hackathon | Theme: Credit Card Fraud Detection (ML + ANN + Deployment)

---

## Problem Statement
Build a robust, real-world fraud detection system using Machine Learning models and an
Artificial Neural Network (ANN) on the Credit Card Fraud Detection dataset from Kaggle.

---

## Dataset
- Source: [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions | 492 frauds (0.17% — highly imbalanced)
- Features: V1–V28 (PCA transformed), Time, Amount, Class

---

## Project Structure
```
fraud-detection-hackathon/
│
├── data/
│   └── creditcard.csv
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── models/
│   ├── xgb_model.pkl
│   ├── scaler.pkl
│   ├── feature_selector.pkl
│   ├── selected_features.pkl
│   └── ann_model.h5
│
├── app/
│   └── app.py
│
├── outputs/
│   ├── model_comparison.csv
│   └── plots/
│
├── requirements.txt
└── README.md
```

---

## Approach

### 1. Data Understanding & Cleaning
- No missing values found
- Severe class imbalance identified — 0.17% fraud
- Explored distributions of Amount and Time by class

### 2. Exploratory Data Analysis
- Univariate and bivariate analysis on all features
- Fraud transactions tend to have significantly lower amounts
- Key fraud indicators identified — V14, V12, V10, V17, V11

### 3. Preprocessing
- Log transformation on Amount to reduce skewness
- Hour and is_night features extracted from Time
- StandardScaler applied on all features
- SMOTE applied on training set only to handle class imbalance
- Stratified 80/20 train-test split

### 4. Feature Engineering & Selection
- Created Amount_log, Hour, is_night
- SelectFromModel with Random Forest reduced 31 features to 16
- Selected: V2, V3, V4, V7, V8, V9, V10, V11, V12, V14, V16, V17, V18, V19, V21, Amount_log

### 5. Machine Learning Models Trained
| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 0.0527 | 0.9184 | 0.0996 | 0.9714 |
| Random Forest | 0.7692 | 0.8163 | 0.7921 | 0.9638 |
| XGBoost | 0.6159 | 0.8673 | 0.7203 | 0.9763 |
| LightGBM | 0.3281 | 0.8571 | 0.4746 | 0.9466 |
| XGBoost Tuned | 0.7679 | 0.8776 | 0.8190 | 0.9760 |
| Random Forest Tuned | 0.7619 | 0.8163 | 0.7882 | 0.9600 |
| ANN | 0.1559 | 0.9082 | 0.2661 | 0.9719 |

### 6. ANN Architecture
- Input → Dense(128) → BatchNorm → Dropout(0.3)
- Dense(64) → BatchNorm → Dropout(0.2)
- Dense(32) → BatchNorm → Dropout(0.2)
- Dense(16) → Dense(1, sigmoid)
- Optimizer: Adam(lr=0.001) | Loss: binary_crossentropy
- EarlyStopping + ReduceLROnPlateau callbacks

### 7. Hyperparameter Tuning
- RandomizedSearchCV with 5-fold StratifiedKFold
- Scoring metric: ROC-AUC
- Best XGBoost params: n_estimators=300, max_depth=6, learning_rate=0.2

### 8. Overfitting Analysis
- Tree models show mild overfitting gap of 0.02–0.05
- Test AUC remains 0.976 — excellent generalization
- Regularization applied: reg_alpha=0.1, reg_lambda=1.5, max_depth reduced to 4

### 9. Final Model — XGBoost Tuned
Selected based on:
- Highest F1 score (0.819)
- Highest Recall (0.877) — catches 88% of all frauds
- ROC-AUC 0.976 on unseen test data
- Fast inference suitable for real-time deployment

---

## Why Recall Over Precision
In fraud detection, missing a fraud (false negative) causes direct financial loss.
A false alarm (false positive) only inconveniences a customer.
Therefore Recall is the primary optimization metric.

---

## Deployment
- Framework: Streamlit
- Platform: Streamlit Community Cloud (free)
- Live App: [Click Here](YOUR_DEPLOYED_APP_LINK)

### How to Run Locally
```bash
git clone https://github.com/VedantPanchal23/fraud-detection-hackathon.git
cd fraud-detection-hackathon
pip install -r requirements.txt
cd app
streamlit run app.py
```

---

## Tech Stack
- Python 3.x
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- xgboost, lightgbm
- tensorflow / keras
- streamlit, joblib

---

## Author
- Hackathon: ML InnovateX
- Theme: Credit Card Fraud Detection