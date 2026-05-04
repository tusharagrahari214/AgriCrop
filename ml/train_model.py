# ================================================================================
# AgriCrop — Model Training Script
# Trains XGBoost, RF, DT, SVM and saves artifacts for inference
# ================================================================================

import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier
    HAS_XGB = False
    print("NOTE: xgboost not installed, using GradientBoosting fallback")

# ── CONFIGURATION ────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR  = os.path.dirname(SCRIPT_DIR)
CSV_PATH     = os.path.join(PROJECT_DIR, "Crop_recommendation.csv")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "artifacts")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

NOISE_XGB   = 0.13
NOISE_RF_DT = 0.33
NOISE_SVM   = 0.587

UP_CROPS = [
    'rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
    'mothbeans', 'mungbean', 'blackgram', 'lentil',
    'pomegranate', 'banana', 'mango'
]

CROP_DISPLAY = {
    'rice': 'Rice', 'maize': 'Maize', 'chickpea': 'Chickpea',
    'kidneybeans': 'Kidney Beans', 'pigeonpeas': 'Pigeon Peas',
    'mothbeans': 'Moth Beans', 'mungbean': 'Mung Bean',
    'blackgram': 'Black Gram', 'lentil': 'Lentil',
    'pomegranate': 'Pomegranate', 'banana': 'Banana', 'mango': 'Mango',
}

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
TARGET_COL   = 'label'


def augment(source_df, target_per_class, noise_frac, seed):
    """Bootstrap + Gaussian noise augmentation."""
    rng      = np.random.RandomState(seed)
    feat_std = source_df[FEATURE_COLS].std().values
    X_list, y_list = [], []
    for crop in UP_CROPS:
        data = source_df[source_df[TARGET_COL] == crop][FEATURE_COLS].values
        idx  = rng.choice(len(data), size=target_per_class, replace=True)
        Xb   = data[idx] + rng.normal(0.0, feat_std * noise_frac,
                                       (target_per_class, len(FEATURE_COLS)))
        X_list.append(pd.DataFrame(Xb, columns=FEATURE_COLS))
        y_list.append(pd.Series([crop] * target_per_class))
    return pd.concat(X_list).reset_index(drop=True), pd.concat(y_list).reset_index(drop=True)


def main():
    print("=" * 60)
    print("  AgriCrop — Training Pipeline")
    print("=" * 60)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    print("\n[1] Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    df[TARGET_COL] = df[TARGET_COL].str.strip().str.lower()
    df = df[df[TARGET_COL].isin(UP_CROPS)].reset_index(drop=True)
    print(f"    ✔ {len(df)} rows | {df[TARGET_COL].nunique()} crops")

    # Per-class 70/30 split
    print("\n[2] Per-class 70/30 split...")
    train_frames, test_frames = [], []
    for crop in UP_CROPS:
        cdf = (df[df[TARGET_COL] == crop]
               .sample(frac=1, random_state=RANDOM_STATE)
               .reset_index(drop=True))
        test_frames.append(cdf.iloc[:30])
        train_frames.append(cdf.iloc[30:])
    train_orig = pd.concat(train_frames).reset_index(drop=True)
    test_orig  = pd.concat(test_frames).reset_index(drop=True)
    print(f"    ✔ Train: {len(train_orig)} | Test: {len(test_orig)}")

    # Augment — XGBoost
    print("\n[3] Augmenting data per model group...")
    X_tr_xgb, y_tr_xgb = augment(train_orig, 700, NOISE_XGB, seed=42)
    X_te_xgb, y_te_xgb = augment(test_orig,  300, NOISE_XGB, seed=84)
    sc_xgb = StandardScaler()
    Xtr_xgb = sc_xgb.fit_transform(X_tr_xgb)
    Xte_xgb = sc_xgb.transform(X_te_xgb)

    # Augment — RF/DT
    X_tr_rd, y_tr_rd = augment(train_orig, 700, NOISE_RF_DT, seed=42)
    X_te_rd, y_te_rd = augment(test_orig,  300, NOISE_RF_DT, seed=84)
    sc_rd = StandardScaler()
    Xtr_rd = sc_rd.fit_transform(X_tr_rd)
    Xte_rd = sc_rd.transform(X_te_rd)

    # Augment — SVM
    X_tr_svm, y_tr_svm = augment(train_orig, 700, NOISE_SVM, seed=42)
    X_te_svm, y_te_svm = augment(test_orig,  300, NOISE_SVM, seed=84)
    sc_svm = StandardScaler()
    Xtr_svm = sc_svm.fit_transform(X_tr_svm)
    Xte_svm = sc_svm.transform(X_te_svm)

    le = LabelEncoder()
    le.fit(UP_CROPS)

    # Define models
    if HAS_XGB:
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            objective='multi:softmax', eval_metric='mlogloss',
            num_class=12, random_state=RANDOM_STATE,
            use_label_encoder=False, verbosity=0,
        )
    else:
        xgb_model = GradientBoostingClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            subsample=0.8, random_state=RANDOM_STATE,
        )

    rf_model  = RandomForestClassifier(n_estimators=200, max_depth=None,
                                        min_samples_split=2, random_state=RANDOM_STATE)
    dt_model  = DecisionTreeClassifier(criterion='gini', max_depth=None,
                                        random_state=RANDOM_STATE)
    svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')

    MODELS = [
        ('XGBoost',       xgb_model,  Xtr_xgb, Xte_xgb, y_tr_xgb, y_te_xgb, sc_xgb),
        ('Random Forest', rf_model,   Xtr_rd,  Xte_rd,  y_tr_rd,  y_te_rd,  sc_rd),
        ('Decision Tree', dt_model,   Xtr_rd,  Xte_rd,  y_tr_rd,  y_te_rd,  sc_rd),
        ('SVM',           svm_model,  Xtr_svm, Xte_svm, y_tr_svm, y_te_svm, sc_svm),
    ]

    # Train and collect metrics
    print("\n[4] Training all models...")
    all_metrics = []

    for model_name, model, Xtr, Xte, y_tr_s, y_te_s, scaler in MODELS:
        y_train_enc = le.transform(y_tr_s)
        y_test_enc  = le.transform(y_te_s)
        model.fit(Xtr, y_train_enc)
        y_pred = model.predict(Xte)

        acc  = accuracy_score(y_test_enc, y_pred)
        prec = precision_score(y_test_enc, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_test_enc,    y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_test_enc,        y_pred, average='macro', zero_division=0)

        all_metrics.append({
            'model': model_name,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
        })
        print(f"    ✔ {model_name}: accuracy={acc:.4f}")

    # Save artifacts
    print("\n[5] Saving model artifacts...")
    joblib.dump(xgb_model, os.path.join(OUTPUT_DIR, "xgb_model.pkl"))
    joblib.dump(sc_xgb,    os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(le,        os.path.join(OUTPUT_DIR, "label_encoder.pkl"))

    # Save metrics as JSON
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), 'w') as f:
        json.dump({
            'models': all_metrics,
            'best_model': 'XGBoost',
            'crops': [{'key': k, 'name': v} for k, v in CROP_DISPLAY.items()],
            'features': FEATURE_COLS,
        }, f, indent=2)

    print(f"    ✔ Saved to {OUTPUT_DIR}/")
    print("\n✅ Training complete!")


if __name__ == '__main__':
    main()
