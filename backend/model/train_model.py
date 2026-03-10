import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

print("Loading dataset...")
df = pd.read_csv('backend/data/orders.csv')
print(f"   Loaded {len(df)} orders")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────

# Encode categorical columns
df['payment_type_enc'] = (df['payment_type'] == 'COD').astype(int)  # 1 = COD, 0 = Prepaid
df['device_mobile'] = (df['device_type'] == 'mobile').astype(int)   # 1 = mobile

# Encode category
cat_encoder = LabelEncoder()
df['category_enc'] = cat_encoder.fit_transform(df['category'])

# Encode subcategory
subcat_encoder = LabelEncoder()
df['subcategory_enc'] = subcat_encoder.fit_transform(df['subcategory'])

# Derived features
df['is_new_account'] = (df['days_since_account_creation'] < 30).astype(int)
df['is_high_returner'] = (df['past_returns'] >= 4).astype(int)
df['is_bulk_order'] = (df['order_quantity'] >= 3).astype(int)
df['value_per_item'] = df['order_value'] / df['order_quantity']

# Final feature set
FEATURES = [
    'order_value',
    'category_enc',
    'subcategory_enc',
    'payment_type_enc',
    'past_returns',
    'pincode_tier',
    'days_since_account_creation',
    'device_mobile',
    'order_quantity',
    'is_new_account',
    'is_high_returner',
    'is_bulk_order',
    'value_per_item',
]

TARGET = 'returned'

X = df[FEATURES]
y = df[TARGET]

print(f"   Features: {FEATURES}")
print(f"   Return rate in dataset: {y.mean()*100:.1f}%")

# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n   Train size: {len(X_train)} | Test size: {len(X_test)}")

# ─────────────────────────────────────────────
# TRAIN XGBOOST
# ─────────────────────────────────────────────

print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),  # handles class imbalance
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

print("   Training complete ✅")

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n── Model Evaluation ──")
print(classification_report(y_test, y_pred, target_names=['No Return', 'Return']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

# Feature importance
print("\n── Feature Importance ──")
importances = pd.Series(model.feature_importances_, index=FEATURES)
print(importances.sort_values(ascending=False).apply(lambda x: f"{x:.4f}"))

# ─────────────────────────────────────────────
# RISK SCORE FUNCTION
# ─────────────────────────────────────────────
# Converts raw probability (0-1) → risk score (0-100)
# and assigns Low / Medium / High label

def get_risk_label(score):
    if score <= 40:
        return 'Low'
    elif score <= 70:
        return 'Medium'
    else:
        return 'High'

# Quick test on 5 sample orders
print("\n── Sample Predictions ──")
sample = X_test.head(5).copy()
sample['risk_score'] = (model.predict_proba(sample)[:, 1] * 100).astype(int)
sample['risk_label'] = sample['risk_score'].apply(get_risk_label)
sample['actual'] = y_test.head(5).values
print(sample[['risk_score', 'risk_label', 'actual']].to_string(index=False))

# ─────────────────────────────────────────────
# SAVE MODEL + ENCODERS
# ─────────────────────────────────────────────

os.makedirs('backend/model', exist_ok=True)

joblib.dump(model, 'backend/model/shipsure_model.pkl')
joblib.dump(cat_encoder, 'backend/model/category_encoder.pkl')
joblib.dump(subcat_encoder, 'backend/model/subcategory_encoder.pkl')
joblib.dump(FEATURES, 'backend/model/feature_list.pkl')

print("\n✅ Model saved to backend/model/")
print("   shipsure_model.pkl")
print("   category_encoder.pkl")
print("   subcategory_encoder.pkl")
print("   feature_list.pkl")
