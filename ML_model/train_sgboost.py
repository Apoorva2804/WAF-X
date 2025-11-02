import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import pickle
import os

# === Load dataset ===
csv_path = r'D:\Apoo\Advance - WAF\ML_model\All_data.csv'

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")

df = pd.read_csv(csv_path)

# === Preprocessing ===
if 'class' not in df.columns:
    raise ValueError("The dataset must contain a 'class' column.")

df = df.dropna(subset=['class'])
df.fillna(0, inplace=True)

# === Feature columns ===
feature_cols = [
    'single_q', 'double_q', 'dashes', 'braces', 'spaces', 'percentages',
    'semicolons', 'angle_brackets', 'special_chars', 'path_length',
    'body_length', 'badwords_count'
]

missing_features = [col for col in feature_cols if col not in df.columns]
if missing_features:
    raise ValueError(f"Missing required feature columns: {', '.join(missing_features)}")

X = df[feature_cols]
y = df['class']

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train XGBoost ===
print("\n🚀 Training XGBoost model...")
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)

# === Train LightGBM ===
print("\n🚀 Training LightGBM model...")
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train, y_train)

# === Evaluate XGBoost ===
print("\n📊 XGBoost Results:")
xgb_pred = xgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, xgb_pred))
print(classification_report(y_test, xgb_pred))

# === Evaluate LightGBM ===
print("\n📊 LightGBM Results:")
lgb_pred = lgb_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, lgb_pred))
print(classification_report(y_test, lgb_pred))

# === Save models ===
save_dir = os.path.dirname(csv_path)
os.makedirs(save_dir, exist_ok=True)

xgb_model_path = os.path.join(save_dir, 'xgb_model.pkl')
lgb_model_path = os.path.join(save_dir, 'lgb_model.pkl')

with open(xgb_model_path, 'wb') as f:
    pickle.dump(xgb_model, f)

with open(lgb_model_path, 'wb') as f:
    pickle.dump(lgb_model, f)

print(f"\n✅ Models saved successfully at:")
print(f"   XGBoost: {xgb_model_path}")
print(f"   LightGBM: {lgb_model_path}")
