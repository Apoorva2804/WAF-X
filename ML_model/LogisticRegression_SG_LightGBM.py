import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, roc_curve, roc_auc_score
)
import matplotlib.pyplot as plt
import pickle
import os
import xgboost as xgb
import lightgbm as lgb
import sys

sys.stdout.reconfigure(encoding='utf-8')

# 📂 Load dataset
csv_path = r".\ML_model\All_data.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Dataset not found at: {csv_path}")
df = pd.read_csv(csv_path)

# 🧹 Preprocessing
if 'class' not in df.columns:
    raise ValueError("Dataset must contain a 'class' column.")
df = df.dropna(subset=['class'])
df.fillna('', inplace=True)

# 🧾 Feature extraction
numeric_features = [
    'single_q', 'double_q', 'dashes', 'braces', 'spaces', 'percentages',
    'semicolons', 'angle_brackets', 'special_chars', 'path_length',
    'body_length', 'badwords_count'
]
for col in numeric_features:
    if col not in df.columns:
        df[col] = 0
df['path'] = df.get('path', '')
df['body'] = df.get('body', '')

badwords = ['badword1', 'badword2']
def extract_features(path, body):
    path_length = len(str(path))
    body_length = len(str(body))
    badword_count = sum(word in str(body).lower() for word in badwords)
    return [path_length, body_length, badword_count]
df['log_features'] = df.apply(lambda r: extract_features(r['path'], r['body']), axis=1)

# 🎯 Train-test split
X_lr = np.array(df['log_features'].tolist())
X_xgb_lgb = df[numeric_features].astype(float)
y = df['class'].astype(int)
X_train_lr, X_test_lr, y_train, y_test = train_test_split(X_lr, y, test_size=0.2, random_state=42)
X_train_xgb, X_test_xgb, _, _ = train_test_split(X_xgb_lgb, y, test_size=0.2, random_state=42)

# ⚙️ Logistic Regression
scaler = StandardScaler()
X_train_lr_scaled = scaler.fit_transform(X_train_lr)
X_test_lr_scaled = scaler.transform(X_test_lr)
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_lr_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_lr_scaled)
y_proba_lr = lr_model.predict_proba(X_test_lr_scaled)[:, 1]

# ⚙️ XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_model.fit(X_train_xgb, y_train)
y_pred_xgb = xgb_model.predict(X_test_xgb)
y_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

# ⚙️ LightGBM
lgb_model = lgb.LGBMClassifier(random_state=42)
lgb_model.fit(X_train_xgb, y_train)
y_pred_lgb = lgb_model.predict(X_test_xgb)
y_proba_lgb = lgb_model.predict_proba(X_test_xgb)[:, 1]

# 📊 Evaluation
def evaluate_model(name, y_true, y_pred):
    print(f"\n{name} Metrics")
    print("-" * 40)
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model("Logistic Regression", y_test, y_pred_lr)
evaluate_model("XGBoost", y_test, y_pred_xgb)
evaluate_model("LightGBM", y_test, y_pred_lgb)

# ======================================================
# 📈 Improved ROC Curve Visualization
# ======================================================
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_proba_lgb)

auc_lr = roc_auc_score(y_test, y_proba_lr)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
auc_lgb = roc_auc_score(y_test, y_proba_lgb)

plt.figure(figsize=(8, 6))
plt.style.use('seaborn-v0_8-whitegrid')

# Plot ROC curves with clear colors & labels
plt.plot(fpr_lr, tpr_lr, color='royalblue', linewidth=2,
         label=f"Logistic Regression (AUC = {auc_lr:.3f})")
plt.plot(fpr_xgb, tpr_xgb, color='darkorange', linewidth=2, linestyle='--',
         label=f"XGBoost (AUC = {auc_xgb:.3f})")
plt.plot(fpr_lgb, tpr_lgb, color='green', linewidth=2, linestyle='-.',
         label=f"LightGBM (AUC = {auc_lgb:.3f})")

# Random classifier line
plt.plot([0, 1], [0, 1], 'k:', label='Random Classifier')

# Zoom in to make differences visible
plt.xlim(0, 0.3)
plt.ylim(0.7, 1.0)

# Labels and title
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve Comparison (Zoomed View)", fontsize=14, fontweight='bold')

# Legend formatting
plt.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)

plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 📉 Enhanced Accuracy Comparison Bar Plot
accuracies = {
    "Logistic Regression": accuracy_score(y_test, y_pred_lr),
    "XGBoost": accuracy_score(y_test, y_pred_xgb),
    "LightGBM": accuracy_score(y_test, y_pred_lgb)
}
plt.figure(figsize=(7, 5))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['skyblue', 'orange', 'lightgreen'])
plt.title("Model Accuracy Comparison", fontsize=14, fontweight='bold')
plt.ylabel("Accuracy", fontsize=12)
plt.ylim(0.8, 1.01)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.3f}", ha='center', fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 💾 Save Models
save_dir = os.path.dirname(csv_path)
os.makedirs(save_dir, exist_ok=True)
with open(os.path.join(save_dir, "logistic_regression_model.pkl"), "wb") as f:
    pickle.dump(lr_model, f)
with open(os.path.join(save_dir, "xgb_model.pkl"), "wb") as f:
    pickle.dump(xgb_model, f)
with open(os.path.join(save_dir, "lgb_model.pkl"), "wb") as f:
    pickle.dump(lgb_model, f)
print("\nAll models saved successfully.")
