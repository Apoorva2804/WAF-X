import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# === Load dataset ===
csv_path = r'D:\Apoo\Advance - WAF\ML_model\All_data.csv'

# Check if the CSV exists
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Error: Dataset not found at {csv_path}")

http = pd.read_csv(csv_path)

# === Check for missing values ===
missing_values = http.isna().sum()
print("Missing values per column:\n", missing_values)

# === Ensure required columns exist ===
required_columns = ['path', 'body', 'class']
missing_columns = [col for col in required_columns if col not in http.columns]

if missing_columns:
    raise ValueError(f"Error: Missing columns: {', '.join(missing_columns)}")

# === Fill missing text fields ===
http['path'] = http['path'].fillna('')
http['body'] = http['body'].fillna('')

# === Example badwords list (replace with real list later) ===
badwords = ['badword1', 'badword2']

# === Feature extraction function ===
def ExtractFeatures(path, body):
    path_length = len(str(path))
    body_length = len(str(body))
    badword_count = sum(word in str(body).lower() for word in badwords)
    return [path_length, body_length, badword_count]

# === Extract features for all rows ===
http['features'] = http.apply(lambda row: ExtractFeatures(row['path'], row['body']), axis=1)

# === Prepare data ===
X = np.array(http['features'].tolist())
y = http['class'].values

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train logistic regression model ===
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# === Define save path (same folder as CSV) ===
save_dir = os.path.dirname(csv_path)
model_filename = os.path.join(save_dir, 'ml_model(trained).pkl')

# === Save model ===
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

print(f"\n✅ Model saved successfully at: {model_filename}")

# === Evaluate model ===
y_pred = model.predict(X_test)
print("Logistic Regression Model Evaluation:")
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

