import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ---------------- Paths ---------------- #
data_path = os.path.join(os.path.dirname(__file__), "..", "dataset")
models_path = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(models_path, exist_ok=True)

# ---------------- Load Dataset ---------------- #
gesture_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
dfs = []
for file in gesture_files:
    df = pd.read_csv(os.path.join(data_path, file))
    df["gesture"] = file.split(".")[0]
    dfs.append(df)

X = pd.concat(dfs, ignore_index=True)
y = X.pop("gesture")

# ---------------- Handle NaN ---------------- #
X = X.fillna(0)  # Missing values ko 0 se replace kar diya


# ---------------- Scaling & Encoding ---------------- #
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------- Split Dataset ---------------- #
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# ---------------- Train Random Forest ---------------- #
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

# ---------------- Train SVM ---------------- #
svm_model = SVC(kernel="rbf", probability=True, random_state=42)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))
print("\nSVM Classification Report:\n", classification_report(y_test, y_pred_svm))

# ---------------- Save Models & Scaler/Encoder ---------------- #
joblib.dump(scaler, os.path.join(models_path, "scaler.pkl"))
joblib.dump(le, os.path.join(models_path, "label_encoder.pkl"))
joblib.dump(rf_model, os.path.join(models_path, "rf_model.pkl"))
joblib.dump(svm_model, os.path.join(models_path, "svm_model.pkl"))

print(f"\n✅ Models, scaler & label encoder saved in {models_path}")
