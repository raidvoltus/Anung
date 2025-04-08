import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier

# Load data training
df = pd.read_csv("data/training_data.csv")

# Target dan fitur
X = df.drop(columns=["target"])
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train LightGBM
lgb_model = LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
lgb_model.fit(X_train_scaled, y_train)

# Save model dan scaler
joblib.dump(lgb_model, "data/lgb_model.pkl")
joblib.dump(scaler, "data/scaler.pkl")

print("[INFO] Training selesai dan model disimpan.")
