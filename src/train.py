
import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

DATA_PATH = "data/traffic_sample.csv"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("Loading sample...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

# Remove missing labels
df = df.dropna(subset=["Label"]).copy()

# Binary target: Benign = 0, Bot = 1
labels = df["Label"].astype(str).str.strip().str.lower()
y = labels.map({"benign": 0, "bot": 1})

if y.isna().any():
    unknown = labels[y.isna()].unique().tolist()
    raise ValueError(f"Unexpected labels: {unknown}")

# Use numeric traffic features; exclude target and timestamp
X = df.drop(columns=["Label", "Timestamp"], errors="ignore")
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Drop entirely empty and constant columns
X = X.dropna(axis=1, how="all")
X = X.loc[:, X.nunique(dropna=True) > 1]

if X.shape[1] == 0:
    raise ValueError("No usable numeric features were found.")

print("Rows:", len(X))
print("Features:", X.shape[1])
print("Labels:")
print(y.value_counts().rename(index={0: "Benign", 1: "Bot"}))

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Fit preprocessing only on training data
imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\nTraining neural network...")
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=30,
    early_stopping=True,
    random_state=42
)

model.fit(X_train, y_train)


pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
report = classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["Benign", "Bot"],
    output_dict=True,
    zero_division=0,
)
matrix = confusion_matrix(y_test, pred, labels=[0, 1])

print("\nAccuracy:", round(accuracy, 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    pred,
    labels=[0, 1],
    target_names=["Benign", "Bot"],
    zero_division=0,
))

print("Confusion matrix (rows=true, columns=predicted):")
print(matrix)

# Save evaluation results for the dashboard
joblib.dump(
    {
        "accuracy": float(accuracy),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "class_names": ["Benign", "Bot"],
    },
    "outputs/evaluation_results.joblib",
)

print("Evaluation results saved to outputs/evaluation_results.joblib")


# Save model and preprocessing components together
joblib.dump(
    {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": X.columns.tolist(),
        "class_names": {0: "Benign", 1: "Bot"}
    },
    "models/ids_model.joblib"
)

# Save a small set of original, unscaled features for the dashboard
demo = X.iloc[:100].copy()
demo["Label"] = y.iloc[:100].map({0: "Benign", 1: "Bot"})
demo.to_csv("outputs/demo_flows.csv", index=False)

print("\nModel saved to models/ids_model.joblib")
print("Demo flows saved to outputs/demo_flows.csv")
