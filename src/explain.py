
import time
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from lime.lime_tabular import LimeTabularExplainer

MODEL_PATH = "models/ids_model.joblib"
DATA_PATH = "data/traffic_sample.csv"

print("Loading model and data...")
saved = joblib.load(MODEL_PATH)
model = saved["model"]
imputer = saved["imputer"]
scaler = saved["scaler"]
features = saved["features"]

df = pd.read_csv(DATA_PATH, low_memory=False)
df.columns = df.columns.str.strip()

X = df.reindex(columns=features)
X = X.apply(pd.to_numeric, errors="coerce")
X = X.replace([np.inf, -np.inf], np.nan)

# Transform using the same preprocessing fitted during training
X_processed = scaler.transform(imputer.transform(X))

# Explain one Bot flow if available, otherwise the first flow
labels = df["Label"].astype(str).str.strip().str.lower()
bot_indices = np.flatnonzero(labels.to_numpy() == "bot")
row_index = int(bot_indices[0]) if len(bot_indices) else 0

instance = X_processed[row_index:row_index + 1]
prediction = int(model.predict(instance)[0])
probabilities = model.predict_proba(instance)[0]

print("\nSelected flow index:", row_index)
print("Predicted class:", saved["class_names"][prediction])
print("Class probabilities:", {
    saved["class_names"][int(cls)]: round(float(prob), 4)
    for cls, prob in zip(model.classes_, probabilities)
})

# SHAP: use a small background sample to keep computation manageable
rng = np.random.default_rng(42)
background_indices = rng.choice(
    len(X_processed),
    size=min(30, len(X_processed)),
    replace=False
)
background = X_processed[background_indices]

print("\nGenerating SHAP explanation...")
start = time.perf_counter()

explainer = shap.KernelExplainer(
    model.predict_proba,
    background
)
shap_values = explainer.shap_values(
    instance,
    nsamples=100
)

elapsed = time.perf_counter() - start
print(f"SHAP runtime: {elapsed:.2f} seconds")

# Support common SHAP output formats
if isinstance(shap_values, list):
    values = np.asarray(shap_values[1]).reshape(-1)
else:
    values = np.asarray(shap_values)
    if values.ndim == 3:
        # Usually (samples, features, classes)
        values = values[0, :, 1]
    else:
        values = values.reshape(-1)

shap_df = pd.DataFrame({
    "Feature": features,
    "SHAP value": values
})
shap_df["Absolute contribution"] = shap_df["SHAP value"].abs()
shap_df = shap_df.sort_values(
    "Absolute contribution",
    ascending=False
)
shap_df.to_csv("outputs/shap_explanation.csv", index=False)

top_shap = shap_df.head(12).sort_values("SHAP value")
plt.figure(figsize=(9, 6))
plt.barh(top_shap["Feature"], top_shap["SHAP value"])
plt.axvline(0, linewidth=0.8)
plt.xlabel("SHAP value (effect on class probability)")
plt.title("SHAP explanation for one traffic flow")
plt.tight_layout()
plt.savefig("outputs/shap_explanation.png", dpi=160)
plt.close()

print("Saved SHAP results to outputs/shap_explanation.csv")
print("Saved SHAP plot to outputs/shap_explanation.png")

# LIME: explain the same flow in the model's scaled feature space
print("\nGenerating LIME explanation...")
lime_start = time.perf_counter()

lime_explainer = LimeTabularExplainer(
    training_data=X_processed,
    feature_names=features,
    class_names=["Benign", "Bot"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_exp = lime_explainer.explain_instance(
    instance[0],
    model.predict_proba,
    num_features=12,
    num_samples=1000
)

lime_elapsed = time.perf_counter() - lime_start
print(f"LIME runtime: {lime_elapsed:.2f} seconds")

lime_exp.save_to_file("outputs/lime_explanation.html")

lime_df = pd.DataFrame(
    lime_exp.as_list(label=prediction),
    columns=["Feature condition", "LIME weight"]
)
lime_df.to_csv("outputs/lime_explanation.csv", index=False)

print("Saved LIME results to outputs/lime_explanation.html")
print("Saved LIME results to outputs/lime_explanation.csv")
print("\nExplanation generation complete.")
