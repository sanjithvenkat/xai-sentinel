
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from lime.lime_tabular import LimeTabularExplainer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "ids_model.joblib"
DATA_PATH = BASE_DIR / "data" / "traffic_sample.csv"
OUTPUT_DIR = BASE_DIR / "outputs"


def explain_selected_record(row):
    """Generate SHAP and LIME explanations for one original traffic row."""

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    imputer = saved["imputer"]
    scaler = saved["scaler"]
    features = list(saved["features"])
    class_names = saved["class_names"]

    # Ensure the selected row has exactly the expected input columns.
    row = row.reindex(columns=features)
    row = row.apply(pd.to_numeric, errors="coerce")
    row = row.replace([np.inf, -np.inf], np.nan)

    # Use the same preprocessing fitted during model training.
    instance = scaler.transform(imputer.transform(row))

    prediction = int(model.predict(instance)[0])
    probabilities = model.predict_proba(instance)[0]

    # Use a small background sample for manageable SHAP runtime.
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    X = df.reindex(columns=features)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)

    X_processed = scaler.transform(imputer.transform(X))
    rng = np.random.default_rng(42)
    indices = rng.choice(
        len(X_processed),
        size=min(30, len(X_processed)),
        replace=False,
    )
    background = X_processed[indices]

    start = time.perf_counter()
    explainer = shap.KernelExplainer(model.predict_proba, background)
    values = explainer.shap_values(instance, nsamples=100)
    shap_seconds = time.perf_counter() - start

    # Handle common SHAP output formats.
    if isinstance(values, list):
        shap_values = np.asarray(values[1]).reshape(-1)
    else:
        values = np.asarray(values)
        if values.ndim == 3:
            shap_values = values[0, :, 1]
        else:
            shap_values = values.reshape(-1)

    shap_df = pd.DataFrame({
        "Feature": features,
        "SHAP value": shap_values,
    })
    shap_df["Absolute contribution"] = shap_df["SHAP value"].abs()
    shap_df = shap_df.sort_values(
        "Absolute contribution", ascending=False
    )
    shap_df.to_csv(OUTPUT_DIR / "shap_explanation.csv", index=False)

    # Explain the same record with LIME.
    lime_start = time.perf_counter()
    lime_explainer = LimeTabularExplainer(
        training_data=X_processed,
        feature_names=features,
        class_names=[
            class_names[i] for i in sorted(class_names)
        ],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )
    lime_exp = lime_explainer.explain_instance(
        instance[0],
        model.predict_proba,
        num_features=12,
        num_samples=1000,
    )
    lime_seconds = time.perf_counter() - lime_start

    lime_exp.save_to_file(str(OUTPUT_DIR / "lime_explanation.html"))
    lime_df = pd.DataFrame(
        lime_exp.as_list(label=prediction),
        columns=["Feature condition", "LIME weight"],
    )
    lime_df.to_csv(OUTPUT_DIR / "lime_explanation.csv", index=False)

        # Save metadata identifying the record explained
    import json

    metadata = {
        "prediction": class_names[prediction],
        "shap_seconds": round(shap_seconds, 3),
        "lime_seconds": round(lime_seconds, 3),
    }

    with (OUTPUT_DIR / "explanation_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return {
        "prediction": prediction,
        "class_names": class_names,
        "probabilities": probabilities,
        "shap_seconds": shap_seconds,
        "lime_seconds": lime_seconds,
    }
