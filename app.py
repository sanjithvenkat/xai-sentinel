
from pathlib import Path

import pandas as pd
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "ids_model.joblib"

st.set_page_config(
    page_title="Explainable SDN Intrusion Detection",
    page_icon="",
    layout="wide",
)

st.title("Explainable AI for SDN Intrusion Detection")
st.caption(
    "Machine-learning predictions, SHAP/LIME explanations, "
    "and Mininet/Open vSwitch monitoring"
)

st.info(
    "Scope note: AI predictions currently use CIC-IDS2018 dataset "
    "features. Live OpenFlow statistics are displayed separately; "
    "they are not yet fed into the trained classifier."
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section",
    [
    "Overview",
    "Intrusion Detection",
    "Model Evaluation",
    "SHAP Explanation",
    "LIME Explanation",
    "Live SDN Monitoring",
],
)

# Safe CSV loading
def load_csv(filename):
    path = OUTPUT_DIR / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {filename}: {exc}")
        return None


if page == "Overview":
    st.header("Project Overview")

    model_exists = MODEL_PATH.exists()
    shap_exists = (OUTPUT_DIR / "shap_explanation.csv").exists()
    lime_exists = (OUTPUT_DIR / "lime_explanation.csv").exists()
    flow_exists = (OUTPUT_DIR / "flow_stats.csv").exists()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trained model", "Available" if model_exists else "Missing")
    c2.metric("SHAP output", "Available" if shap_exists else "Missing")
    c3.metric("LIME output", "Available" if lime_exists else "Missing")
    c4.metric("SDN flow data", "Available" if flow_exists else "Missing")

    st.subheader("System Architecture")
    st.markdown(
        """
        1. **Traffic dataset:** CIC-IDS2018 sampled traffic records.
        2. **Detection model:** trained MLP classifier.
        3. **Explainability:** SHAP and LIME feature explanations.
        4. **SDN monitoring:** Mininet and Open vSwitch flow counters.
        """
    )

    st.subheader("Evaluation note")
    st.warning(
        "The previously observed 99.98% accuracy came from a held-out "
        "split of the same sampled dataset. It does not establish "
        "performance on unseen networks or live traffic."
    )



elif page == "Intrusion Detection":
    st.header("Intrusion Detection")

    demo = load_csv("demo_flows.csv")

    if demo is None or demo.empty:
        st.warning("outputs/demo_flows.csv was not found or is empty.")
    else:
        st.write("Sample records used for the model demonstration.")
        st.dataframe(demo.head(100), use_container_width=True)

        label_col = next(
            (col for col in demo.columns if col.lower() == "label"),
            None,
        )

        if label_col:
            st.subheader("Dataset Label Distribution")
            st.bar_chart(demo[label_col].value_counts())

        st.divider()
        st.subheader("Run AI Prediction")

        st.write(
            "Select a traffic record and run it through the saved "
            "classifier to obtain a new prediction."
        )

        if MODEL_PATH.exists():
            import joblib

            saved = joblib.load(MODEL_PATH)
            features = list(saved["features"])
            model = saved["model"]
            imputer = saved["imputer"]
            scaler = saved["scaler"]
            class_names = saved["class_names"]

            missing_features = [
                feature for feature in features
                if feature not in demo.columns
            ]

            if missing_features:
                st.error(
                    "Prediction unavailable: the dataset is missing "
                    f"{len(missing_features)} required model features."
                )
                st.write("Missing features:", missing_features)
            else:
                row_index = st.selectbox(
                    "Choose a traffic record",
                    options=list(range(len(demo))),
                    format_func=lambda i: f"Record {i}",
                )

                if st.button("Run Prediction", type="primary"):
                    try:
                        row = demo.iloc[[row_index]][features].copy()
                        row = row.apply(
                            pd.to_numeric, errors="coerce"
                        )
                        transformed = imputer.transform(row)
                        transformed = scaler.transform(transformed)

                        prediction = model.predict(transformed)[0]
                        predicted_label = class_names.get(
                            int(prediction), str(prediction)
                        )

                        st.subheader("Prediction Result")
                        st.success(f"Predicted class: {predicted_label}")

                        from src.explain_selected import explain_selected_record

                        with st.spinner("Generating SHAP and LIME explanations..."):
                            explanation = explain_selected_record(
                                demo.iloc[[row_index]][features]
                            )

                        st.success("SHAP and LIME explanations generated for this record.")
                        st.write(
                            f"SHAP runtime: {explanation['shap_seconds']:.2f} seconds"
                        )
                        st.write(
                            f"LIME runtime: {explanation['lime_seconds']:.2f} seconds"
                        )    

                        if hasattr(model, "predict_proba"):
                            probabilities = model.predict_proba(
                                transformed
                            )[0]
                            probability_data = pd.DataFrame({
                                "Class": [
                                    class_names.get(int(c), str(c))
                                    for c in model.classes_
                                ],
                                "Probability": probabilities,
                            })
                            probability_data["Probability (%)"] = (
                                probability_data["Probability"] * 100
                            ).round(2)

                            st.dataframe(
                                probability_data[
                                    ["Class", "Probability (%)"]
                                ],
                                hide_index=True,
                                use_container_width=True,
                            )
                            st.bar_chart(
                                probability_data.set_index("Class")[
                                    "Probability (%)"
                                ]
                            )

                        st.caption(
                            "This prediction uses the saved classifier "
                            "and its original imputer and scaler. "
                            "It is a dataset-record demonstration, not "
                            "a prediction from live OpenFlow counters."
                        )

                    except Exception as exc:
                        st.error(f"Prediction failed: {exc}")
        else:
            st.error("The trained model file was not found.")


elif page == "Model Evaluation":
    st.header("Model Evaluation")
    st.write(
        "Metrics from the held-out test split created during model training."
    )

    evaluation_path = OUTPUT_DIR / "evaluation_results.joblib"

    if not evaluation_path.exists():
        st.warning(
            "Evaluation results have not been generated yet. "
            "Save the training-script changes, then run src/train.py "
            "to train the model and generate the evaluation results."
        )
    else:
        import joblib

        results = joblib.load(evaluation_path)
        report = results["classification_report"]
        matrix = results["confusion_matrix"]

        st.subheader("Overall Accuracy")
        st.metric("Accuracy", f"{results['accuracy']:.2%}")

        st.subheader("Classification Report")
        report_df = pd.DataFrame(report).T

        # Display the two class rows and overall summary metrics.
        display_rows = [
            row for row in ["Benign", "Bot", "macro avg", "weighted avg"]
            if row in report_df.index
        ]
        st.dataframe(
            report_df.loc[display_rows].round(4),
            use_container_width=True,
        )

        st.subheader("Confusion Matrix")
        matrix_df = pd.DataFrame(
            matrix,
            index=["Actual Benign", "Actual Bot"],
            columns=["Predicted Benign", "Predicted Bot"],
        )
        st.dataframe(matrix_df, use_container_width=True)

        st.caption(
            "Rows represent actual classes; columns represent predicted "
            "classes. These results are from a held-out split of the "
            "sampled CIC-IDS2018 data, not an independent live-network test."
        )


elif page == "SHAP Explanation":
    st.header("SHAP Feature Explanation")

    import json

    metadata_path = OUTPUT_DIR / "explanation_metadata.json"

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        st.info(
            f"Latest explained prediction: "
            f"{metadata['prediction']} | "
            f"SHAP: {metadata['shap_seconds']:.3f}s | "
            f"LIME: {metadata['lime_seconds']:.3f}s"
        )
    

    shap_data = load_csv("shap_explanation.csv")

    if shap_data is not None and not shap_data.empty:
        if {"Feature", "SHAP value"}.issubset(shap_data.columns):
            plot_data = shap_data.copy()
            plot_data["SHAP value"] = pd.to_numeric(
                plot_data["SHAP value"], errors="coerce"
            )
            plot_data = plot_data.dropna(subset=["SHAP value"])
            plot_data["Absolute contribution"] = (
                plot_data["SHAP value"].abs()
            )
            plot_data = plot_data.nlargest(
                10, "Absolute contribution"
            )

            st.subheader("Top 10 influential features")
            st.bar_chart(
                plot_data.sort_values("Absolute contribution")
                .set_index("Feature")["SHAP value"]
            )
            st.dataframe(plot_data, use_container_width=True)
        else:
            st.warning("Expected Feature and SHAP value columns were not found.")
            st.dataframe(shap_data, use_container_width=True)
    else:
        st.warning("SHAP explanation CSV was not found or is empty.")

    st.caption(
        "SHAP values explain this model's behavior for the analyzed "
        "example; they do not establish causality."
    )

elif page == "LIME Explanation":
    st.header("LIME Local Explanation")

    import json

    metadata_path = OUTPUT_DIR / "explanation_metadata.json"

    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)

        st.info(
            f"Latest explained prediction: "
            f"{metadata['prediction']} | "
            f"SHAP: {metadata['shap_seconds']:.3f}s | "
            f"LIME: {metadata['lime_seconds']:.3f}s"
        )

    lime_data = load_csv("lime_explanation.csv")

    if lime_data is not None:
        st.dataframe(lime_data, use_container_width=True)
    else:
        st.warning("LIME explanation CSV was not found.")

    lime_html = OUTPUT_DIR / "lime_explanation.html"
    if lime_html.exists():
        st.success(
            "The interactive LIME explanation is available in the "
            "saved HTML file."
        )
        st.code(str(lime_html), language="text")
        st.caption(
            "Open this HTML file in your browser to inspect the original "
            "LIME visualization."
        )


elif page == "Live SDN Monitoring":
    st.header("Live SDN Flow Statistics")

    flow_data = load_csv("flow_stats.csv")

    if flow_data is None:
        st.warning(
            "No flow_stats.csv found yet. Run the collector in Ubuntu "
            "while Mininet's switch is active."
        )
    elif flow_data.empty:
        st.info("The flow statistics file is empty.")
    else:
        latest = flow_data.iloc[-1]

        c1, c2, c3 = st.columns(3)
        c1.metric("Flow entries recorded", len(flow_data))
        c2.metric("Packets (latest row)", int(latest["packets"]))
        c3.metric("Bytes (latest row)", int(latest["bytes"]))

        st.subheader("Collected flow records")
        st.dataframe(flow_data, use_container_width=True)

        if {"timestamp", "packets"}.issubset(flow_data.columns):
            chart_data = flow_data[["timestamp", "packets"]].copy()
            chart_data["timestamp"] = pd.to_datetime(
                chart_data["timestamp"], errors="coerce"
            )
            chart_data = chart_data.dropna().set_index("timestamp")
            st.subheader("Recorded packet counters")
            st.line_chart(chart_data["packets"])

        st.caption(
            "These are switch flow-rule counters, not intrusion predictions."
        )

        
    st.header("Extracted SDN Traffic Features")

    sdn_data = load_csv("sdn_features.csv")

    if sdn_data is None:
        st.info("No sdn_features.csv found yet. Run extract_sdn_features.py in Ubuntu.")
    elif sdn_data.empty:
        st.info("The SDN features file is empty.")
    else:
        # Convert counters and rates to numeric values for charts.
        for column in [
            "packets", "bytes", "duration_seconds",
            "bytes_per_second", "packets_per_second",
            "icmp_type", "arp_operation"
        ]:
            if column in sdn_data.columns:
                sdn_data[column] = pd.to_numeric(
                    sdn_data[column], errors="coerce"
                )

        c1, c2, c3 = st.columns(3)
        c1.metric("Feature records", len(sdn_data))
        c2.metric(
            "Protocols observed",
            sdn_data["protocol"].nunique() if "protocol" in sdn_data.columns else 0
        )
        c3.metric(
            "Total recorded packets",
            int(sdn_data["packets"].fillna(0).sum())
            if "packets" in sdn_data.columns else 0
        )

        if "protocol" in sdn_data.columns:
            st.subheader("Protocol Distribution")
            st.bar_chart(sdn_data["protocol"].value_counts())


        st.subheader("Traffic Summary by Protocol")

        if {"protocol", "packets", "bytes"}.issubset(sdn_data.columns):
            protocol_summary = (
                sdn_data.groupby("protocol")[["packets", "bytes"]]
                .sum()
                .sort_values("packets", ascending=False)
            )

            st.dataframe(protocol_summary, use_container_width=True)

            st.bar_chart(protocol_summary["packets"])

        st.subheader("Extracted Traffic Records")
        st.dataframe(sdn_data, use_container_width=True)

        st.caption(
            "These are extracted Open vSwitch flow features. "
            "They are not yet inputs to the trained CIC-IDS2018 classifier."
        )


        
        # Live detector results
        st.subheader("Live SDN Detection Results")

        detection_path = OUTPUT_DIR / "live_detection.csv"

        if detection_path.exists():
            detection_df = pd.read_csv(detection_path)

            if detection_df.empty:
                st.info("No detection records are available yet.")
            elif "detection_status" not in detection_df.columns:
                st.warning(
                    "This CSV does not contain detection_status. "
                    "Run live_detector.py again."
                )
            else:
                # Summary metrics
                total = len(detection_df)
                review_count = detection_df["detection_status"].eq(
                    "REVIEW"
                ).sum()
                anomaly_count = detection_df["detection_status"].eq(
                    "ANOMALY_REVIEW"
                ).sum()
                no_activity = detection_df["detection_status"].eq(
                    "NO_ACTIVITY"
                ).sum()
                no_rule = detection_df["detection_status"].eq(
                    "NO_RULE_TRIGGERED"
                ).sum()

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric("Total records", total)
                c2.metric("Review", int(review_count))
                c3.metric("Anomaly review", int(anomaly_count))
                c4.metric("No activity", int(no_activity))
                c5.metric("No rule triggered", int(no_rule))

                # Status distribution
                st.markdown("#### Status Distribution")
                status_counts = (
                    detection_df["detection_status"]
                    .value_counts()
                    .rename_axis("Status")
                    .reset_index(name="Records")
                )
                st.bar_chart(
                    status_counts.set_index("Status")["Records"]
                )

                # Display readable results
                st.markdown("#### Individual Flow Results")

                preferred_columns = [
                    "timestamp",
                    "protocol",
                    "packets",
                    "bytes",
                    "packets_per_second",
                    "bytes_per_second",
                    "detection_status",
                    "explanation",
                ]

                display_columns = [
                    col for col in preferred_columns
                    if col in detection_df.columns
                ]

                st.dataframe(
                    detection_df[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

                st.caption(
                    "Statuses are generated by configured rules. "
                    "They are not confirmed attack labels or predictions "
                    "from the offline Benign/Bot model."
                )
        else:
            st.warning(
                "live_detection.csv not found. "
                "Run live_detector.py and copy its output into this project's outputs folder."
            )

        
