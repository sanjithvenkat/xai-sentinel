
from pathlib import Path

import pandas as pd
import streamlit as st

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = BASE_DIR / "models" / "ids_model.joblib"

st.set_page_config(
    page_title="Explainable SDN Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Explainable AI for SDN Intrusion Detection")
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

    if demo is None:
        st.warning("outputs/demo_flows.csv was not found.")
    else:
        st.write("Sample records used for the model demonstration.")
        st.dataframe(demo.head(100), use_container_width=True)

        label_col = next(
            (col for col in demo.columns if col.lower() == "label"),
            None,
        )
        if label_col:
            st.subheader("Label distribution")
            st.bar_chart(demo[label_col].value_counts())
        else:
            st.info(
                "No label column was detected in the demo file. "
                "The records are shown without inventing predictions."
            )



elif page == "SHAP Explanation":
    st.header("SHAP Feature Explanation")

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
