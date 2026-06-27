import streamlit as st
import sys
import subprocess
from pathlib import Path
import pandas as pd

st.set_page_config(page_title="Admin Panel", layout="wide")
st.title("Admin panel")
st.caption("Run history, MLflow integration, and pipeline controls")
st.divider()

st.subheader("Pipeline controls")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Re-run EDA", use_container_width=True):
        with st.spinner("Running EDA..."):
            r = subprocess.run(
                [sys.executable, "src/eda.py"],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                st.success("EDA complete")
            else:
                st.error(r.stderr[:500])

with col2:
    if st.button("Re-run validation", use_container_width=True):
        with st.spinner("Running validation..."):
            r = subprocess.run(
                [sys.executable, "src/validation/evaluate.py"],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                st.success("Validation complete")
            else:
                st.error(r.stderr[:500])

with col3:
    st.link_button(
        "Open MLflow UI",
        "http://localhost:5000",
        use_container_width=True
    )

st.divider()
st.subheader("MLflow — experiment runs")
st.markdown("""
All pipeline runs are tracked in MLflow under the
**`synthetic-rare-disease`** experiment. Runs include:

| Run name | What's tracked |
|---|---|
| `random_forest`, `svm`, `gradient_boosting`, `logistic_regression` | Model accuracy, F1, confusion matrices |
| `ctgan_generation` | Epochs, n_synth, generated records |
| `tstr_validation` | JS/KL divergence, TSTR accuracy, privacy score |
""")
st.info("Start MLflow with: `mlflow ui` then visit http://localhost:5000")

st.divider()
st.subheader("Project version and config")
config_path = Path("config/config.yaml")
if config_path.exists():
    st.code(config_path.read_text(), language="yaml")

st.divider()
st.subheader("Output file status")
files = [
    ("Cleaned metadata",    "data/processed/metadata_clean.csv"),
    ("Synthetic data",      "data/synthetic/synthetic_metadata.csv"),
    ("CTGAN model",         "models/ctgan_model.pkl"),
    ("SVM model",           "models/svm.pkl"),
    ("Privacy score",       "reports/privacy_score.csv"),
    ("Validation report",   "reports/validation_report.txt"),
    ("Model comparison",    "reports/model_comparison.csv"),
]
for name, path in files:
    p = Path(path)
    exists = p.exists()
    size   = f"{p.stat().st_size/1024:.1f} KB" if exists else "—"
    col_n, col_s, col_sz = st.columns([3,1,1])
    col_n.text(name)
    col_s.markdown(
        "<span style='color:green'>Present</span>" if exists
        else "<span style='color:red'>Missing</span>",
        unsafe_allow_html=True
    )
    col_sz.text(size)