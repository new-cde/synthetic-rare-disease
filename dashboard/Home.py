import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="SynthMed — Rare Disease Synthetic Data",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 SynthMed")
st.caption("Synthetic Data Generation for Rare Disease Research")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Diseases covered",   "5")
col2.metric("Real records",       "2,000")
col3.metric("Synthetic records",  "1,000")
col4.metric("Avg JS divergence",  "0.0408", delta="Good", delta_color="normal")

st.divider()

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Project overview")
    st.markdown("""
    This platform generates **privacy-preserving synthetic patient records**
    for five rare neurological diseases using CTGAN — a tabular generative
    adversarial network.

    **Pipeline summary:**
    - 2,000 real MRI metadata records ingested and anonymised
    - CTGAN trained on 1,400 training records
    - 1,000 synthetic records generated
    - Validated using KL/JS divergence and TSTR evaluation
    """)

with col_r:
    st.subheader("Model performance")
    results = {
        "Model":         ["SVM ✅", "Random Forest", "Gradient Boosting", "Logistic Regression"],
        "Val Accuracy":  [0.8167, 0.8067, 0.8233, 0.8233],
        "Test Accuracy": [0.8933, 0.8900, 0.8867, 0.8767],
        "Test F1":       [0.8921, 0.8881, 0.8840, 0.8744],
    }
    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("SVM selected as base model by reviewer — highest test accuracy")

st.divider()
st.subheader("Privacy and quality summary")

c1, c2, c3, c4, c5 = st.columns(5)
cols = [c1, c2, c3, c4, c5]
metrics = [
    ("disease",          0.0071, 0.0424),
    ("inheritance",      0.0048, 0.0349),
    ("category",         0.0129, 0.0567),
    ("affected_systems", 0.0032, 0.0280),
    ("prevalence",       0.0071, 0.0421),
]
for col, (name, kl, js) in zip(cols, metrics):
    col.metric(name, f"JS {js}", delta="Good", delta_color="normal")

st.divider()
st.caption(
    "Built by Abinaya M · SRM Institute of Science and Technology · "
    "Synthetic Data Generation for Rare Disease Research Internship"
)