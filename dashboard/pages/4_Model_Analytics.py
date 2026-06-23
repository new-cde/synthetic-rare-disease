import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Model Analytics", layout="wide")
st.title("Model Analytics")
st.caption("Classifier comparison and SVM performance — base model selected by reviewer")

st.divider()
st.subheader("Model comparison — all four classifiers")

comparison = {
    "Model":         ["SVM","Random Forest","Gradient Boosting","Logistic Regression"],
    "Val Accuracy":  [0.8167, 0.8067, 0.8233, 0.8233],
    "Val F1":        [0.8107, 0.8011, 0.8194, 0.8167],
    "Test Accuracy": [0.8933, 0.8900, 0.8867, 0.8767],
    "Test F1":       [0.8921, 0.8881, 0.8840, 0.8744],
}
df_cmp = pd.DataFrame(comparison)
st.dataframe(
    df_cmp.style.highlight_max(
        subset=["Test Accuracy","Test F1"],
        color="#d4edda"
    ),
    use_container_width=True,
    hide_index=True
)

fig = go.Figure()
fig.add_bar(
    x=df_cmp["Model"], y=df_cmp["Val Accuracy"],
    name="Val Accuracy", marker_color="#4C78A8"
)
fig.add_bar(
    x=df_cmp["Model"], y=df_cmp["Test Accuracy"],
    name="Test Accuracy", marker_color="#F58518"
)
fig.update_layout(
    barmode="group",
    title="Val vs test accuracy per model",
    yaxis=dict(range=[0.8, 0.95]),
    yaxis_title="Accuracy"
)
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("SVM — selected base model")
col1, col2, col3 = st.columns(3)
col1.metric("Test accuracy", "89.33%")
col2.metric("Test F1",       "0.8921")
col3.metric("Val accuracy",  "81.67%")

st.divider()
st.subheader("Confusion matrices")
cm_dir = Path("reports/figures")
cm_col1, cm_col2 = st.columns(2)

svm_val  = cm_dir / "cm_svm_val.png"
svm_test = cm_dir / "cm_svm_test.png"

if svm_val.exists():
    cm_col1.image(str(svm_val),  caption="SVM — Validation set", use_container_width=True)
if svm_test.exists():
    cm_col2.image(str(svm_test), caption="SVM — Test set",       use_container_width=True)

st.divider()
st.subheader("All model confusion matrices")
models = ["random_forest","logistic_regression","gradient_boosting"]
for m in models:
    with st.expander(m.replace("_"," ").title()):
        c1, c2 = st.columns(2)
        p_val  = cm_dir / f"cm_{m}_val.png"
        p_test = cm_dir / f"cm_{m}_test.png"
        if p_val.exists():
            c1.image(str(p_val),  caption="Validation", use_container_width=True)
        if p_test.exists():
            c2.image(str(p_test), caption="Test",       use_container_width=True)