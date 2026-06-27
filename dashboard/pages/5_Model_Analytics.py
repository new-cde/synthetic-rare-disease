import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(page_title="Model Analytics", layout="wide")
st.title("Model analytics")
st.caption("SVM — selected base model · classifier comparison · CTGAN training metrics")

st.divider()
st.subheader("Model comparison — all four classifiers")

df_cmp = pd.DataFrame({
    "Model":         ["SVM ✅","Random Forest","Gradient Boosting","Logistic Regression"],
    "Val Accuracy":  [0.8167, 0.8067, 0.8233, 0.8233],
    "Val F1":        [0.8107, 0.8011, 0.8194, 0.8167],
    "Test Accuracy": [0.8933, 0.8900, 0.8867, 0.8767],
    "Test F1":       [0.8921, 0.8881, 0.8840, 0.8744],
})
st.dataframe(
    df_cmp.style.highlight_max(
        subset=["Test Accuracy","Test F1"], color="#d4edda"
    ),
    use_container_width=True, hide_index=True
)

fig = go.Figure()
fig.add_bar(x=df_cmp["Model"], y=df_cmp["Val Accuracy"],
            name="Val accuracy", marker_color="#4C78A8")
fig.add_bar(x=df_cmp["Model"], y=df_cmp["Test Accuracy"],
            name="Test accuracy", marker_color="#F58518")
fig.update_layout(barmode="group", yaxis=dict(range=[0.8,0.96]),
                  title="Accuracy comparison — val vs test",
                  yaxis_title="Accuracy")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("SVM — base model performance")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Test accuracy", "89.33%")
c2.metric("Test F1",       "0.8921")
c3.metric("Val accuracy",  "81.67%")
c4.metric("Val F1",        "0.8107")

# Feature importance from Random Forest (proxy for SVM)
st.divider()
st.subheader("Feature importance (Random Forest)")
rf_path = Path("models/random_forest.pkl")
if rf_path.exists():
    with open(rf_path,"rb") as f:
        bundle = pickle.load(f)
    rf_model = bundle["model"]
    feat_names  = ["inheritance","category","affected_systems","prevalence"]
    importances = rf_model.feature_importances_
    df_imp = pd.DataFrame({
        "Feature":    feat_names,
        "Importance": importances
    }).sort_values("Importance", ascending=True)
    fig_imp = px.bar(df_imp, x="Importance", y="Feature",
                     orientation="h", color="Importance",
                     color_continuous_scale="Teal",
                     title="Feature importance scores")
    st.plotly_chart(fig_imp, use_container_width=True)
else:
    st.info("Train models first to see feature importance.")

st.divider()
st.subheader("CTGAN training loss")
loss_path = Path("reports/ctgan_loss.csv")
if loss_path.exists():
    df_loss = pd.read_csv(loss_path)
    fig_loss = px.line(df_loss, title="CTGAN generator and discriminator loss",
                       labels={"index":"Epoch","value":"Loss","variable":"Component"})
    st.plotly_chart(fig_loss, use_container_width=True)
else:
    st.info("Re-run `python src/models/generate.py` to capture CTGAN loss curves.")

st.divider()
st.subheader("Confusion matrices — SVM")
col1, col2 = st.columns(2)
for col, split, label in [
    (col1, "cm_svm_val.png",  "Validation set"),
    (col2, "cm_svm_test.png", "Test set")
]:
    p = Path("reports/figures") / split
    if p.exists():
        col.image(str(p), caption=f"SVM — {label}", use_container_width=True)

st.subheader("All models — confusion matrices")
for m_name, m_key in [
    ("Random Forest",        "random_forest"),
    ("Gradient Boosting",    "gradient_boosting"),
    ("Logistic Regression",  "logistic_regression"),
]:
    with st.expander(m_name):
        c1, c2 = st.columns(2)
        for col, split in [(c1,"val"),(c2,"test")]:
            p = Path("reports/figures") / f"cm_{m_key}_{split}.png"
            if p.exists():
                col.image(str(p), caption=split.title(),
                          use_container_width=True)