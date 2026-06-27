import streamlit as st
import sys
import pandas as pd
import plotly.express as px
import subprocess
from pathlib import Path

st.set_page_config(page_title="Synthetic Generator", layout="wide")
st.title("Synthetic data generator")
st.caption("Configure and generate synthetic patient records using CTGAN")

st.divider()
st.subheader("Generation configuration")

col1, col2, col3 = st.columns(3)
with col1:
    disease_filter = st.selectbox("Disease filter (display only)", [
        "All diseases",
        "fukuyama_muscular_dystrophy",
        "hallervorden_spatz_disease",
        "moyamoya_disease",
        "pachygyria_cerebellar_hypoplasia",
        "walker_warburg_syndrome"
    ])
with col2:
    model_sel = st.selectbox("Generative model", ["CTGAN (SDV)"])
with col3:
    n_display = st.number_input("Records to display", 10, 1000, 100, 10)

col_a, col_b = st.columns(2)
with col_a:
    epochs = st.slider("Training epochs", 50, 500, 300, 50)
with col_b:
    n_synth = st.slider("Records to generate", 100, 2000, 1000, 100)

st.divider()

if st.button("Generate synthetic data", type="primary", use_container_width=True):
    with st.spinner(f"Training CTGAN for {epochs} epochs and generating {n_synth} records..."):
        result = subprocess.run(
            [sys.executable, "src/models/generate.py"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            st.success("Generation complete!")
        else:
            st.error("Generation failed — check terminal for details")
            st.code(result.stderr)

st.divider()
synth_path = Path("data/synthetic/synthetic_metadata.csv")

if synth_path.exists():
    df_s = pd.read_csv(synth_path)
    if disease_filter != "All diseases":
        df_s = df_s[df_s["disease"] == disease_filter]

    c1,c2,c3 = st.columns(3)
    c1.metric("Total synthetic records", "1,000")
    c2.metric("Unique diseases",         df_s["disease"].nunique())
    c3.metric("Null values",             int(df_s.isnull().sum().sum()))

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Synthetic disease distribution")
        fig = px.bar(
            df_s["disease"].value_counts().reset_index(),
            x="disease", y="count", color="disease",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Real vs synthetic — disease")
        df_real = pd.read_csv("data/processed/metadata_clean.csv")
        real_pct  = df_real["disease"].value_counts(normalize=True).reset_index()
        synth_pct = df_s["disease"].value_counts(normalize=True).reset_index()
        real_pct["source"]  = "Real"
        synth_pct["source"] = "Synthetic"
        combined = pd.concat([real_pct, synth_pct])
        combined.columns = ["disease","proportion","source"]
        fig2 = px.bar(combined, x="disease", y="proportion",
                      color="source", barmode="group",
                      color_discrete_sequence=["#4C78A8","#F58518"])
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Synthetic records preview")
    st.dataframe(df_s.head(n_display), use_container_width=True)
else:
    st.info("No synthetic data found yet. Click Generate above to create records.")