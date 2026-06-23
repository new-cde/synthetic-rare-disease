import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Synthetic Generator", layout="wide")
st.title("Synthetic Data Generator")
st.caption("Generate and explore synthetic patient records")

synth_path = Path("data/synthetic/synthetic_metadata.csv")

st.subheader("Generation configuration")
col1, col2, col3 = st.columns(3)
with col1:
    disease = st.selectbox("Disease filter", [
        "All diseases",
        "fukuyama_muscular_dystrophy",
        "hallervorden_spatz_disease",
        "moyamoya_disease",
        "pachygyria_cerebellar_hypoplasia",
        "walker_warburg_syndrome"
    ])
with col2:
    model_used = st.selectbox("Model used", ["CTGAN"], disabled=True)
with col3:
    n_records = st.number_input(
        "Records to display", min_value=10,
        max_value=1000, value=100, step=10
    )

st.divider()

if synth_path.exists():
    df_synth = pd.read_csv(synth_path)

    if disease != "All diseases":
        df_synth = df_synth[df_synth["disease"] == disease]

    st.success(
        f"Showing {min(n_records, len(df_synth))} of "
        f"{len(df_synth)} synthetic records"
        + (f" for {disease}" if disease != "All diseases" else "")
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Total synthetic records", len(df_synth))
    col2.metric("Unique diseases",         df_synth["disease"].nunique())
    col3.metric("Null values",             int(df_synth.isnull().sum().sum()))

    st.divider()
    st.subheader("Synthetic disease distribution")
    fig = px.bar(
        df_synth["disease"].value_counts().reset_index(),
        x="disease", y="count",
        color="disease",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Synthetic records preview")
    st.dataframe(
        df_synth.head(n_records),
        use_container_width=True
    )
else:
    st.warning(
        "Synthetic data not found. "
        "Run `python src/models/generate.py` first."
    )