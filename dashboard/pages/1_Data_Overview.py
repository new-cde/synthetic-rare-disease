import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data Overview")
st.caption("Cleaned metadata — 2,000 records across 5 rare neurological diseases")

df = pd.read_csv("data/processed/metadata_clean.csv")

st.divider()
col1, col2, col3 = st.columns(3)
col1.metric("Total records",  len(df))
col2.metric("Diseases",       df["disease"].nunique())
col3.metric("Features",       len(df.columns))

st.divider()
st.subheader("Disease distribution")
fig = px.bar(
    df["disease"].value_counts().reset_index(),
    x="disease", y="count",
    color="disease",
    labels={"disease": "Disease", "count": "Records"},
    color_discrete_sequence=px.colors.qualitative.Set2
)
fig.update_layout(showlegend=False)
st.plotly_chart(fig, use_container_width=True)

col_l, col_r = st.columns(2)

with col_l:
    st.subheader("Inheritance patterns")
    fig2 = px.pie(
        df, names="inheritance",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig2, use_container_width=True)

with col_r:
    st.subheader("Disease categories")
    fig3 = px.pie(
        df, names="category",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.subheader("Split distribution")
split_counts = df.groupby(["disease","split"]).size().reset_index(name="count")
fig4 = px.bar(
    split_counts, x="disease", y="count",
    color="split", barmode="group",
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Raw data preview")
st.dataframe(df.head(50), use_container_width=True)