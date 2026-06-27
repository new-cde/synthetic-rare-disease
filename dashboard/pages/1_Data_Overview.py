import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Data Overview", layout="wide")
st.title("Data overview")
st.caption("Cleaned metadata — 2,000 records across 5 rare neurological diseases")

df = pd.read_csv("data/processed/metadata_clean.csv")

st.divider()
c1,c2,c3,c4 = st.columns(4)
c1.metric("Total records",  len(df))
c2.metric("Diseases",       df["disease"].nunique())
c3.metric("Features",       len(df.columns))
c4.metric("Null values",    int(df.isnull().sum().sum()))

st.divider()

col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Disease distribution")
    fig = px.bar(
        df["disease"].value_counts().reset_index(),
        x="disease", y="count", color="disease",
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.subheader("Inheritance patterns")
    fig2 = px.pie(df, names="inheritance",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig2, use_container_width=True)

col_l2, col_r2 = st.columns(2)
with col_l2:
    st.subheader("Disease categories")
    fig3 = px.pie(df, names="category",
                  color_discrete_sequence=px.colors.qualitative.Set3)
    st.plotly_chart(fig3, use_container_width=True)

with col_r2:
    st.subheader("Train / val / test split")
    split_counts = df.groupby(["disease","split"]).size().reset_index(name="count")
    fig4 = px.bar(split_counts, x="disease", y="count",
                  color="split", barmode="group",
                  color_discrete_sequence=px.colors.qualitative.Set1)
    fig4.update_layout(xaxis_title="", yaxis_title="Records")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.subheader("Correlation heatmap")
cat_cols = ["disease","inheritance","category","affected_systems","prevalence"]
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))
corr = df_enc[cat_cols].corr()
fig5 = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                 zmin=-1, zmax=1, title="Feature correlation matrix")
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.subheader("Raw data preview")
disease_filter = st.multiselect(
    "Filter by disease", df["disease"].unique().tolist(),
    default=df["disease"].unique().tolist()
)
st.dataframe(
    df[df["disease"].isin(disease_filter)].head(100),
    use_container_width=True
)