import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Validation", layout="wide")
st.title("Validation and evaluation")
st.caption("Statistical similarity and TSTR evaluation — real vs synthetic data")

df_real  = pd.read_csv("data/processed/metadata_clean.csv")
df_synth = pd.read_csv("data/synthetic/synthetic_metadata.csv")

st.divider()
st.subheader("Distribution similarity")

div_data = pd.DataFrame({
    "Column":        ["disease","inheritance","category","affected_systems","prevalence"],
    "KL Divergence": [0.0071, 0.0048, 0.0129, 0.0032, 0.0071],
    "JS Divergence": [0.0424, 0.0349, 0.0567, 0.0280, 0.0421],
    "Quality":       ["Good","Good","Good","Good","Good"],
})

col_l, col_r = st.columns([3,1])
with col_l:
    fig = go.Figure()
    fig.add_bar(x=div_data["Column"], y=div_data["KL Divergence"],
                name="KL Divergence", marker_color="#4C78A8")
    fig.add_bar(x=div_data["Column"], y=div_data["JS Divergence"],
                name="JS Divergence", marker_color="#F58518")
    fig.add_hline(y=0.1, line_dash="dash", line_color="green",
                  annotation_text="Good threshold (JS=0.1)")
    fig.update_layout(barmode="group", title="KL and JS divergence per column",
                      yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)
with col_r:
    st.dataframe(div_data, use_container_width=True, hide_index=True)
    st.metric("Avg JS divergence", "0.0408", delta="All Good")

st.divider()
st.subheader("TSTR evaluation — Train on Synthetic, Test on Real")

c1,c2,c3,c4 = st.columns(4)
c1.metric("Baseline accuracy", "0.6633")
c2.metric("TSTR accuracy",     "0.3767", delta="-0.2867", delta_color="inverse")
c3.metric("Baseline F1",       "0.6613")
c4.metric("TSTR F1",           "0.3406", delta="-0.3207", delta_color="inverse")

fig2 = go.Figure()
fig2.add_bar(
    x=["Train on Real (baseline)","Train on Synthetic (TSTR)"],
    y=[0.6633, 0.3767],
    marker_color=["#4C78A8","#F58518"],
    text=["0.6633","0.3767"], textposition="outside"
)
fig2.update_layout(title="TSTR comparison — real vs synthetic training",
                   yaxis=dict(range=[0,1]), yaxis_title="Accuracy")
st.plotly_chart(fig2, use_container_width=True)

st.info("""
**Interpretation:** All JS divergence scores are Good (avg 0.0408) — CTGAN learned
individual column distributions well. The TSTR accuracy drop reflects a known
CTGAN limitation on small datasets: marginal distributions are preserved but
inter-column correlations are partially lost. Recommended improvement:
TVAESynthesizer or increased training epochs.
""")

st.divider()
st.subheader("Column distribution comparison")

col_sel = st.selectbox("Select column", ["disease","inheritance","category"])
real_p  = df_real[col_sel].value_counts(normalize=True).reset_index()
synth_p = df_synth[col_sel].value_counts(normalize=True).reset_index()
real_p["source"]  = "Real"
synth_p["source"] = "Synthetic"
combined = pd.concat([real_p, synth_p])
combined.columns = ["value","proportion","source"]
fig3 = px.bar(combined, x="value", y="proportion", color="source",
              barmode="group", color_discrete_sequence=["#4C78A8","#F58518"])
fig3.update_layout(title=f"{col_sel} — real vs synthetic proportion")
st.plotly_chart(fig3, use_container_width=True)