import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="Validation", layout="wide")
st.title("Validation and Evaluation")
st.caption("Statistical similarity and TSTR evaluation — real vs synthetic")

real_path  = Path("data/processed/metadata_clean.csv")
synth_path = Path("data/synthetic/synthetic_metadata.csv")

if not real_path.exists() or not synth_path.exists():
    st.error("Data files not found.")
    st.stop()

df_real  = pd.read_csv(real_path)
df_synth = pd.read_csv(synth_path)

st.divider()
st.subheader("Distribution similarity — KL and JS divergence")

divergence_data = {
    "Column":        ["disease","inheritance","category","affected_systems","prevalence"],
    "KL Divergence": [0.0071, 0.0048, 0.0129, 0.0032, 0.0071],
    "JS Divergence": [0.0424, 0.0349, 0.0567, 0.0280, 0.0421],
    "Quality":       ["Good","Good","Good","Good","Good"],
}
df_div = pd.DataFrame(divergence_data)

col1, col2 = st.columns([2, 1])
with col1:
    fig = go.Figure()
    fig.add_bar(
        x=df_div["Column"], y=df_div["KL Divergence"],
        name="KL Divergence", marker_color="#4C78A8"
    )
    fig.add_bar(
        x=df_div["Column"], y=df_div["JS Divergence"],
        name="JS Divergence", marker_color="#F58518"
    )
    fig.add_hline(
        y=0.1, line_dash="dash",
        line_color="green", annotation_text="Good threshold (0.1)"
    )
    fig.update_layout(
        barmode="group",
        title="KL and JS divergence per column",
        yaxis_title="Divergence score"
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.dataframe(df_div, use_container_width=True, hide_index=True)
    st.metric("Average JS divergence", "0.0408", delta="All Good")

st.divider()
st.subheader("TSTR evaluation — Train on Synthetic, Test on Real")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline accuracy",  "0.6633")
col2.metric("TSTR accuracy",      "0.3767", delta="-0.2867", delta_color="inverse")
col3.metric("Baseline F1",        "0.6613")
col4.metric("TSTR F1",            "0.3406", delta="-0.3207", delta_color="inverse")

fig2 = go.Figure()
fig2.add_bar(
    x=["Train on Real (Baseline)", "Train on Synthetic (TSTR)"],
    y=[0.6633, 0.3767],
    marker_color=["#4C78A8","#F58518"],
    text=["0.6633","0.3767"],
    textposition="outside"
)
fig2.update_layout(
    title="TSTR comparison",
    yaxis=dict(range=[0, 1]),
    yaxis_title="Accuracy"
)
st.plotly_chart(fig2, use_container_width=True)

st.info("""
**Interpretation:** JS divergence scores are all rated Good (avg 0.0408) —
CTGAN learned individual column distributions well. The TSTR accuracy drop
reflects a known CTGAN limitation on small datasets: marginal distributions
are preserved but inter-column correlations are partially lost.
Recommended improvement: TVAESynthesizer or increased training epochs.
""")

st.divider()
st.subheader("Real vs synthetic distribution comparison")
cat_cols = ["disease", "inheritance", "category"]
selected = st.selectbox("Select column", cat_cols)

real_pct  = df_real[selected].value_counts(normalize=True).reset_index()
synth_pct = df_synth[selected].value_counts(normalize=True).reset_index()
real_pct["source"]  = "Real"
synth_pct["source"] = "Synthetic"
combined = pd.concat([real_pct, synth_pct])
combined.columns = ["value","proportion","source"]

fig3 = px.bar(
    combined, x="value", y="proportion",
    color="source", barmode="group",
    color_discrete_sequence=["#4C78A8","#F58518"]
)
fig3.update_layout(title=f"{selected} — real vs synthetic")
st.plotly_chart(fig3, use_container_width=True)