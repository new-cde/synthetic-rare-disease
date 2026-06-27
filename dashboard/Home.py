import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="SynthMed — Rare Disease Platform",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stSidebar"]{background:#0F1117}
[data-testid="stSidebar"] * {color:#FAFAFA}
.metric-card{background:var(--background-color);border:1px solid rgba(49,51,63,0.2);
    border-radius:12px;padding:1rem 1.25rem;text-align:center}
</style>
""", unsafe_allow_html=True)

st.title("🧬 SynthMed")
st.caption("Synthetic Data Generation Platform for Rare Neurological Disease Research")
st.divider()

# Load privacy score
priv_path = Path("reports/privacy_score.csv")
priv_score = 0.0
risk_level = "Unknown"
if priv_path.exists():
    p = pd.read_csv(priv_path).iloc[0]
    priv_score = p["privacy_score"]
    risk_level = p["risk_level"]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Diseases",          "5",      help="Rare neurological diseases")
col2.metric("Real records",      "2,000",  help="Original MRI metadata records")
col3.metric("Synthetic records", "1,000",  help="CTGAN-generated records")
col4.metric("Avg JS divergence", "0.0408", delta="Good", delta_color="normal")
col5.metric("Privacy score",     f"{priv_score}/100",
            delta=f"Risk: {risk_level}", delta_color="normal")

st.divider()

col_l, col_r = st.columns([3, 2])

with col_l:
    st.subheader("Platform overview")
    st.markdown(f"""
    **SynthMed** generates privacy-preserving synthetic patient records for
    five rare neurological diseases, enabling researchers to work with
    realistic datasets without risking patient confidentiality.

    | Component | Detail |
    |---|---|
    | Generative model | CTGAN via SDV |
    | Base classifier | SVM (89.33% test accuracy) |
    | Validation | KL/JS divergence + TSTR |
    | Privacy score | {priv_score}/100 — {risk_level} risk |
    | Avg JS divergence | 0.0408 — all columns rated Good |
    | Experiment tracking | MLflow |
    | Data versioning | DVC |
    """)

with col_r:
    st.subheader("Model comparison")
    df_cmp = pd.DataFrame({
        "Model":         ["SVM ✅", "Random Forest", "Grad. Boosting", "Logistic Reg."],
        "Test Acc":      [0.8933, 0.8900, 0.8867, 0.8767],
        "Test F1":       [0.8921, 0.8881, 0.8840, 0.8744],
    })
    st.dataframe(
        df_cmp.style.highlight_max(subset=["Test Acc","Test F1"], color="#d4edda"),
        use_container_width=True, hide_index=True
    )
    st.caption("SVM selected as base model — reviewer approved")

st.divider()
st.subheader("Distribution quality — all columns")

cols = st.columns(5)
metrics = [
    ("disease",          0.0071, 0.0424),
    ("inheritance",      0.0048, 0.0349),
    ("category",         0.0129, 0.0567),
    ("affected_systems", 0.0032, 0.0280),
    ("prevalence",       0.0071, 0.0421),
]
for col, (name, kl, js) in zip(cols, metrics):
    col.metric(name.replace("_"," ").title(), f"JS {js}", delta="Good")

st.divider()
st.caption("Abinaya M · SRM Institute of Science and Technology · Internship 2026")