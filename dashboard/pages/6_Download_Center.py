import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Download Center", layout="wide")
st.title("Download center")
st.caption("Export synthetic data, reports, validation results, and figures")
st.divider()

def dl_btn(label, path, filename, mime):
    p = Path(path)
    if p.exists():
        st.download_button(label, p.read_bytes(), filename, mime,
                           use_container_width=True)
    else:
        st.button(label + " — not found", disabled=True,
                  use_container_width=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Data exports")
    dl_btn("Download synthetic CSV",   "data/synthetic/synthetic_metadata.csv","synthetic_metadata.csv","text/csv")
    dl_btn("Download cleaned real CSV","data/processed/metadata_clean.csv",    "metadata_clean.csv",    "text/csv")
    dl_btn("Download model comparison","reports/model_comparison.csv",          "model_comparison.csv",  "text/csv")
    dl_btn("Download privacy report",  "reports/privacy_score.csv",            "privacy_score.csv",     "text/csv")

with col2:
    st.subheader("Reports")
    dl_btn("Download validation report","reports/validation_report.txt","validation_report.txt","text/plain")

with col3:
    st.subheader("Figures")
    figs = [
        ("Model comparison chart",  "reports/figures/model_comparison.png",     "model_comparison.png"),
        ("Divergence scores",       "reports/figures/divergence_scores.png",     "divergence_scores.png"),
        ("TSTR comparison",         "reports/figures/tstr_comparison.png",       "tstr_comparison.png"),
        ("Correlation heatmap",     "reports/figures/correlation_heatmap.png",   "correlation_heatmap.png"),
        ("Class distribution",      "reports/figures/class_distribution.png",    "class_distribution.png"),
        ("K-Means clusters PCA",    "reports/figures/kmeans_clusters_pca.png",   "kmeans_clusters_pca.png"),
        ("Pairplot",                "reports/figures/pairplot.png",              "pairplot.png"),
    ]
    for label, path, fname in figs:
        dl_btn(f"Download {label}", path, fname, "image/png")