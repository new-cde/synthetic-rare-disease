import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Download Center", layout="wide")
st.title("Download Center")
st.caption("Export synthetic data, reports and validation results")

st.divider()

def download_button(label: str, path: str, filename: str, mime: str):
    p = Path(path)
    if p.exists():
        st.download_button(
            label=label,
            data=p.read_bytes(),
            file_name=filename,
            mime=mime,
            use_container_width=True
        )
    else:
        st.button(label + " (not found)", disabled=True, use_container_width=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Data exports")
    download_button(
        "Synthetic data CSV",
        "data/synthetic/synthetic_metadata.csv",
        "synthetic_metadata.csv",
        "text/csv"
    )
    download_button(
        "Cleaned real data CSV",
        "data/processed/metadata_clean.csv",
        "metadata_clean.csv",
        "text/csv"
    )
    download_button(
        "Model comparison CSV",
        "reports/model_comparison.csv",
        "model_comparison.csv",
        "text/csv"
    )

with col2:
    st.subheader("Validation reports")
    download_button(
        "Validation report TXT",
        "reports/validation_report.txt",
        "validation_report.txt",
        "text/plain"
    )

with col3:
    st.subheader("Figures")
    figures = [
        ("Model comparison chart", "reports/figures/model_comparison.png",      "model_comparison.png"),
        ("Divergence scores",      "reports/figures/divergence_scores.png",      "divergence_scores.png"),
        ("TSTR comparison",        "reports/figures/tstr_comparison.png",        "tstr_comparison.png"),
        ("Correlation heatmap",    "reports/figures/correlation_heatmap.png",    "correlation_heatmap.png"),
        ("Class distribution",     "reports/figures/class_distribution.png",     "class_distribution.png"),
    ]
    for label, path, filename in figures:
        download_button(label, path, filename, "image/png")

st.divider()
st.caption(
    "All exports are generated from the live pipeline. "
    "Re-run `python src/models/generate.py` to refresh synthetic data."
)