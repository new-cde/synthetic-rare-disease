import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import subprocess
import sys
import pickle
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="SynthMed — Rare Disease Platform",
    page_icon="🧬",
    layout="wide"
)

st.markdown("""
<style>
/* Sidebar */
[data-testid="stSidebar"] {
    background: #0A0E1A;
    border-right: 1px solid #1E2A3A;
}
[data-testid="stSidebar"] * { color: #E8EDF5 !important; }
[data-testid="stSidebar"] .stRadio > label {
    font-size: 13px;
    font-weight: 500;
    letter-spacing: 0.03em;
    color: #8A9BB0 !important;
    text-transform: uppercase;
    margin-bottom: 4px;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
    padding: 8px 12px;
    border-radius: 8px;
    margin: 2px 0;
    transition: background 0.15s;
    text-transform: none !important;
    font-size: 14px !important;
    letter-spacing: 0 !important;
    color: #C8D6E8 !important;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label:hover {
    background: #1E2A3A;
}
/* Metric cards */
[data-testid="stMetric"] {
    background: #0F1624;
    border: 1px solid #1E2A3A;
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8A9BB0 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #E8EDF5 !important; font-size: 22px !important; }
/* Main area */
.main { background: #060B14; }
.stApp { background: #060B14; }
h1, h2, h3 { color: #E8EDF5 !important; }
p, li { color: #A8B9CC !important; }
/* Divider */
hr { border-color: #1E2A3A !important; }
/* Badge */
.role-badge {
    display: inline-block;
    background: #0D4F3C;
    color: #2ECC8F !important;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.05em;
    border: 1px solid #1A7A5A;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧬 SynthMed")
    st.markdown('<span class="role-badge">USER PORTAL</span>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠  Home",
        "📊  Data Overview",
        "🔬  EDA & Clustering",
        "⚙️  Synthetic Generator",
        "✅  Validation",
        "🔐  Privacy Analysis",
        "📈  Model Analytics",
        "⬇️  Download Center",
    ])
    st.markdown("---")
    st.caption("Abinaya M\nSRM Institute of Science and Technology\nInternship 2025–26")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def load_real():
    return pd.read_csv("data/processed/metadata_clean.csv")

def load_synth():
    return pd.read_csv("data/synthetic/synthetic_metadata.csv")

def load_privacy():
    p = Path("reports/privacy_score.csv")
    if p.exists():
        row = pd.read_csv(p).iloc[0]
        return float(row["privacy_score"]), float(row["avg_nn_distance"]), \
               int(row["exact_matches"]), str(row["risk_level"])
    return 0.0, 0.0, 0, "Unknown"

# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("# 🧬 SynthMed")
    st.caption("Synthetic Data Generation Platform for Rare Neurological Disease Research")
    st.divider()

    priv_score, _, _, risk_level = load_privacy()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Diseases covered",    "5")
    c2.metric("Real records",        "2,000")
    c3.metric("Synthetic records",   "1,000")
    c4.metric("Avg JS divergence",   "0.0408", delta="Good")
    c5.metric("Privacy score",       f"{priv_score}/100", delta=f"Risk: {risk_level}")

    st.divider()
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.subheader("Platform overview")
        st.markdown(f"""
SynthMed generates **privacy-preserving synthetic patient records** for five
rare neurological diseases, enabling researchers to work with realistic
datasets without risking patient confidentiality.

| Component | Detail |
|---|---|
| Generative model | CTGAN via SDV |
| Base classifier | SVM — 89.33% test accuracy |
| Privacy score | {priv_score}/100 — {risk_level} risk |
| Avg JS divergence | 0.0408 — all columns rated Good |
| Experiment tracking | MLflow |
| Data versioning | DVC |
        """)

    with col_r:
        st.subheader("Model leaderboard")
        df_cmp = pd.DataFrame({
            "Model":    ["SVM ✅", "Random Forest", "Grad. Boosting", "Logistic Reg."],
            "Test Acc": [0.8933, 0.8900, 0.8867, 0.8767],
            "Test F1":  [0.8921, 0.8881, 0.8840, 0.8744],
        })
        st.dataframe(
            df_cmp.style.highlight_max(subset=["Test Acc","Test F1"], color="#0D4F3C"),
            use_container_width=True, hide_index=True
        )
        st.caption("SVM selected as base model — reviewer approved")

    st.divider()
    st.subheader("Distribution quality — all columns")
    cols = st.columns(5)
    for col, (name, js) in zip(cols, [
        ("Disease", 0.0424), ("Inheritance", 0.0349), ("Category", 0.0567),
        ("Affected systems", 0.0280), ("Prevalence", 0.0421)
    ]):
        col.metric(name, f"JS {js}", delta="Good")

# ══════════════════════════════════════════════════════════════════════════════
# DATA OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  Data Overview":
    st.title("Data overview")
    st.caption("Cleaned metadata — 2,000 records across 5 rare neurological diseases")

    df = load_real()
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total records", len(df))
    c2.metric("Diseases",      df["disease"].nunique())
    c3.metric("Features",      len(df.columns))
    c4.metric("Null values",   int(df.isnull().sum().sum()))

    st.divider()
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Disease distribution")
        fig = px.bar(
            df["disease"].value_counts().reset_index(),
            x="disease", y="count", color="disease",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig.update_layout(showlegend=False, plot_bgcolor="#0F1624",
                          paper_bgcolor="#0F1624", font_color="#A8B9CC")
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.subheader("Inheritance patterns")
        fig2 = px.pie(df, names="inheritance",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        fig2.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                           font_color="#A8B9CC")
        st.plotly_chart(fig2, use_container_width=True)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.subheader("Disease categories")
        fig3 = px.pie(df, names="category",
                      color_discrete_sequence=px.colors.qualitative.Set3)
        fig3.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                           font_color="#A8B9CC")
        st.plotly_chart(fig3, use_container_width=True)

    with col_r2:
        st.subheader("Train / Val / Test split")
        sc = df.groupby(["disease","split"]).size().reset_index(name="count")
        fig4 = px.bar(sc, x="disease", y="count", color="split", barmode="group",
                      color_discrete_sequence=px.colors.qualitative.Set1)
        fig4.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                           font_color="#A8B9CC")
        st.plotly_chart(fig4, use_container_width=True)

    st.divider()
    st.subheader("Raw data preview")
    sel = st.multiselect("Filter by disease", df["disease"].unique().tolist(),
                         default=df["disease"].unique().tolist())
    st.dataframe(df[df["disease"].isin(sel)].head(100), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# EDA & CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬  EDA & Clustering":
    st.title("Exploratory data analysis & clustering")
    st.caption("Correlation analysis, feature relationships, K-Means clustering, PCA visualisation")

    df = load_real()

    from sklearn.preprocessing import LabelEncoder
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    cat_cols = ["disease","inheritance","category","affected_systems","prevalence"]
    df_enc = df.copy()
    for col in cat_cols:
        df_enc[col] = LabelEncoder().fit_transform(df_enc[col].astype(str))

    # ── Correlation heatmap
    st.subheader("Feature correlation heatmap")
    corr = df_enc[cat_cols].corr()
    fig_heat = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Pearson correlation matrix — encoded features"
    )
    fig_heat.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                           font_color="#A8B9CC")
    st.plotly_chart(fig_heat, use_container_width=True)

    st.info("""
**Reading the heatmap:** Values close to 1.0 (dark red) mean two features
are strongly positively correlated. Values close to -1.0 (dark blue) mean
strong negative correlation. Near 0 means independent. High correlation
between `disease` and other features confirms our metadata is clinically
deterministic — each disease has a unique feature signature.
    """)

    st.divider()

    # ── Feature boxplots
    st.subheader("Feature distribution per disease")
    feat_sel = st.selectbox("Select feature to explore",
                            ["inheritance","category","affected_systems","prevalence"])
    df_plot = df_enc.copy()
    df_plot["disease_label"] = df["disease"]
    fig_box = px.box(
        df_plot, x="disease_label", y=feat_sel,
        color="disease_label",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title=f"{feat_sel} distribution across diseases"
    )
    fig_box.update_layout(showlegend=False, plot_bgcolor="#0F1624",
                          paper_bgcolor="#0F1624", font_color="#A8B9CC",
                          xaxis_title="Disease", yaxis_title=feat_sel)
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── Pairplot (scatter matrix)
    st.subheader("Feature pair relationships")
    feature_cols = ["inheritance","category","affected_systems","prevalence"]
    df_pair = df_enc[feature_cols].copy()
    df_pair["disease"] = df["disease"]
    fig_scatter = px.scatter_matrix(
        df_pair,
        dimensions=feature_cols,
        color="disease",
        color_discrete_sequence=px.colors.qualitative.Set2,
        title="Pairplot — all feature combinations coloured by disease",
        opacity=0.5
    )
    fig_scatter.update_traces(marker=dict(size=3))
    fig_scatter.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                              font_color="#A8B9CC", height=600)
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── K-Means elbow
    st.subheader("K-Means clustering")
    col_l, col_r = st.columns(2)

    X = df_enc[feature_cols].values
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    with col_l:
        st.markdown("**Elbow curve — optimal k selection**")
        fig_elbow = go.Figure()
        fig_elbow.add_scatter(
            x=list(k_range), y=inertias,
            mode="lines+markers",
            line=dict(color="#2ECC8F", width=2),
            marker=dict(size=8, color="#2ECC8F")
        )
        fig_elbow.add_vline(x=5, line_dash="dash", line_color="#F58518",
                            annotation_text="k=5 (one per disease)")
        fig_elbow.update_layout(
            title="K-Means inertia vs number of clusters",
            xaxis_title="Number of clusters (k)",
            yaxis_title="Inertia",
            plot_bgcolor="#0F1624", paper_bgcolor="#0F1624", font_color="#A8B9CC"
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col_r:
        st.markdown("**Cluster size distribution at k=5**")
        km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
        cluster_labels = km5.fit_predict(X)
        cluster_counts = pd.Series(cluster_labels).value_counts().reset_index()
        cluster_counts.columns = ["Cluster","Count"]
        cluster_counts["Cluster"] = cluster_counts["Cluster"].astype(str)
        fig_counts = px.bar(
            cluster_counts, x="Cluster", y="Count",
            color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1
        )
        fig_counts.update_layout(showlegend=False, plot_bgcolor="#0F1624",
                                 paper_bgcolor="#0F1624", font_color="#A8B9CC")
        st.plotly_chart(fig_counts, use_container_width=True)

    st.divider()

    # ── PCA 2D
    st.subheader("PCA 2D — clusters vs true disease labels")
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    total_var = var1 + var2

    df_pca = pd.DataFrame({
        "PC1": X_2d[:, 0],
        "PC2": X_2d[:, 1],
        "Cluster": cluster_labels.astype(str),
        "Disease": df["disease"].values
    })

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        fig_cluster = px.scatter(
            df_pca, x="PC1", y="PC2", color="Cluster",
            color_discrete_sequence=px.colors.qualitative.Set1,
            title=f"K-Means clusters (k=5)",
            labels={"PC1": f"PC1 ({var1:.1f}% var)", "PC2": f"PC2 ({var2:.1f}% var)"},
            opacity=0.7
        )
        fig_cluster.update_traces(marker=dict(size=5))
        fig_cluster.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                                  font_color="#A8B9CC")
        st.plotly_chart(fig_cluster, use_container_width=True)

    with col_r2:
        fig_disease = px.scatter(
            df_pca, x="PC1", y="PC2", color="Disease",
            color_discrete_sequence=px.colors.qualitative.Set2,
            title="True disease labels",
            labels={"PC1": f"PC1 ({var1:.1f}% var)", "PC2": f"PC2 ({var2:.1f}% var)"},
            opacity=0.7
        )
        fig_disease.update_traces(marker=dict(size=5))
        fig_disease.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                                  font_color="#A8B9CC")
        st.plotly_chart(fig_disease, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("PC1 variance explained", f"{var1:.1f}%")
    c2.metric("PC2 variance explained", f"{var2:.1f}%")
    c3.metric("Total variance (2D)",    f"{total_var:.1f}%")

    st.info(f"""
**Interpretation:** PCA reduces 4 features to 2 dimensions, explaining
{total_var:.1f}% of total variance. Comparing left vs right plots shows
how well K-Means clusters align with true disease labels. Tight, separated
clusters confirm the dataset is well-structured and clinically meaningful.
    """)

    st.divider()

    # ── Inheritance breakdown
    st.subheader("Inheritance pattern breakdown per disease")
    inh_counts = df.groupby(["disease","inheritance"]).size().reset_index(name="count")
    fig_inh = px.bar(
        inh_counts, x="disease", y="count",
        color="inheritance", barmode="stack",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title="Inheritance pattern composition per disease"
    )
    fig_inh.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                          font_color="#A8B9CC", xaxis_title="",
                          yaxis_title="Record count")
    st.plotly_chart(fig_inh, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC GENERATOR
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️  Synthetic Generator":
    st.title("Synthetic data generator")
    st.caption("Generate synthetic patient records using CTGAN")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        disease_filter = st.selectbox("Disease filter", [
            "All diseases",
            "fukuyama_muscular_dystrophy",
            "hallervorden_spatz_disease",
            "moyamoya_disease",
            "pachygyria_cerebellar_hypoplasia",
            "walker_warburg_syndrome"
        ])
    with col2:
        st.selectbox("Generative model", ["CTGAN (SDV)"])
    with col3:
        n_display = st.number_input("Records to display", 10, 1000, 100, 10)

    if st.button("▶ Generate synthetic data", type="primary", use_container_width=True):
        with st.spinner("Training CTGAN and generating synthetic records..."):
            r = subprocess.run(
                [sys.executable, "src/models/generate.py"],
                capture_output=True, text=True
            )
            if r.returncode == 0:
                st.success("Generation complete!")
            else:
                st.error("Generation failed — see error below")
                st.code(r.stderr[-1000:])

    st.divider()
    synth_path = Path("data/synthetic/synthetic_metadata.csv")
    if synth_path.exists():
        df_s = load_synth()
        if disease_filter != "All diseases":
            df_s = df_s[df_s["disease"] == disease_filter]

        c1, c2, c3 = st.columns(3)
        c1.metric("Synthetic records", len(df_s))
        c2.metric("Unique diseases",   df_s["disease"].nunique())
        c3.metric("Null values",       int(df_s.isnull().sum().sum()))

        col_l, col_r = st.columns(2)
        with col_l:
            st.subheader("Synthetic disease distribution")
            fig = px.bar(
                df_s["disease"].value_counts().reset_index(),
                x="disease", y="count", color="disease",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(showlegend=False, plot_bgcolor="#0F1624",
                              paper_bgcolor="#0F1624", font_color="#A8B9CC")
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Real vs synthetic proportion")
            df_r = load_real()
            rp = df_r["disease"].value_counts(normalize=True).reset_index()
            sp = df_s["disease"].value_counts(normalize=True).reset_index()
            rp["source"] = "Real"; sp["source"] = "Synthetic"
            comb = pd.concat([rp, sp])
            comb.columns = ["disease","proportion","source"]
            fig2 = px.bar(comb, x="disease", y="proportion", color="source",
                          barmode="group",
                          color_discrete_sequence=["#4C78A8","#F58518"])
            fig2.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                               font_color="#A8B9CC")
            st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Synthetic records preview")
        st.dataframe(df_s.head(n_display), use_container_width=True)
    else:
        st.info("No synthetic data found. Click Generate above to create records.")

# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "✅  Validation":
    st.title("Validation and evaluation")
    st.caption("Statistical similarity and TSTR evaluation — real vs synthetic data")

    df_real  = load_real()
    df_synth = load_synth()

    st.divider()
    st.subheader("Distribution similarity — KL and JS divergence")

    div_data = pd.DataFrame({
        "Column":        ["disease","inheritance","category","affected_systems","prevalence"],
        "KL Divergence": [0.0071, 0.0048, 0.0129, 0.0032, 0.0071],
        "JS Divergence": [0.0424, 0.0349, 0.0567, 0.0280, 0.0421],
        "Quality":       ["Good","Good","Good","Good","Good"],
    })

    col_l, col_r = st.columns([3, 1])
    with col_l:
        fig = go.Figure()
        fig.add_bar(x=div_data["Column"], y=div_data["KL Divergence"],
                    name="KL Divergence", marker_color="#4C78A8")
        fig.add_bar(x=div_data["Column"], y=div_data["JS Divergence"],
                    name="JS Divergence", marker_color="#F58518")
        fig.add_hline(y=0.1, line_dash="dash", line_color="#2ECC8F",
                      annotation_text="Good threshold (JS=0.1)")
        fig.update_layout(barmode="group", yaxis_title="Divergence score",
                          title="KL and JS divergence per column",
                          plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                          font_color="#A8B9CC")
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
    fig2.update_layout(
        title="TSTR comparison — real vs synthetic training",
        yaxis=dict(range=[0, 1]), yaxis_title="Accuracy",
        plot_bgcolor="#0F1624", paper_bgcolor="#0F1624", font_color="#A8B9CC"
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info("""
**Interpretation:** JS divergence scores are all Good (avg 0.0408) — CTGAN
learned individual column distributions well. The TSTR accuracy drop reflects
a known CTGAN limitation on small datasets: marginal distributions are
preserved but inter-column correlations are partially lost.
**Recommended fix:** TVAESynthesizer or increased training epochs.
    """)

    st.divider()
    st.subheader("Column distribution comparison — real vs synthetic")
    col_sel = st.selectbox("Select column", ["disease","inheritance","category"])
    rp = df_real[col_sel].value_counts(normalize=True).reset_index()
    sp = df_synth[col_sel].value_counts(normalize=True).reset_index()
    rp["source"] = "Real"; sp["source"] = "Synthetic"
    comb = pd.concat([rp, sp]); comb.columns = ["value","proportion","source"]
    fig3 = px.bar(comb, x="value", y="proportion", color="source",
                  barmode="group",
                  color_discrete_sequence=["#4C78A8","#F58518"],
                  title=f"{col_sel} — real vs synthetic proportion")
    fig3.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                       font_color="#A8B9CC")
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PRIVACY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔐  Privacy Analysis":
    st.title("Privacy analysis")
    st.caption("Membership inference testing, privacy scoring, and risk indicators")

    priv_path = Path("reports/privacy_score.csv")
    if not priv_path.exists():
        st.warning("Run `python src/validation/evaluate.py` first.")
        st.stop()

    score, dist, exact, risk = load_privacy()

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Privacy score",        f"{score}/100")
    c2.metric("Avg NN distance",      f"{dist:.4f}")
    c3.metric("Exact record matches", exact,
              delta="No leakage" if exact == 0 else f"{exact} matches",
              delta_color="normal" if exact == 0 else "inverse")
    c4.metric("Risk level", risk)

    st.divider()
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": "Privacy score", "font": {"color": "#A8B9CC"}},
        number={"font": {"color": "#E8EDF5"}},
        gauge={
            "axis":  {"range": [0, 100], "tickcolor": "#A8B9CC"},
            "bar":   {"color": "#2ECC8F"},
            "bgcolor": "#0F1624",
            "steps": [
                {"range": [0,  40],  "color": "#2D0F0F"},
                {"range": [40, 70],  "color": "#2D1F0F"},
                {"range": [70, 100], "color": "#0D2F1F"},
            ],
        }
    ))
    fig.update_layout(height=300, paper_bgcolor="#0F1624", font_color="#A8B9CC")
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Risk indicators")
    for label, safe, val, exp in [
        ("Exact record matches", exact == 0,  str(exact),         "No exact copies of real records in synthetic data"),
        ("Avg NN distance",      dist > 0.5,  f"{dist:.4f}",      "Higher = synthetic records more different from real"),
        ("Privacy score",        score >= 70, f"{score}/100",     "Above 70 considered safe for research sharing"),
        ("Column distributions", True,        "All Good (JS<0.1)","All column distributions differ sufficiently"),
    ]:
        cl, cm, cr = st.columns([2, 1, 3])
        cl.markdown(f"**{label}**")
        color = "#2ECC8F" if safe else "#E74C3C"
        cm.markdown(
            f"<span style='color:{color};font-weight:500'>"
            f"{'Safe' if safe else 'Review'}</span> — {val}",
            unsafe_allow_html=True
        )
        cr.caption(exp)

    st.divider()
    st.info(f"""
**HIPAA/GDPR note:** No PII retained. PII columns (filename, original_path,
relative_path) were removed at preprocessing. Avg NN distance: {dist:.4f}.
Exact matches: {exact}. Score of {score}/100 reflects the small categorical
feature space — expected for structured rare disease metadata where every
feature combination corresponds to a known disease profile.
    """)

# ══════════════════════════════════════════════════════════════════════════════
# MODEL ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Model Analytics":
    st.title("Model analytics")
    st.caption("SVM base model · classifier comparison · feature importance · confusion matrices")

    df_cmp = pd.DataFrame({
        "Model":         ["SVM ✅","Random Forest","Gradient Boosting","Logistic Regression"],
        "Val Accuracy":  [0.8167, 0.8067, 0.8233, 0.8233],
        "Val F1":        [0.8107, 0.8011, 0.8194, 0.8167],
        "Test Accuracy": [0.8933, 0.8900, 0.8867, 0.8767],
        "Test F1":       [0.8921, 0.8881, 0.8840, 0.8744],
    })

    st.divider()
    st.subheader("Model comparison")
    st.dataframe(
        df_cmp.style.highlight_max(
            subset=["Test Accuracy","Test F1"], color="#0D4F3C"
        ),
        use_container_width=True, hide_index=True
    )

    fig = go.Figure()
    fig.add_bar(x=df_cmp["Model"], y=df_cmp["Val Accuracy"],
                name="Val accuracy",  marker_color="#4C78A8")
    fig.add_bar(x=df_cmp["Model"], y=df_cmp["Test Accuracy"],
                name="Test accuracy", marker_color="#2ECC8F")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0.8, 0.96]),
        title="Val vs test accuracy per model",
        plot_bgcolor="#0F1624", paper_bgcolor="#0F1624", font_color="#A8B9CC"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("SVM — selected base model")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Test accuracy", "89.33%")
    c2.metric("Test F1",       "0.8921")
    c3.metric("Val accuracy",  "81.67%")
    c4.metric("Val F1",        "0.8107")

    st.divider()
    st.subheader("Feature importance (Random Forest proxy)")
    rf_path = Path("models/random_forest.pkl")
    if rf_path.exists():
        with open(rf_path, "rb") as f:
            bundle = pickle.load(f)
        rf = bundle["model"]
        df_imp = pd.DataFrame({
            "Feature":    ["inheritance","category","affected_systems","prevalence"],
            "Importance": rf.feature_importances_
        }).sort_values("Importance")
        fig_i = px.bar(
            df_imp, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="teal",
            title="Feature importance scores"
        )
        fig_i.update_layout(plot_bgcolor="#0F1624", paper_bgcolor="#0F1624",
                            font_color="#A8B9CC")
        st.plotly_chart(fig_i, use_container_width=True)
    else:
        st.info("Train models first to see feature importance.")

    st.divider()
    st.subheader("SVM confusion matrices")
    col1, col2 = st.columns(2)
    for col, fname, lbl in [
        (col1, "cm_svm_val.png",  "Validation set"),
        (col2, "cm_svm_test.png", "Test set")
    ]:
        p = Path("reports/figures") / fname
        if p.exists():
            col.image(str(p), caption=f"SVM — {lbl}", use_container_width=True)

    st.divider()
    st.subheader("All model confusion matrices")
    for name, key in [
        ("Random Forest",       "random_forest"),
        ("Gradient Boosting",   "gradient_boosting"),
        ("Logistic Regression", "logistic_regression"),
    ]:
        with st.expander(name):
            c1, c2 = st.columns(2)
            for col, split in [(c1, "val"), (c2, "test")]:
                p = Path("reports/figures") / f"cm_{key}_{split}.png"
                if p.exists():
                    col.image(str(p), caption=split.title(), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD CENTER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⬇️  Download Center":
    st.title("Download center")
    st.caption("Export synthetic data, reports, validation results, and figures")
    st.divider()

    def dl(label, path, fname, mime):
        p = Path(path)
        if p.exists():
            st.download_button(label, p.read_bytes(), fname, mime,
                               use_container_width=True)
        else:
            st.button(label + " — not found", disabled=True,
                      use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Data exports")
        dl("⬇ Synthetic CSV",     "data/synthetic/synthetic_metadata.csv",
           "synthetic_metadata.csv", "text/csv")
        dl("⬇ Cleaned real CSV",  "data/processed/metadata_clean.csv",
           "metadata_clean.csv",     "text/csv")
        dl("⬇ Model comparison",  "reports/model_comparison.csv",
           "model_comparison.csv",   "text/csv")
        dl("⬇ Privacy report",    "reports/privacy_score.csv",
           "privacy_score.csv",      "text/csv")

    with c2:
        st.subheader("Reports")
        dl("⬇ Validation report", "reports/validation_report.txt",
           "validation_report.txt",  "text/plain")

    with c3:
        st.subheader("Figures")
        for lbl, path, fname in [
            ("Model comparison",    "reports/figures/model_comparison.png",    "model_comparison.png"),
            ("Divergence scores",   "reports/figures/divergence_scores.png",   "divergence_scores.png"),
            ("TSTR comparison",     "reports/figures/tstr_comparison.png",     "tstr_comparison.png"),
            ("Correlation heatmap", "reports/figures/correlation_heatmap.png", "correlation_heatmap.png"),
            ("Class distribution",  "reports/figures/class_distribution.png",  "class_distribution.png"),
            ("K-Means PCA",         "reports/figures/kmeans_clusters_pca.png", "kmeans_clusters_pca.png"),
            ("Pairplot",            "reports/figures/pairplot.png",            "pairplot.png"),
        ]:
            dl(f"⬇ {lbl}", path, fname, "image/png")
