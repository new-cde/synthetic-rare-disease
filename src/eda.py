import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_clean_data(processed_path: str) -> pd.DataFrame:
    path = Path(processed_path) / "metadata_clean.csv"
    logger.info(f"Loading clean data from {path}")
    return pd.read_csv(path)


def encode_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    cat_cols = ["disease", "split", "inheritance",
                "category", "affected_systems", "prevalence"]
    for col in cat_cols:
        if col in df_enc.columns:
            df_enc[col] = LabelEncoder().fit_transform(
                df_enc[col].astype(str)
            )
    return df_enc


# ── Basic distribution plots ──────────────────────────────────────────────────

def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    counts  = df.groupby(["disease", "split"]).size().unstack()
    counts.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Record count per disease and split")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    _save(fig, out_dir / "class_distribution.png")


def plot_inheritance_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["inheritance"].value_counts().plot(
        kind="bar", ax=ax, color="steelblue"
    )
    ax.set_title("Inheritance pattern distribution")
    ax.set_xlabel("Inheritance type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    _save(fig, out_dir / "inheritance_distribution.png")


def plot_disease_category(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["category"].value_counts().plot(
        kind="pie", ax=ax, autopct="%1.1f%%", startangle=140
    )
    ax.set_title("Disease category breakdown")
    ax.set_ylabel("")
    plt.tight_layout()
    _save(fig, out_dir / "disease_category.png")


# ── Correlation analysis ───────────────────────────────────────────────────────

def plot_correlation_heatmap(df_enc: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = ["disease", "inheritance", "category",
                    "affected_systems", "prevalence"]
    corr = df_enc[feature_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax
    )
    ax.set_title("Feature correlation heatmap")
    plt.tight_layout()
    _save(fig, out_dir / "correlation_heatmap.png")
    logger.info("Correlation matrix:\n" + corr.to_string())


def plot_pairplot(df: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    df_enc = df.copy()
    for col in feature_cols:
        df_enc[col] = LabelEncoder().fit_transform(
            df_enc[col].astype(str)
        )
    df_enc["disease_label"] = df["disease"]

    fig = sns.pairplot(
        df_enc[feature_cols + ["disease_label"]],
        hue="disease_label",
        palette="Set2",
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20}
    ).fig
    fig.suptitle("Pairplot of features coloured by disease", y=1.01)
    _save(fig, out_dir / "pairplot.png")


def plot_feature_boxplots(df: pd.DataFrame,
                          df_enc: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, col in enumerate(feature_cols):
        df_plot = df_enc.copy()
        df_plot["disease_label"] = df["disease"]
        df_plot.boxplot(
            column=col,
            by="disease_label",
            ax=axes[i],
            grid=False
        )
        axes[i].set_title(f"{col} by disease")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=30)
    plt.suptitle("Feature distributions per disease")
    plt.tight_layout()
    _save(fig, out_dir / "feature_boxplots.png")


# ── Cluster analysis ───────────────────────────────────────────────────────────

def plot_kmeans_clusters(df_enc: pd.DataFrame,
                         df: pd.DataFrame, out_dir: Path) -> None:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    X = df_enc[feature_cols].values

    # Elbow curve
    inertias = []
    k_range  = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(k_range), inertias, "o-", color="steelblue")
    ax.set_title("K-Means elbow curve")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    plt.tight_layout()
    _save(fig, out_dir / "kmeans_elbow.png")

    # Fit with k=5 (one per disease)
    km     = KMeans(n_clusters=5, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    # PCA to 2D for visualisation
    pca    = PCA(n_components=2, random_state=42)
    X_2d   = pca.fit_transform(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1 — clusters
    scatter = axes[0].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels, cmap="Set1", alpha=0.6, s=20
    )
    axes[0].set_title("K-Means clusters (PCA 2D)")
    axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    # Plot 2 — true disease labels
    disease_enc = LabelEncoder().fit_transform(df["disease"])
    scatter2    = axes[1].scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=disease_enc, cmap="Set2", alpha=0.6, s=20
    )
    axes[1].set_title("True disease labels (PCA 2D)")
    axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    plt.colorbar(scatter2, ax=axes[1], label="Disease")

    plt.suptitle("K-Means clusters vs true labels")
    plt.tight_layout()
    _save(fig, out_dir / "kmeans_clusters_pca.png")

    var_explained = sum(pca.explained_variance_ratio_) * 100
    logger.info(f"PCA 2 components explain {var_explained:.1f}% of variance")
    logger.info(f"K-Means cluster distribution: "
                f"{pd.Series(labels).value_counts().to_dict()}")


# ── Summary ────────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    logger.info("=== EDA Summary ===")
    logger.info(f"Total records     : {len(df)}")
    logger.info(f"Diseases          : {df['disease'].nunique()}")
    logger.info(f"Splits            : {df['split'].value_counts().to_dict()}")
    logger.info(f"Inheritance types : {df['inheritance'].nunique()}")
    logger.info(f"Categories        : {df['category'].nunique()}")
    print("\nDisease value counts:")
    print(df["disease"].value_counts().to_string())
    print("\nInheritance value counts:")
    print(df["inheritance"].value_counts().to_string())
    print("\nCategory value counts:")
    print(df["category"].value_counts().to_string())


# ── Helper ─────────────────────────────────────────────────────────────────────

def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def run_eda():
    config         = load_config()
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df     = load_clean_data(processed_path)
    df_enc = encode_for_analysis(df)

    print_summary(df)

    logger.info("Generating distribution plots...")
    plot_class_distribution(df, out_dir)
    plot_inheritance_distribution(df, out_dir)
    plot_disease_category(df, out_dir)

    logger.info("Generating correlation analysis...")
    plot_correlation_heatmap(df_enc, out_dir)
    plot_pairplot(df, out_dir)
    plot_feature_boxplots(df, df_enc, out_dir)

    logger.info("Generating cluster analysis...")
    plot_kmeans_clusters(df_enc, df, out_dir)

    logger.success(f"EDA complete — all charts saved to {out_dir}")


if __name__ == "__main__":
    run_eda()