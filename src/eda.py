import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import yaml
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_clean_data(processed_path: str) -> pd.DataFrame:
    path = Path(processed_path) / "metadata_clean.csv"
    logger.info(f"Loading clean data from {path}")
    return pd.read_csv(path)


def plot_class_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    counts = df.groupby(["disease", "split"]).size().unstack()
    counts.plot(kind="bar", ax=ax, colormap="Set2")
    ax.set_title("Record count per disease and split")
    ax.set_xlabel("Disease")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    out = out_dir / "class_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_inheritance_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["inheritance"].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Inheritance pattern distribution")
    ax.set_xlabel("Inheritance type")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    out = out_dir / "inheritance_distribution.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


def plot_disease_category(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    df["category"].value_counts().plot(
        kind="pie", ax=ax, autopct="%1.1f%%", startangle=140
    )
    ax.set_title("Disease category breakdown")
    ax.set_ylabel("")
    plt.tight_layout()
    out = out_dir / "disease_category.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {out}")


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


def run_eda():
    config = load_config()
    processed_path = config["data"]["processed_path"]

    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_clean_data(processed_path)
    print_summary(df)
    plot_class_distribution(df, out_dir)
    plot_inheritance_distribution(df, out_dir)
    plot_disease_category(df, out_dir)

    logger.success(f"EDA complete — 3 charts saved to {out_dir}")


if __name__ == "__main__":
    run_eda()