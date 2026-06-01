import pandas as pd
import yaml
from pathlib import Path
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_metadata(raw_path: str) -> pd.DataFrame:
    metadata_path = Path(raw_path) / "rare_neuro_mri_curated" / "metadata.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found at {metadata_path}")
    logger.info(f"Loading metadata from {metadata_path}")
    return pd.read_csv(metadata_path)


def summarise_dataset(df: pd.DataFrame) -> None:
    logger.info("=== Dataset Summary ===")
    logger.info(f"Total records : {len(df)}")
    logger.info(f"Columns       : {list(df.columns)}")
    logger.info(f"Splits        : {df['split'].value_counts().to_dict()}")
    logger.info("Records per disease:")
    summary = df.groupby(["disease", "split"]).size().unstack(fill_value=0)
    print("\n" + summary.to_string())


def run_ingestion():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    logger.info("Starting data ingestion")
    df = load_metadata(raw_path)
    summarise_dataset(df)
    logger.success("Ingestion complete — data is ready for preprocessing")
    return df


if __name__ == "__main__":
    run_ingestion()