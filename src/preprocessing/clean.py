import pandas as pd
import yaml
from pathlib import Path
from loguru import logger


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def drop_pii_columns(df: pd.DataFrame) -> pd.DataFrame:
    pii_cols = ["filename", "original_path", "relative_path"]
    existing = [c for c in pii_cols if c in df.columns]
    logger.info(f"Dropping PII/path columns: {existing}")
    return df.drop(columns=existing)


def clean_categorical(df: pd.DataFrame) -> pd.DataFrame:
    cat_cols = ["disease", "split", "inheritance", "category"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].str.strip().str.lower()
            logger.info(f"Normalised column '{col}' — {df[col].nunique()} unique values")
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    before = df.isnull().sum().sum()
    df = df.fillna("unknown")
    after = df.isnull().sum().sum()
    logger.info(f"Missing values: {before} before → {after} after")
    return df


def report(original: pd.DataFrame, cleaned: pd.DataFrame) -> None:
    logger.info("=== Cleaning Report ===")
    logger.info(f"Shape before : {original.shape}")
    logger.info(f"Shape after  : {cleaned.shape}")
    logger.info(f"Columns kept : {list(cleaned.columns)}")


def run_cleaning():
    config = load_config()
    raw_path = config["data"]["raw_path"]
    processed_path = config["data"]["processed_path"]

    Path(processed_path).mkdir(parents=True, exist_ok=True)

    metadata_path = Path(raw_path) / "rare_neuro_mri_curated" / "metadata.csv"
    logger.info(f"Loading raw metadata from {metadata_path}")
    df_raw = pd.read_csv(metadata_path)

    df = drop_pii_columns(df_raw.copy())
    df = clean_categorical(df)
    df = handle_missing(df)

    report(df_raw, df)

    out_path = Path(processed_path) / "metadata_clean.csv"
    df.to_csv(out_path, index=False)
    logger.success(f"Cleaned data saved to {out_path}")
    return df


if __name__ == "__main__":
    run_cleaning()