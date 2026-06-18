import pandas as pd
import mlflow
import pickle
import yaml
from pathlib import Path
from loguru import logger
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_train_data(processed_path: str) -> pd.DataFrame:
    path     = Path(processed_path) / "metadata_clean.csv"
    df       = pd.read_csv(path)
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    logger.info(f"Training data shape: {train_df.shape}")
    return train_df


def build_metadata(df: pd.DataFrame) -> SingleTableMetadata:
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df)
    logger.info("Metadata schema detected from dataframe")
    return metadata


def train_ctgan(df: pd.DataFrame,
                metadata: SingleTableMetadata,
                epochs: int) -> CTGANSynthesizer:
    logger.info(f"Training CTGAN — epochs={epochs}")
    model = CTGANSynthesizer(
        metadata,
        epochs=epochs,
        verbose=True,
        cuda=False
    )
    model.fit(df)
    logger.success("CTGAN training complete")
    return model


def save_model(model: CTGANSynthesizer, model_dir: str) -> str:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(model_dir) / "ctgan_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {out_path}")
    return out_path


def generate_and_save(model: CTGANSynthesizer,
                      n: int,
                      synthetic_path: str) -> pd.DataFrame:
    Path(synthetic_path).mkdir(parents=True, exist_ok=True)
    logger.info(f"Generating {n} synthetic records")
    synthetic_df = model.sample(num_rows=n)
    out_path     = Path(synthetic_path) / "synthetic_metadata.csv"
    synthetic_df.to_csv(out_path, index=False)
    logger.success(f"Synthetic data saved to {out_path}")
    return synthetic_df


def run_generation():
    config         = load_config()
    epochs         = config["model"]["epochs"]
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    n_synth        = 1000

    mlflow.set_experiment("synthetic-rare-disease")

    with mlflow.start_run(run_name="ctgan_generation"):
        mlflow.log_params({
            "model":   "ctgan",
            "epochs":  epochs,
            "seed":    seed,
            "n_synth": n_synth
        })

        df_train     = load_train_data(processed_path)
        metadata     = build_metadata(df_train)
        model        = train_ctgan(df_train, metadata, epochs)
        model_path   = save_model(model, "models/")
        synthetic_df = generate_and_save(model, n_synth, synthetic_path)

        mlflow.log_metric("synthetic_records", len(synthetic_df))
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(
            str(Path(synthetic_path) / "synthetic_metadata.csv")
        )

        logger.info("\nPreview of synthetic records:")
        print(synthetic_df.head(10).to_string())

        logger.info("\nSynthetic disease distribution:")
        print(synthetic_df["disease"].value_counts().to_string())

        logger.info("\nNull check:")
        print(synthetic_df.isnull().sum().to_string())

    logger.success("Generation complete — check http://localhost:5000")


if __name__ == "__main__":
    run_generation()