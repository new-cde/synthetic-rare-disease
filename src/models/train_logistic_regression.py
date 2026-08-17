import pandas as pd
import pickle
import yaml
import numpy as np
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_and_encode(processed_path: str):
    df = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    feature_cols = ["inheritance", "category", "affected_systems", "prevalence"]
    target_col   = "disease"

    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])

    rng = np.random.default_rng(42)
    for col in feature_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col]   = le.transform(val_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))

    X_train = train_df[feature_cols].copy().astype(float)
    X_train += rng.normal(0, 1.2, X_train.shape)
    X_val  = val_df[feature_cols].copy().astype(float)
    X_val  += rng.normal(0, 0.8, X_val.shape)
    X_test = test_df[feature_cols].copy().astype(float)
    X_test += rng.normal(0, 0.8, X_test.shape)

    le_target   = LabelEncoder()
    y_train_raw = le_target.fit_transform(train_df[target_col])
    n_noisy     = int(0.08 * len(y_train_raw))
    noisy_idx   = rng.choice(len(y_train_raw), size=n_noisy, replace=False)
    y_train     = y_train_raw.copy()
    y_train[noisy_idx] = rng.integers(0, 5, size=n_noisy)

    y_val  = le_target.transform(val_df[target_col])
    y_test = le_target.transform(test_df[target_col])

    return X_train, y_train, X_val, y_val, X_test, y_test, le_target


def run():
    config = load_config()
    seed   = config["project"]["seed"]
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, le = load_and_encode(
        config["data"]["processed_path"]
    )

    model = LogisticRegression(
        max_iter=1000, C=0.1,
        solver="saga", random_state=seed
    )

    mlflow.set_experiment("synthetic-rare-disease")
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model",    "Logistic Regression")
        mlflow.log_param("C",        0.1)
        mlflow.log_param("solver",   "saga")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("seed",     seed)

        model.fit(X_train, y_train)
        logger.success("Logistic Regression trained")

        results = {}
        for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
            y_pred = model.predict(X)
            acc = accuracy_score(y, y_pred)
            f1  = f1_score(y, y_pred, average="weighted")
            logger.info(f"Logistic Regression | {split_name} | Acc: {acc:.4f} | F1: {f1:.4f}")
            print(classification_report(y, y_pred, target_names=le.classes_))

            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(confusion_matrix(y, y_pred),
                                   display_labels=le.classes_).plot(ax=ax, xticks_rotation=30, colorbar=False)
            ax.set_title(f"Logistic Regression — {split_name}")
            plt.tight_layout()
            fig.savefig(out_dir / f"cm_logistic_regression_{split_name}.png", dpi=150)
            plt.close(fig)

            results[f"{split_name}_accuracy"] = round(acc, 4)
            results[f"{split_name}_f1"]       = round(f1, 4)

        mlflow.log_metrics(results)

        model_path = Path("models/logistic_regression.pkl")
        model_path.parent.mkdir(exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump({"model": model, "label_encoder": le}, f)
        mlflow.log_artifact(str(model_path))

        pd.DataFrame([{"model": "Logistic Regression", **results}]).to_csv(
            "reports/metrics_logistic_regression.csv", index=False
        )
        logger.success(f"Logistic Regression — Val Acc: {results['val_accuracy']} | Test Acc: {results['test_accuracy']}")


if __name__ == "__main__":
    run()