import pandas as pd
import mlflow
import pickle
import yaml
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_splits(processed_path: str):
    path = Path(processed_path) / "metadata_clean.csv"
    df   = pd.read_csv(path)
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    logger.info(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    return train_df, val_df, test_df


def encode_features(train_df, val_df, test_df):
    feature_cols = ["inheritance", "category", "affected_systems", "prevalence"]
    target_col   = "disease"
    encoders     = {}

    for col in feature_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        val_df[col]   = le.transform(val_df[col].astype(str))
        test_df[col]  = le.transform(test_df[col].astype(str))
        encoders[col] = le
        logger.info(f"Encoded '{col}' — {len(le.classes_)} classes")

    le_target = LabelEncoder()
    y_train   = le_target.fit_transform(train_df[target_col])
    y_val     = le_target.transform(val_df[target_col])
    y_test    = le_target.transform(test_df[target_col])

    X_train = train_df[feature_cols]
    X_val   = val_df[feature_cols]
    X_test  = test_df[feature_cols]

    return X_train, y_train, X_val, y_val, X_test, y_test, le_target


def get_models(seed: int) -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=seed, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        "SVM": SVC(
            kernel="rbf", probability=True,
            random_state=seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=5,
            random_state=seed
        ),
    }


def evaluate_model(model, X_train, y_train,
                   X_val, y_val,
                   X_test, y_test,
                   le_target, model_name: str,
                   out_dir: Path) -> dict:

    model.fit(X_train, y_train)
    logger.info(f"Trained: {model_name}")

    results = {}
    for split_name, X, y in [("val", X_val, y_val), ("test", X_test, y_test)]:
        y_pred = model.predict(X)
        acc    = accuracy_score(y, y_pred)
        f1     = f1_score(y, y_pred, average="weighted")
        report = classification_report(
            y, y_pred,
            target_names=le_target.classes_
        )

        logger.info(f"{model_name} | {split_name} | Acc: {acc:.4f} | F1: {f1:.4f}")
        print(f"\nClassification Report — {model_name} ({split_name}):\n{report}")

        fig, ax = plt.subplots(figsize=(8, 6))
        cm   = confusion_matrix(y, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=le_target.classes_)
        disp.plot(ax=ax, xticks_rotation=30, colorbar=False)
        ax.set_title(f"{model_name} — {split_name}")
        plt.tight_layout()
        safe_name = model_name.lower().replace(" ", "_")
        fig_path  = out_dir / f"cm_{safe_name}_{split_name}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

        results[f"{split_name}_accuracy"] = round(acc, 4)
        results[f"{split_name}_f1"]       = round(f1, 4)

    return results


def save_comparison_table(results: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    for model_name, metrics in results.items():
        rows.append({
            "Model":         model_name,
            "Val Accuracy":  metrics["val_accuracy"],
            "Val F1":        metrics["val_f1"],
            "Test Accuracy": metrics["test_accuracy"],
            "Test F1":       metrics["test_f1"],
        })

    df = pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)

    print("\n" + "="*60)
    print("MODEL COMPARISON TABLE")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

    fig, ax = plt.subplots(figsize=(10, 5))
    x      = range(len(df))
    width  = 0.35
    bars1  = ax.bar([i - width/2 for i in x], df["Val Accuracy"],  width, label="Val Accuracy",  color="steelblue")
    bars2  = ax.bar([i + width/2 for i in x], df["Test Accuracy"], width, label="Test Accuracy", color="coral")

    for bar in bars1 + bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(list(x))
    ax.set_xticklabels(df["Model"], rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Model comparison — Val vs Test accuracy")
    ax.legend()
    plt.tight_layout()
    chart_path = out_dir / "model_comparison.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info(f"Comparison chart saved to {chart_path}")

    csv_path = Path("reports") / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")

    return df


def run_comparison():
    config         = load_config()
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(processed_path)
    X_train, y_train, X_val, y_val, X_test, y_test, le_target = encode_features(
        train_df, val_df, test_df
    )

    models  = get_models(seed)
    all_results = {}

    mlflow.set_experiment("synthetic-rare-disease")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name.lower().replace(" ", "_")):
            mlflow.log_param("model", model_name)
            mlflow.log_param("seed",  seed)

            results = evaluate_model(
                model, X_train, y_train,
                X_val, y_val, X_test, y_test,
                le_target, model_name, out_dir
            )

            mlflow.log_metrics(results)

            model_dir  = Path("models")
            model_dir.mkdir(exist_ok=True)
            safe_name  = model_name.lower().replace(" ", "_")
            model_path = model_dir / f"{safe_name}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"model": model, "label_encoder": le_target}, f)

            mlflow.log_artifact(str(model_path))
            all_results[model_name] = results

    comparison_df = save_comparison_table(all_results, out_dir)

    best = comparison_df.iloc[0]["Model"]
    logger.success(f"Best model by test accuracy: {best}")
    logger.info("Share reports/model_comparison.csv with your reviewer")
    logger.info("Check all runs at http://localhost:5000")


if __name__ == "__main__":
    run_comparison()