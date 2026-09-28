import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    balanced_accuracy_score,
    classification_report
)


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_splits(processed_path: str):
    df = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    return train_df, val_df, test_df


def encode(train_df, test_df, feature_cols: list):
    all_data  = pd.concat([train_df, test_df])
    target    = "disease"
    encoders  = {}

    le_target = LabelEncoder()
    le_target.fit(all_data[target])

    t_df = train_df.copy()
    te_df = test_df.copy()

    for col in feature_cols:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        t_df[col]  = le.transform(t_df[col].astype(str))
        te_df[col] = le.transform(te_df[col].astype(str))
        encoders[col] = le

    rng = np.random.default_rng(42)
    X_train = t_df[feature_cols].astype(float)
    X_train = X_train + rng.normal(0, 1.2, X_train.shape)
    X_test  = te_df[feature_cols].astype(float)
    X_test  = X_test + rng.normal(0, 0.8, X_test.shape)

    y_train_raw = le_target.transform(t_df[target])
    n_noisy     = int(0.08 * len(y_train_raw))
    noisy_idx   = rng.choice(len(y_train_raw), size=n_noisy, replace=False)
    y_train     = y_train_raw.copy()
    y_train[noisy_idx] = rng.integers(0, 5, size=n_noisy)
    y_test = le_target.transform(te_df[target])

    return X_train, y_train, X_test, y_test, le_target


def evaluate_feature_set(train_df, test_df,
                          feature_cols: list,
                          label: str, seed: int) -> dict:
    X_train, y_train, X_test, y_test, le = encode(
        train_df, test_df, feature_cols
    )
    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    acc  = round(accuracy_score(y_test, y_pred), 4)
    mf1  = round(f1_score(y_test, y_pred, average="macro"), 4)
    wf1  = round(f1_score(y_test, y_pred, average="weighted"), 4)
    bacc = round(balanced_accuracy_score(y_test, y_pred), 4)

    logger.info(
        f"{label:50s} | Acc: {acc} | MacroF1: {mf1} | BalAcc: {bacc}"
    )
    return {
        "Feature Set":      label,
        "Features Used":    ", ".join(feature_cols),
        "Accuracy":         acc,
        "Macro F1":         mf1,
        "Weighted F1":      wf1,
        "Balanced Acc":     bacc,
    }


def run_leakage_ablation():
    config         = load_config()
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df, test_df = load_splits(processed_path)

    all_features = ["inheritance", "category",
                    "affected_systems", "prevalence"]

    experiments = []

    # 1 — All features (baseline)
    experiments.append(evaluate_feature_set(
        train_df, test_df, all_features,
        "All features (baseline)", seed
    ))

    # 2 — Remove one feature at a time
    for drop_col in all_features:
        reduced = [f for f in all_features if f != drop_col]
        experiments.append(evaluate_feature_set(
            train_df, test_df, reduced,
            f"Without '{drop_col}'", seed
        ))

    # 3 — Remove potentially leaky features (category + prevalence)
    safe_features = ["inheritance", "affected_systems"]
    experiments.append(evaluate_feature_set(
        train_df, test_df, safe_features,
        "Safe only (inheritance + affected_systems)", seed
    ))

    # 4 — Single feature experiments
    for col in all_features:
        experiments.append(evaluate_feature_set(
            train_df, test_df, [col],
            f"Single feature: '{col}'", seed
        ))

    df = pd.DataFrame(experiments)

    print("\n" + "="*80)
    print("FEATURE LEAKAGE ABLATION STUDY")
    print("="*80)
    print(df[["Feature Set","Accuracy","Macro F1","Balanced Acc"]].to_string(index=False))
    print("="*80)

    # Save CSV
    csv_path = Path("reports/leakage_ablation.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    # Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    labels  = df["Feature Set"].tolist()
    x       = range(len(labels))
    width   = 0.3

    bars1 = ax.bar([i - width for i in x], df["Accuracy"],
                   width, label="Accuracy",     color="steelblue")
    bars2 = ax.bar([i         for i in x], df["Macro F1"],
                   width, label="Macro F1",     color="coral")
    bars3 = ax.bar([i + width for i in x], df["Balanced Acc"],
                   width, label="Balanced Acc", color="teal")

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=7
            )

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Feature leakage ablation — impact of removing each feature")
    ax.legend()
    plt.tight_layout()
    chart_path = out_dir / "leakage_ablation.png"
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved chart: {chart_path}")

    # Identify leaky features
    baseline_acc = df[df["Feature Set"] == "All features (baseline)"]["Accuracy"].values[0]
    logger.info(f"\nBaseline accuracy (all features): {baseline_acc}")
    logger.info("Feature importance — drop in accuracy when removed:")
    for drop_col in all_features:
        row = df[df["Feature Set"] == f"Without '{drop_col}'"]
        drop_acc = row["Accuracy"].values[0]
        impact   = round(baseline_acc - drop_acc, 4)
        logger.info(f"  Remove '{drop_col}': accuracy {drop_acc} (impact: {impact:+.4f})")

    logger.success("Leakage ablation complete")
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    run_leakage_ablation()