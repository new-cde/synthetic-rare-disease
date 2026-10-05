import pandas as pd
import numpy as np
import yaml
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    accuracy_score, classification_report
)

warnings.filterwarnings("ignore")

CAT_COLS  = ["disease", "inheritance", "category",
              "affected_systems", "prevalence"]
FEAT_COLS = ["disease", "inheritance", "category",
              "affected_systems", "prevalence"]


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, synthetic_path: str):
    real_df  = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(
        Path(synthetic_path) / "synthetic_metadata.csv"
    )
    real_train = real_df[real_df["split"] == "train"]
    return real_train, synth_df


def encode(real_df: pd.DataFrame,
           synth_df: pd.DataFrame):
    all_data = pd.concat([real_df, synth_df])
    encoders = {}
    for col in FEAT_COLS:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def enc(df):
        d = df[FEAT_COLS].copy()
        for col in FEAT_COLS:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d.astype(float).values

    # Balance: use same number from each
    n       = min(len(real_df), len(synth_df))
    X_real  = enc(real_df.sample(n,  random_state=42))
    X_synth = enc(synth_df.sample(n, random_state=42))

    X = np.vstack([X_real, X_synth])
    y = np.array([1] * n + [0] * n)   # 1=real, 0=synthetic

    return X, y


def run_distinguishability(X, y, seed: int) -> dict:
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    models = {
        "Random Forest":     RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=seed
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=seed
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=seed
        ),
    }

    results = {}
    all_fprs, all_tprs, all_aucs = {}, {}, {}

    for name, clf in models.items():
        y_prob = cross_val_predict(
            clf, X, y, cv=cv, method="predict_proba"
        )[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y, y_prob)
        acc = accuracy_score(y, y_pred)

        fpr, tpr, _ = roc_curve(y, y_prob)
        all_fprs[name] = fpr
        all_tprs[name] = tpr
        all_aucs[name] = auc

        risk = ("Low — indistinguishable"  if auc < 0.55 else
                "Low — near chance"        if auc < 0.60 else
                "Medium"                   if auc < 0.75 else
                "High")

        results[name] = {
            "AUC":      round(auc, 4),
            "Accuracy": round(acc, 4),
            "Risk":     risk,
        }

        logger.info(
            f"{name:25s} | AUC={auc:.4f} | "
            f"Acc={acc:.4f} | {risk}"
        )

    return results, all_fprs, all_tprs, all_aucs


def plot_roc(fprs, tprs, aucs, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))

    colors = ["steelblue", "coral", "teal"]
    for (name, fpr), (_, tpr), color in zip(
        fprs.items(), tprs.items(), colors
    ):
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC={aucs[name]:.4f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1,
            label="Random (AUC=0.5)")
    ax.axvline(0.5, color="gray", linestyle=":",
               linewidth=1, alpha=0.5)

    ax.set_xlabel("False positive rate", fontsize=12)
    ax.set_ylabel("True positive rate",  fontsize=12)
    ax.set_title(
        "Real vs synthetic distinguishability\n"
        "ROC curves — AUC near 0.5 = indistinguishable",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    plt.tight_layout()
    path = out_dir / "distinguishability_roc.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def run():
    config         = load_config()
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_train, synth_df = load_data(processed_path, synthetic_path)

    logger.info(f"Real train: {len(real_train)} | "
                f"Synthetic: {len(synth_df)}")
    logger.info("Building combined real+synthetic dataset...")

    X, y = encode(real_train, synth_df)
    logger.info(f"Combined dataset: {X.shape} "
                f"({y.sum()} real, {(y==0).sum()} synthetic)")

    logger.info("\n=== Distinguishability Test (5-fold CV) ===")
    results, fprs, tprs, aucs = run_distinguishability(X, y, seed)

    plot_roc(fprs, tprs, aucs, out_dir)

    # Save results
    df_results = pd.DataFrame(results).T
    df_results.index.name = "Classifier"
    df_results.to_csv("reports/distinguishability.csv")

    # Print summary
    print("\n" + "="*65)
    print("REAL VS SYNTHETIC DISTINGUISHABILITY TEST")
    print("="*65)
    print(f"Dataset: {len(real_train)} real + "
          f"{len(synth_df)} synthetic records")
    print(f"Method: 5-fold stratified CV, balanced classes")
    print(f"Interpretation: AUC near 0.5 = indistinguishable = good\n")
    print(df_results.to_string())
    print("\n" + "-"*65)

    mean_auc = np.mean([r["AUC"] for r in results.values()])
    print(f"Mean AUC across classifiers: {mean_auc:.4f}")
    if mean_auc < 0.55:
        print("Overall: SYNTHETIC DATA IS INDISTINGUISHABLE FROM REAL")
    elif mean_auc < 0.65:
        print("Overall: SYNTHETIC DATA IS NEAR-INDISTINGUISHABLE")
    else:
        print("Overall: SYNTHETIC DATA IS DISTINGUISHABLE FROM REAL")
    print("="*65)

    logger.success("Distinguishability test complete")
    logger.info("Saved: reports/distinguishability.csv")
    logger.info("       reports/figures/distinguishability_roc.png")


if __name__ == "__main__":
    run()