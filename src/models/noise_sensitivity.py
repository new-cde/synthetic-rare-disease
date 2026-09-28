import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score

LABEL_NOISE = [0.0, 0.02, 0.05, 0.08, 0.10, 0.15]
SEEDS       = [42, 123, 456, 789, 1024]
FEATS       = ["inheritance", "category", "affected_systems", "prevalence"]
TARGET      = "disease"


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_encoded(processed_path):
    df = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    for col in FEATS:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
    le = LabelEncoder()
    df[TARGET] = le.fit_transform(df[TARGET])
    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    return (tr[FEATS].values, tr[TARGET].values,
            te[FEATS].values, te[TARGET].values, len(le.classes_))


def make_models(seed):
    return {
        "SVM": SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=8,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.8, random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, solver="saga", random_state=seed),
    }


def run_once(X_tr, y_tr, X_te, y_te, n_classes, label_p, feat_noise, seed):
    rng = np.random.default_rng(seed)
    Xtr, Xte = X_tr.astype(float), X_te.astype(float)
    if feat_noise:
        Xtr = Xtr + rng.normal(0, 1.2, Xtr.shape)   # same as baseline scripts
        Xte = Xte + rng.normal(0, 0.8, Xte.shape)
    y = y_tr.copy()
    if label_p > 0:
        n = int(label_p * len(y))
        idx = rng.choice(len(y), size=n, replace=False)
        y[idx] = rng.integers(0, n_classes, size=n)
    rows = []
    for name, model in make_models(seed).items():
        model.fit(Xtr, y)
        pred = model.predict(Xte)
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_te, pred),
            "Macro F1": f1_score(y_te, pred, average="macro"),
        })
    return rows


def run():
    cfg = load_config()
    X_tr, y_tr, X_te, y_te, k = load_encoded(cfg["data"]["processed_path"])
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for feat_noise in (True, False):
        for p in LABEL_NOISE:
            for seed in SEEDS:
                for r in run_once(X_tr, y_tr, X_te, y_te, k, p, feat_noise, seed):
                    records.append({"Feature Noise": feat_noise,
                                    "Label Noise": p,
                                    "Effective Noise": round(p * (k - 1) / k, 4),
                                    "Seed": seed, **r})
            logger.info(f"done: feature_noise={feat_noise}, label_noise={p:.0%}")

    raw = pd.DataFrame(records)
    raw.to_csv("reports/noise_sensitivity_raw.csv", index=False)

    agg = (raw.groupby(["Feature Noise", "Model", "Label Noise"])
              .agg(Acc_mean=("Accuracy", "mean"), Acc_sd=("Accuracy", "std"),
                   F1_mean=("Macro F1", "mean"), F1_sd=("Macro F1", "std"))
              .round(4).reset_index())
    agg.to_csv("reports/noise_sensitivity.csv", index=False)

    for fn in (True, False):
        sub = agg[agg["Feature Noise"] == fn].copy()
        sub["cell"] = sub.apply(
            lambda r: f"{r.Acc_mean:.3f}±{r.Acc_sd:.3f}", axis=1)
        table = sub.pivot(index="Model", columns="Label Noise", values="cell")
        print("\n" + "=" * 90)
        print(f"TEST ACCURACY (mean±SD over {len(SEEDS)} seeds) | "
              f"Gaussian feature noise: {'ON' if fn else 'OFF'}")
        print("=" * 90)
        print(table.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, fn in zip(axes, (True, False)):
        for model in agg["Model"].unique():
            s = agg[(agg["Feature Noise"] == fn) & (agg["Model"] == model)]
            ax.errorbar(s["Label Noise"] * 100, s["Acc_mean"],
                        yerr=s["Acc_sd"], marker="o", capsize=3, label=model)
        ax.axvline(8, color="grey", linestyle="--", linewidth=1)
        ax.set_title(f"Feature noise {'ON' if fn else 'OFF'}")
        ax.set_xlabel("Label noise (%)")
        ax.set_ylim(0, 1.05)
    axes[0].set_ylabel("Test accuracy")
    axes[0].legend(fontsize=8)
    plt.suptitle("Sensitivity of classifiers to label and feature noise")
    plt.tight_layout()
    fig.savefig(out_dir / "noise_sensitivity.png", dpi=150)
    plt.close(fig)
    logger.success("Saved reports/noise_sensitivity.csv and figure")


if __name__ == "__main__":
    run()