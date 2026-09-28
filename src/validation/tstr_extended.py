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
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

FEATS  = ["inheritance", "category", "affected_systems", "prevalence"]
TARGET = "disease"
SEEDS  = [42, 123, 456, 789, 1024]
METRICS = ["Accuracy", "MacroF1", "WeightedF1", "BalancedAcc",
           "MacroPrecision", "MacroRecall", "ROC_AUC"]


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(cfg):
    real  = pd.read_csv(Path(cfg["data"]["processed_path"]) / "metadata_clean.csv")
    synth = pd.read_csv(Path(cfg["data"]["synthetic_path"]) / "synthetic_metadata.csv")
    logger.info(f"Synthetic file: {len(synth)} rows")
    enc = {c: LabelEncoder().fit(pd.concat([real[c], synth[c]]).astype(str))
           for c in FEATS}
    le_t = LabelEncoder().fit(real[TARGET])

    def prep(df):
        X = np.column_stack([enc[c].transform(df[c].astype(str))
                             for c in FEATS]).astype(float)
        return X, le_t.transform(df[TARGET])

    tr = real[real["split"] == "train"]
    te = real[real["split"] == "test"]
    return prep(tr), prep(te), prep(synth), le_t


def make_models(seed):
    return {
        "SVM": SVC(kernel="rbf", C=0.5, gamma="scale",
                   probability=True, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=5, min_samples_leaf=8,
            max_features="sqrt", random_state=seed, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=60, max_depth=2, learning_rate=0.08,
            subsample=0.8, random_state=seed),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, C=0.1, solver="saga", random_state=seed),
    }


def score(model, X, y, k):
    pred = model.predict(X)
    out = {
        "Accuracy":       accuracy_score(y, pred),
        "MacroF1":        f1_score(y, pred, average="macro", zero_division=0),
        "WeightedF1":     f1_score(y, pred, average="weighted", zero_division=0),
        "BalancedAcc":    balanced_accuracy_score(y, pred),
        "MacroPrecision": precision_score(y, pred, average="macro", zero_division=0),
        "MacroRecall":    recall_score(y, pred, average="macro", zero_division=0),
    }
    try:
        proba = model.predict_proba(X)
        full = np.zeros((len(X), k))
        full[:, model.classes_] = proba
        out["ROC_AUC"] = roc_auc_score(y, full, multi_class="ovr",
                                       average="macro", labels=list(range(k)))
    except Exception:
        out["ROC_AUC"] = np.nan
    return out, pred


def perturb(X, y, seed, noise, is_train):
    if not noise:
        return X, y
    rng = np.random.default_rng(seed + (0 if is_train else 1))
    X = X + rng.normal(0, 1.2 if is_train else 0.8, X.shape)
    y = y.copy()
    if is_train:
        n = int(0.08 * len(y))
        idx = rng.choice(len(y), size=n, replace=False)
        y[idx] = rng.integers(0, 5, size=n)
    return X, y


def run():
    cfg = load_config()
    (Xr_tr, yr_tr), (Xr_te, yr_te), (Xs, ys), le_t = load_data(cfg)
    k = len(le_t.classes_)
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = {
        "TRTR": ((Xr_tr, yr_tr), (Xr_te, yr_te)),
        "TSTR": ((Xs, ys),       (Xr_te, yr_te)),
        "TRTS": ((Xr_tr, yr_tr), (Xs, ys)),
    }

    rows, cms = [], {}
    for protocol, noise in (("Noise OFF", False), ("Noise ON", True)):
        for arm, ((Xa, ya), (Xb, yb)) in arms.items():
            for seed in SEEDS:
                Xa_p, ya_p = perturb(Xa, ya, seed, noise, True)
                Xb_p, yb_p = perturb(Xb, yb, seed, noise, False)
                for name, model in make_models(seed).items():
                    model.fit(Xa_p, ya_p)
                    m, pred = score(model, Xb_p, yb_p, k)
                    rows.append({"Protocol": protocol, "Arm": arm,
                                 "Model": name, "Seed": seed, **m})
                    if (protocol == "Noise OFF" and name == "SVM"
                            and seed == 42):
                        cms[arm] = (yb_p, pred)
            logger.info(f"done: {protocol} / {arm}")

    raw = pd.DataFrame(rows)
    raw.to_csv("reports/tstr_extended_raw.csv", index=False)
    agg = (raw.groupby(["Protocol", "Arm", "Model"])[METRICS]
              .agg(["mean", "std"]).round(4))
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()
    agg.to_csv("reports/tstr_extended.csv", index=False)

    for protocol in ("Noise OFF", "Noise ON"):
        for metric in ("Accuracy", "MacroF1", "BalancedAcc", "ROC_AUC"):
            sub = agg[agg["Protocol"] == protocol].copy()
            sub["cell"] = sub.apply(
                lambda r: f"{r[metric + '_mean']:.3f}±{r[metric + '_std']:.3f}",
                axis=1)
            t = sub.pivot(index="Model", columns="Arm", values="cell")
            t = t[["TRTR", "TSTR", "TRTS"]]
            print("\n" + "=" * 70)
            print(f"{metric} | {protocol} (mean±SD over {len(SEEDS)} seeds)")
            print("=" * 70)
            print(t.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, protocol in zip(axes, ("Noise OFF", "Noise ON")):
        sub = agg[agg["Protocol"] == protocol]
        models = sub["Model"].unique()
        x = np.arange(len(models))
        for i, arm in enumerate(["TRTR", "TSTR", "TRTS"]):
            s = sub[sub["Arm"] == arm].set_index("Model").loc[models]
            ax.bar(x + (i - 1) * 0.27, s["Accuracy_mean"], 0.27,
                   yerr=s["Accuracy_std"], capsize=3, label=arm)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15)
        ax.set_title(protocol)
        ax.set_ylim(0, 1.1)
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    plt.suptitle("TRTR vs TSTR vs TRTS")
    plt.tight_layout()
    fig.savefig(out_dir / "trtr_tstr_trts.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    labels = [c.replace("_", " ")[:18] for c in le_t.classes_]
    for ax, arm in zip(axes, ["TRTR", "TSTR", "TRTS"]):
        y_true, y_pred = cms[arm]
        ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred, labels=range(k)),
                               display_labels=labels).plot(
            ax=ax, xticks_rotation=45, colorbar=False)
        ax.set_title(f"SVM {arm} (noise off)")
    plt.tight_layout()
    fig.savefig(out_dir / "confusion_trtr_tstr_trts.png", dpi=150)
    plt.close(fig)
    logger.success("Saved reports/tstr_extended.csv and figures")


if __name__ == "__main__":
    run()