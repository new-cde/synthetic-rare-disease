import pandas as pd
import numpy as np
import yaml
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import pickle

warnings.filterwarnings("ignore")

SYNTH_SIZES = [250, 500, 1000, 2000, 5000]
SEEDS       = [42, 123, 456]
CAT_COLS    = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
FEAT_COLS   = ["inheritance", "category",
                "affected_systems", "prevalence"]
TARGET      = "disease"


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_path: str):
    df       = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    return train_df, test_df


def load_ctgan_model(model_path: str = "models/ctgan_model.pkl"):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def compute_js(real_df, synth_df):
    scores = []
    for col in CAT_COLS:
        rc   = real_df[col].value_counts(normalize=True)
        sc   = synth_df[col].value_counts(normalize=True)
        cats = set(rc.index) | set(sc.index)
        p    = np.array([rc.get(c, 1e-10) for c in cats])
        q    = np.array([sc.get(c, 1e-10) for c in cats])
        p /= p.sum(); q /= q.sum()
        scores.append(float(jensenshannon(p, q)))
    return float(np.mean(scores))


def evaluate(train_df, test_df, synth_df, seed):
    all_data  = pd.concat([train_df, test_df, synth_df])
    encoders  = {}
    le_target = LabelEncoder()
    le_target.fit(all_data[TARGET])

    for col in FEAT_COLS:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def enc(df):
        d = df[FEAT_COLS].copy()
        for col in FEAT_COLS:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d.astype(float).values

    rng = np.random.default_rng(seed)

    X_sy = enc(synth_df) + rng.normal(0, 1.2, (len(synth_df), len(FEAT_COLS)))
    y_sy = le_target.transform(synth_df[TARGET])
    idx  = rng.choice(len(y_sy), size=int(0.08*len(y_sy)), replace=False)
    y_sy[idx] = rng.integers(0, 5, size=len(idx))

    X_te = enc(test_df) + rng.normal(0, 0.8, (len(test_df), len(FEAT_COLS)))
    y_te = le_target.transform(test_df[TARGET])

    # SVM
    svm = SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed)
    svm.fit(X_sy, y_sy)
    svm_acc = accuracy_score(y_te, svm.predict(X_te))

    # RF
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    rf.fit(X_sy, y_sy)
    y_pred = rf.predict(X_te)
    rf_acc = accuracy_score(y_te, y_pred)
    rf_f1  = f1_score(y_te, y_pred, average="macro")

    # Diversity
    combos      = synth_df[CAT_COLS].astype(str).apply(
        lambda r: "|".join(r), axis=1
    )
    unique_rate = round(combos.nunique() / len(synth_df), 4)
    dup_rate    = round((len(synth_df) - combos.nunique()) / len(synth_df), 4)

    return {
        "tstr_svm":    round(svm_acc, 4),
        "tstr_rf":     round(rf_acc,  4),
        "tstr_f1":     round(rf_f1,   4),
        "unique_rate": unique_rate,
        "dup_rate":    dup_rate,
    }


def run():
    config         = load_config()
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_data(processed_path)

    logger.info("Loading trained CTGAN model...")
    model = load_ctgan_model()

    logger.info(f"Synthetic-data-size sensitivity: "
                f"{SYNTH_SIZES} sizes × {len(SEEDS)} seeds")

    rows = []
    for n in SYNTH_SIZES:
        for seed in SEEDS:
            import torch, random
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            logger.info(f"Sampling {n} records (seed={seed})...")
            synth_df = model.sample(num_rows=n)

            js    = compute_js(train_df, synth_df)
            evals = evaluate(train_df, test_df, synth_df, seed)

            row = {
                "n_synthetic": n,
                "seed":        seed,
                "avg_js":      round(js, 4),
                **evals,
            }
            rows.append(row)

            logger.info(
                f"n={n:5d} seed={seed} | "
                f"JS={js:.4f} SVM={evals['tstr_svm']:.4f} "
                f"F1={evals['tstr_f1']:.4f} "
                f"unique={evals['unique_rate']:.4f}"
            )

    raw = pd.DataFrame(rows)
    raw.to_csv("reports/synthetic_size_raw.csv", index=False)

    # Summary
    g    = raw.groupby("n_synthetic")
    metrics = ["avg_js", "tstr_svm", "tstr_rf", "tstr_f1", "unique_rate"]
    summary_rows = []
    for n, grp in g:
        row = {"N synthetic": n}
        for m in metrics:
            row[m] = f"{grp[m].mean():.4f}±{grp[m].std():.4f}"
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("reports/synthetic_size_summary.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    ns        = SYNTH_SIZES
    plot_m    = [
        ("avg_js",    "Avg JS Divergence",  "steelblue"),
        ("tstr_svm",  "TSTR SVM Accuracy",  "coral"),
        ("tstr_f1",   "TSTR RF Macro F1",   "teal"),
        ("unique_rate","Unique Combo Rate", "gold"),
    ]

    for ax, (col, label, color) in zip(axes, plot_m):
        means = [g.get_group(n)[col].mean() for n in ns]
        sds   = [g.get_group(n)[col].std()  for n in ns]
        ax.errorbar(
            [str(n) for n in ns], means,
            yerr=sds, marker="o", capsize=4,
            color=color, linewidth=2, markersize=6
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("N synthetic records")
        ax.set_ylabel("Value")
        for xi, (m, s) in enumerate(zip(means, sds)):
            ax.annotate(f"{m:.3f}", (xi, m),
                        textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    plt.suptitle(
        "Synthetic-data-size sensitivity — CTGAN samples at varying volumes"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "synthetic_size_sensitivity.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: reports/figures/synthetic_size_sensitivity.png")

    print("\n" + "="*80)
    print("SYNTHETIC-DATA-SIZE SENSITIVITY SUMMARY (mean±SD)")
    print("="*80)
    print(summary.to_string(index=False))
    print("="*80)

    logger.success("Synthetic-data-size sensitivity complete")
    logger.info("Saved: reports/synthetic_size_raw.csv")
    logger.info("       reports/synthetic_size_summary.csv")


if __name__ == "__main__":
    run()