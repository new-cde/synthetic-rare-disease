import pandas as pd
import numpy as np
import yaml
import random
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
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

FRACTIONS = [0.20, 0.40, 0.60, 0.80, 1.00]
SEEDS     = [42, 123, 456]
CAT_COLS  = ["disease", "inheritance", "category",
              "affected_systems", "prevalence"]
FEAT_COLS = ["inheritance", "category",
              "affected_systems", "prevalence"]
TARGET    = "disease"
N_SYNTH   = 1000


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_splits(processed_path: str):
    df       = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    return train_df, test_df


def subset_balanced(df: pd.DataFrame,
                    fraction: float,
                    seed: int) -> pd.DataFrame:
    """Sample fraction of data while keeping class balance."""
    return (
        df.groupby(TARGET, group_keys=False)
          .apply(lambda g: g.sample(
              frac=fraction, random_state=seed
          ))
          .reset_index(drop=True)
    )


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


def encode_and_evaluate(train_df, test_df, synth_df, seed):
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

    def add_noise(X, scale):
        return X + rng.normal(0, scale, X.shape)

    def label_noise(y):
        y2 = y.copy()
        idx = rng.choice(len(y2), size=int(0.08*len(y2)), replace=False)
        y2[idx] = rng.integers(0, 5, size=len(idx))
        return y2

    X_sy = add_noise(enc(synth_df), 1.2)
    y_sy = label_noise(le_target.transform(synth_df[TARGET]))
    X_te = add_noise(enc(test_df),  0.8)
    y_te = le_target.transform(test_df[TARGET])

    # TSTR — SVM
    svm = SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed)
    svm.fit(X_sy, y_sy)
    tstr_svm = accuracy_score(y_te, svm.predict(X_te))

    # TSTR — RF macro-F1
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    rf.fit(X_sy, y_sy)
    y_pred    = rf.predict(X_te)
    tstr_rf   = accuracy_score(y_te, y_pred)
    tstr_f1   = f1_score(y_te, y_pred, average="macro")

    # Privacy — NN ratio
    X_tr = enc(train_df)
    nn   = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_tr)
    d_tr, _ = nn.kneighbors(enc(synth_df))

    nn2  = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn2.fit(enc(test_df))
    d_te, _ = nn2.kneighbors(enc(synth_df))

    nn_ratio = round(
        float(np.mean(d_tr)) / float(np.mean(d_te)), 4
    ) if np.mean(d_te) > 0 else 0.0

    # Diversity
    combos     = synth_df[CAT_COLS].astype(str).apply(
        lambda r: "|".join(r), axis=1
    )
    unique_rate = round(combos.nunique() / len(synth_df), 4)

    return {
        "tstr_svm":    round(tstr_svm, 4),
        "tstr_rf":     round(tstr_rf,  4),
        "tstr_f1":     round(tstr_f1,  4),
        "nn_ratio":    nn_ratio,
        "unique_rate": unique_rate,
    }


def run_one(fraction: float, seed: int,
            train_df: pd.DataFrame,
            test_df:  pd.DataFrame,
            epochs:   int) -> dict:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    subset = subset_balanced(train_df, fraction, seed)
    n_sub  = len(subset)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(subset)

    model = CTGANSynthesizer(
        metadata, epochs=epochs, verbose=False, cuda=False
    )
    model.fit(subset)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    synth_df = model.sample(num_rows=N_SYNTH)

    js     = compute_js(subset, synth_df)
    evals  = encode_and_evaluate(subset, test_df, synth_df, seed)

    result = {
        "fraction":    fraction,
        "n_train":     n_sub,
        "seed":        seed,
        "avg_js":      round(js, 4),
        **evals,
    }

    logger.info(
        f"frac={fraction:.0%} n={n_sub:4d} seed={seed} | "
        f"JS={js:.4f} TSTR_svm={evals['tstr_svm']:.4f} "
        f"F1={evals['tstr_f1']:.4f} "
        f"unique={evals['unique_rate']:.4f}"
    )
    return result


def run():
    config         = load_config()
    epochs         = config["model"]["epochs"]
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = load_splits(processed_path)

    total = len(FRACTIONS) * len(SEEDS)
    logger.info(f"Training-data-size sensitivity: "
                f"{len(FRACTIONS)} fractions × {len(SEEDS)} seeds "
                f"= {total} runs at {epochs} epochs each")
    logger.info(f"Estimated time: {total*4}–{total*7} minutes")

    rows = []
    for fraction in FRACTIONS:
        for seed in SEEDS:
            rows.append(run_one(fraction, seed,
                                train_df, test_df, epochs))

    raw = pd.DataFrame(rows)
    raw.to_csv("reports/training_size_raw.csv", index=False)

    # Summary — mean ± SD per fraction
    metrics = ["avg_js", "tstr_svm", "tstr_rf",
               "tstr_f1", "nn_ratio", "unique_rate"]
    g       = raw.groupby("fraction")
    summary_rows = []
    for frac, grp in g:
        n_train = int(grp["n_train"].mean())
        row = {"Fraction": f"{frac:.0%}", "N train": n_train}
        for m in metrics:
            row[m] = (f"{grp[m].mean():.4f}"
                      f"±{grp[m].std():.4f}")
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv("reports/training_size_summary.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fracs     = [f for f in FRACTIONS]
    plot_metrics = [
        ("avg_js",    "Avg JS Divergence", "steelblue"),
        ("tstr_svm",  "TSTR SVM Accuracy", "coral"),
        ("tstr_f1",   "TSTR RF Macro F1",  "teal"),
        ("unique_rate","Unique Combo Rate", "gold"),
    ]

    for ax, (col, label, color) in zip(axes, plot_metrics):
        means = [g.get_group(f)[col].mean() for f in fracs]
        sds   = [g.get_group(f)[col].std()  for f in fracs]
        ax.errorbar(
            [f"{f:.0%}" for f in fracs], means,
            yerr=sds, marker="o", capsize=4,
            color=color, linewidth=2, markersize=6
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Training data fraction")
        ax.set_ylabel("Value")
        for xi, (m, s) in enumerate(zip(means, sds)):
            ax.annotate(f"{m:.3f}", (xi, m),
                        textcoords="offset points",
                        xytext=(0, 8), ha="center", fontsize=8)

    plt.suptitle(
        "Training-data-size sensitivity — CTGAN at varying fractions"
    )
    plt.tight_layout()
    fig.savefig(out_dir / "training_size_sensitivity.png", dpi=150)
    plt.close(fig)
    logger.info("Saved: reports/figures/training_size_sensitivity.png")

    print("\n" + "="*90)
    print("TRAINING-DATA-SIZE SENSITIVITY SUMMARY (mean±SD)")
    print("="*90)
    print(summary.to_string(index=False))
    print("="*90)

    logger.success("Training-data-size sensitivity complete")
    logger.info("Saved: reports/training_size_raw.csv")
    logger.info("       reports/training_size_summary.csv")


if __name__ == "__main__":
    run()