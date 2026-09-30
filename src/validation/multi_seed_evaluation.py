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
from scipy.stats import entropy as scipy_entropy
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    balanced_accuracy_score
)
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings("ignore")

SEEDS      = [42, 123, 456, 789, 1024]
CAT_COLS   = ["disease", "inheritance", "category",
               "affected_systems", "prevalence"]
FEAT_COLS  = ["inheritance", "category",
               "affected_systems", "prevalence"]
TARGET     = "disease"
N_SYNTH    = 1000


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_splits(processed_path: str):
    df       = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    return train_df, val_df, test_df


# ── Encode ────────────────────────────────────────────────────────────────────

def encode_splits(train_df, test_df, synth_df):
    all_data = pd.concat([train_df, test_df, synth_df])
    encoders = {}
    for col in FEAT_COLS:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    le_target = LabelEncoder()
    le_target.fit(all_data[TARGET])

    def enc(df):
        d = df[FEAT_COLS].copy()
        for col in FEAT_COLS:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d.astype(float).values, le_target.transform(df[TARGET])

    return enc(train_df), enc(test_df), enc(synth_df), le_target


# ── Marginal fidelity ─────────────────────────────────────────────────────────

def compute_js(real_df, synth_df):
    scores = []
    for col in CAT_COLS:
        real_p  = real_df[col].value_counts(normalize=True)
        synth_p = synth_df[col].value_counts(normalize=True)
        cats    = set(real_p.index) | set(synth_p.index)
        p = np.array([real_p.get(c, 1e-10)  for c in cats])
        q = np.array([synth_p.get(c, 1e-10) for c in cats])
        p /= p.sum(); q /= q.sum()
        scores.append(float(jensenshannon(p, q)))
    return float(np.mean(scores)), float(np.max(scores))


# ── TSTR ──────────────────────────────────────────────────────────────────────

def run_tstr(X_synth, y_synth, X_test, y_test, seed):
    rng = np.random.default_rng(seed)

    # SVM noise-off
    svm = SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed)
    svm.fit(X_synth, y_synth)
    svm_off = accuracy_score(y_test, svm.predict(X_test))

    # SVM noise-on
    Xs_n = X_synth + rng.normal(0, 1.2, X_synth.shape)
    Xt_n = X_test  + rng.normal(0, 0.8, X_test.shape)
    ys_n = y_synth.copy()
    idx  = rng.choice(len(ys_n), size=int(0.08*len(ys_n)), replace=False)
    ys_n[idx] = rng.integers(0, 5, size=len(idx))
    svm_on_clf = SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed)
    svm_on_clf.fit(Xs_n, ys_n)
    svm_on = accuracy_score(y_test, svm_on_clf.predict(Xt_n))

    # RF macro-F1 noise-on
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    rf.fit(Xs_n, ys_n)
    y_pred = rf.predict(Xt_n)
    rf_acc = accuracy_score(y_test, y_pred)
    rf_f1  = f1_score(y_test, y_pred, average="macro")
    rf_bal = balanced_accuracy_score(y_test, y_pred)

    return {
        "svm_noise_off":       round(svm_off, 4),
        "svm_noise_on":        round(svm_on,  4),
        "rf_accuracy":         round(rf_acc,  4),
        "rf_macro_f1":         round(rf_f1,   4),
        "rf_balanced_acc":     round(rf_bal,  4),
    }


# ── Privacy ───────────────────────────────────────────────────────────────────

def compute_privacy(X_train, X_test, X_synth):
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_train)
    dist_train, _ = nn.kneighbors(X_synth)

    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn2.fit(X_test)
    dist_test, _  = nn2.kneighbors(X_synth)

    avg_train = float(np.mean(dist_train))
    avg_test  = float(np.mean(dist_test))
    ratio     = round(avg_train / avg_test, 4) if avg_test > 0 else 0.0
    exact     = int(np.sum(dist_train == 0))

    return {
        "nn_dist_train":  round(avg_train, 4),
        "nn_dist_test":   round(avg_test,  4),
        "nn_ratio":       ratio,
        "exact_matches":  exact,
    }


# ── Diversity ─────────────────────────────────────────────────────────────────

def compute_diversity(synth_df):
    combos     = synth_df[CAT_COLS].astype(str).apply(
        lambda r: "|".join(r), axis=1
    )
    n          = len(synth_df)
    unique_n   = combos.nunique()
    return {
        "unique_rate":     round(unique_n / n, 4),
        "dup_rate":        round((n - unique_n) / n, 4),
    }


# ── Per-seed run ──────────────────────────────────────────────────────────────

def run_one_seed(seed: int, train_df, test_df, epochs: int) -> dict:
    from sdv.single_table import CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger.info(f"Seed {seed} — training CTGAN ({epochs} epochs)...")

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)

    model = CTGANSynthesizer(
        metadata, epochs=epochs, verbose=False, cuda=False
    )
    model.fit(train_df)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    synth_df = model.sample(num_rows=N_SYNTH)

    # Marginal fidelity
    avg_js, max_js = compute_js(train_df, synth_df)

    # Encode
    (X_tr, y_tr), (X_te, y_te), (X_sy, y_sy), _ = encode_splits(
        train_df, test_df, synth_df
    )

    # TSTR
    tstr = run_tstr(X_sy, y_sy, X_te, y_te, seed)

    # Privacy
    priv = compute_privacy(X_tr, X_te, X_sy)

    # Diversity
    div = compute_diversity(synth_df)

    result = {
        "seed":    seed,
        "avg_js":  avg_js,
        "max_js":  max_js,
        **tstr,
        **priv,
        **div,
    }

    logger.info(
        f"Seed {seed} | JS={avg_js:.4f} | "
        f"TSTR_off={tstr['svm_noise_off']:.4f} | "
        f"TSTR_on={tstr['svm_noise_on']:.4f} | "
        f"NN_ratio={priv['nn_ratio']}"
    )
    return result


# ── Summary stats ─────────────────────────────────────────────────────────────

def summarise(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number).drop(columns=["seed"])
    summary = pd.DataFrame({
        "Mean":  numeric.mean().round(4),
        "SD":    numeric.std().round(4),
        "Min":   numeric.min().round(4),
        "Max":   numeric.max().round(4),
    })
    return summary


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_multi_seed(df: pd.DataFrame, out_dir: Path) -> None:
    metrics = {
        "avg_js":         "Avg JS Divergence",
        "svm_noise_on":   "TSTR SVM (noise-on)",
        "rf_macro_f1":    "TSTR RF Macro F1",
        "nn_ratio":       "NN Train/Test Ratio",
    }
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for ax, (col, label) in zip(axes, metrics.items()):
        mean_val = df[col].mean()
        sd_val   = df[col].std()
        ax.bar(df["seed"].astype(str), df[col],
               color="steelblue", alpha=0.8)
        ax.axhline(mean_val, color="red", linestyle="--",
                   linewidth=1.5, label=f"Mean={mean_val:.4f}")
        ax.fill_between(
            range(len(df)), mean_val - sd_val, mean_val + sd_val,
            alpha=0.15, color="red", label=f"±SD={sd_val:.4f}"
        )
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Seed")
        ax.set_ylabel("Value")
        ax.legend(fontsize=8)

    plt.suptitle("Multi-seed evaluation — CTGAN stability across 5 seeds")
    plt.tight_layout()
    path = out_dir / "multi_seed_evaluation.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    config         = load_config()
    epochs         = config["model"]["epochs"]
    processed_path = config["data"]["processed_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df, _, test_df = load_splits(processed_path)

    logger.info(f"Running multi-seed evaluation — {len(SEEDS)} seeds, "
                f"{epochs} epochs each")
    logger.info(f"Seeds: {SEEDS}")
    logger.info(f"Estimated time: {len(SEEDS) * 5}–{len(SEEDS) * 8} minutes")

    all_results = []
    for seed in SEEDS:
        result = run_one_seed(seed, train_df, test_df, epochs)
        all_results.append(result)

    raw_df = pd.DataFrame(all_results)
    raw_df.to_csv("reports/multi_seed_raw.csv", index=False)

    summary = summarise(raw_df)
    summary.to_csv("reports/multi_seed_summary.csv")

    plot_multi_seed(raw_df, out_dir)

    print("\n" + "="*70)
    print(f"MULTI-SEED EVALUATION SUMMARY (n={len(SEEDS)} seeds)")
    print("="*70)
    print(summary.to_string())
    print("\nPer-seed results:")
    print(raw_df.to_string(index=False))
    print("="*70)

    # Key numbers for paper
    print("\nKey numbers for manuscript:")
    for metric, label in [
        ("avg_js",       "Avg JS divergence"),
        ("svm_noise_on", "TSTR SVM noise-on"),
        ("rf_macro_f1",  "TSTR RF Macro F1"),
        ("nn_ratio",     "NN train/test ratio"),
        ("unique_rate",  "Unique combination rate"),
    ]:
        m = raw_df[metric].mean()
        s = raw_df[metric].std()
        print(f"  {label:30s}: {m:.4f} ± {s:.4f}")

    logger.success("Multi-seed evaluation complete")
    logger.info("Saved: reports/multi_seed_raw.csv")
    logger.info("       reports/multi_seed_summary.csv")


if __name__ == "__main__":
    run()