import pandas as pd
import numpy as np
import random
import torch
import mlflow
import pickle
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger

SEED = 42

def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
from sdv.single_table import (
    CTGANSynthesizer,
    TVAESynthesizer,
    GaussianCopulaSynthesizer,
    CopulaGANSynthesizer,
)
from sdv.metadata import SingleTableMetadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, f1_score,
    balanced_accuracy_score
)
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Data ──────────────────────────────────────────────────────────────────────

def load_splits(processed_path: str):
    df = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    train_df = df[df["split"] == "train"].drop(columns=["split"])
    val_df   = df[df["split"] == "val"].drop(columns=["split"])
    test_df  = df[df["split"] == "test"].drop(columns=["split"])
    logger.info(f"Train: {train_df.shape} | Val: {val_df.shape} | Test: {test_df.shape}")
    return train_df, val_df, test_df


# ── Generative models ─────────────────────────────────────────────────────────

def get_generators(metadata: SingleTableMetadata, epochs: int) -> dict:
    return {
        "Gaussian Copula": GaussianCopulaSynthesizer(metadata),
        "CTGAN":           CTGANSynthesizer(metadata, epochs=epochs,
                                            verbose=False, cuda=False),
        "TVAE":            TVAESynthesizer(metadata, epochs=epochs,
                                           cuda=False),
        "CopulaGAN":       CopulaGANSynthesizer(metadata, epochs=epochs,
                                                 verbose=False, cuda=False),
    }


def train_and_sample(model, train_df: pd.DataFrame,
                     n: int, model_name: str, seed: int) -> pd.DataFrame:
    logger.info(f"Training {model_name}...")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model.fit(train_df)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    synthetic_df = model.sample(num_rows=n)
    logger.success(f"{model_name} - generated {len(synthetic_df)} records")
    return synthetic_df


# ── Marginal fidelity ─────────────────────────────────────────────────────────

def marginal_fidelity(real_df: pd.DataFrame,
                      synth_df: pd.DataFrame) -> dict:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    js_scores, kl_scores = [], []

    for col in cat_cols:
        real_c  = real_df[col].value_counts(normalize=True)
        synth_c = synth_df[col].value_counts(normalize=True)
        cats    = set(real_c.index) | set(synth_c.index)
        p = np.array([real_c.get(c, 1e-10) for c in cats])
        q = np.array([synth_c.get(c, 1e-10) for c in cats])
        p /= p.sum(); q /= q.sum()
        js_scores.append(float(jensenshannon(p, q)))
        kl_scores.append(float(entropy(p, q)))

    return {
        "avg_js":  round(np.mean(js_scores), 4),
        "avg_kl":  round(np.mean(kl_scores), 4),
        "max_js":  round(np.max(js_scores),  4),
    }


# ── TSTR utility ──────────────────────────────────────────────────────────────

def tstr_evaluation(train_df: pd.DataFrame,
                    synth_df: pd.DataFrame,
                    test_df:  pd.DataFrame,
                    seed: int) -> dict:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    target_col   = "disease"

    # Fit encoders on combined vocab
    encoders  = {}
    le_target = LabelEncoder()
    all_data  = pd.concat([train_df, synth_df, test_df])
    le_target.fit(all_data[target_col])

    for col in feature_cols:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def encode(df):
        d = df.copy()
        for col in feature_cols:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d[feature_cols].astype(float)

    rng = np.random.default_rng(seed)

    def noisy(X):
        return X + rng.normal(0, 1.2, X.shape)

    X_synth = noisy(encode(synth_df))
    X_test  = noisy(encode(test_df))
    y_synth = le_target.transform(synth_df[target_col])
    y_test  = le_target.transform(test_df[target_col])

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    clf.fit(X_synth, y_synth)
    y_pred = clf.predict(X_test)

    return {
        "tstr_accuracy":          round(accuracy_score(y_test, y_pred), 4),
        "tstr_macro_f1":          round(f1_score(y_test, y_pred, average="macro"), 4),
        "tstr_weighted_f1":       round(f1_score(y_test, y_pred, average="weighted"), 4),
        "tstr_balanced_accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
    }


# ── Diversity ─────────────────────────────────────────────────────────────────

def diversity_metrics(synth_df: pd.DataFrame) -> dict:
    cat_cols    = ["disease", "inheritance", "category",
                   "affected_systems", "prevalence"]
    df_str      = synth_df[cat_cols].astype(str)
    combos      = df_str.apply(lambda r: "|".join(r), axis=1)
    n           = len(synth_df)
    unique_rate = round(combos.nunique() / n, 4)
    exact_dups  = round((n - combos.nunique()) / n, 4)

    return {
        "unique_combination_rate": unique_rate,
        "within_synthetic_repetition_rate": exact_dups,
        "unique_combinations":     int(combos.nunique()),
    }


# ── Comparison table ──────────────────────────────────────────────────────────

def save_comparison_table(results: dict, out_dir: Path) -> pd.DataFrame:
    rows = []
    for model_name, m in results.items():
        rows.append({
            "Generator":            model_name,
            "Avg JS Divergence":    m["avg_js"],
            "Avg KL Divergence":    m["avg_kl"],
            "Max JS Divergence":    m["max_js"],
            "TSTR Accuracy":        m["tstr_accuracy"],
            "TSTR Macro F1":        m["tstr_macro_f1"],
            "TSTR Weighted F1":     m["tstr_weighted_f1"],
            "TSTR Balanced Acc":    m["tstr_balanced_accuracy"],
            "Unique Combo Rate":    m["unique_combination_rate"],
            "Within-Synthetic Repetition Rate": m["within_synthetic_repetition_rate"],
        })

    df = pd.DataFrame(rows)

    print("\n" + "="*80)
    print("GENERATOR COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)

    csv_path = Path("reports/generator_comparison.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"Comparison table saved to {csv_path}")

    generators = df["Generator"].tolist()
    x          = range(len(generators))
    width      = 0.35

    # ── Chart 1: Marginal fidelity ────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6))
    bars_js = ax1.bar(
        [i - width/2 for i in x], df["Avg JS Divergence"],
        width, label="Avg JS Divergence",
        color="#2196F3", edgecolor="white", linewidth=0.8
    )
    bars_kl = ax1.bar(
        [i + width/2 for i in x], df["Avg KL Divergence"],
        width, label="Avg KL Divergence",
        color="#00BCD4", edgecolor="white", linewidth=0.8
    )
    for bar in list(bars_js) + list(bars_kl):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1a1a1a"
        )
    ax1.axhline(
        0.1, color="#4CAF50", linestyle="--",
        linewidth=2.0, label="Good threshold (JS = 0.10)"
    )
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(
        generators, rotation=15, ha="right",
        fontsize=12, fontweight="bold"
    )
    ax1.set_ylim(
        0,
        max(df["Avg JS Divergence"].max(),
            df["Avg KL Divergence"].max()) * 1.5
    )
    ax1.set_xlabel("Generative model", fontsize=14,
                   fontweight="bold", labelpad=12)
    ax1.set_ylabel("Divergence score", fontsize=14,
                   fontweight="bold", labelpad=12)
    ax1.set_title(
        "Generator comparison — marginal fidelity (JS and KL divergence)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax1.legend(fontsize=12, frameon=True)
    ax1.tick_params(axis="y", labelsize=12)
    for label in ax1.get_xticklabels() + ax1.get_yticklabels():
        label.set_fontweight("bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    plt.tight_layout()
    fig1.savefig(out_dir / "generator_comparison_fidelity.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig1)
    logger.info(f"Saved: {out_dir / 'generator_comparison_fidelity.png'}")

    # ── Chart 2: Downstream utility ───────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    bars_acc = ax2.bar(
        [i - width/2 for i in x], df["TSTR Accuracy"],
        width, label="TSTR Accuracy",
        color="#FF5722", edgecolor="white", linewidth=0.8
    )
    bars_f1 = ax2.bar(
        [i + width/2 for i in x], df["TSTR Macro F1"],
        width, label="TSTR Macro F1",
        color="#FF9800", edgecolor="white", linewidth=0.8
    )
    for bar in list(bars_acc) + list(bars_f1):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{bar.get_height():.4f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1a1a1a"
        )
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(
        generators, rotation=15, ha="right",
        fontsize=12, fontweight="bold"
    )
    ax2.set_ylim(0, 1.10)
    ax2.set_xlabel("Generative model", fontsize=14,
                   fontweight="bold", labelpad=12)
    ax2.set_ylabel("Score", fontsize=14,
                   fontweight="bold", labelpad=12)
    ax2.set_title(
        "Generator comparison — downstream utility (TSTR accuracy and Macro F1)",
        fontsize=14, fontweight="bold", pad=15
    )
    ax2.legend(fontsize=12, frameon=True)
    ax2.tick_params(axis="y", labelsize=12)
    for label in ax2.get_xticklabels() + ax2.get_yticklabels():
        label.set_fontweight("bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    fig2.savefig(out_dir / "generator_comparison_utility.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Saved: {out_dir / 'generator_comparison_utility.png'}")

    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def run_generator_comparison():
    config         = load_config()
    seed           = config["project"]["seed"]
    epochs         = config["model"]["epochs"]
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(synthetic_path).mkdir(parents=True, exist_ok=True)

    n_synth = 1000

    train_df, val_df, test_df = load_splits(processed_path)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train_df)
    meta_path = Path("config/metadata.json")
    meta_path.unlink(missing_ok=True)
    metadata.save_to_json(str(meta_path))
    generators  = get_generators(metadata, epochs)
    all_results = {}

    mlflow.set_experiment("synthetic-rare-disease")

    for model_name, model in generators.items():
        logger.info(f"\n{'='*50}\nTraining: {model_name}\n{'='*50}")

        with mlflow.start_run(
            run_name=f"generator_{model_name.lower().replace(' ', '_')}"
        ):
            mlflow.log_params({
                "generator": model_name,
                "epochs":    epochs if model_name != "Gaussian Copula" else "N/A",
                "seed":      seed,
                "n_synth":   n_synth,
            })

            synth_df = train_and_sample(model, train_df, n_synth, model_name, seed)

            # Save synthetic data per generator
            safe_name = model_name.lower().replace(" ", "_")
            synth_path = Path(synthetic_path) / f"synthetic_{safe_name}.csv"
            synth_df.to_csv(synth_path, index=False)

            # Evaluate
            fidelity  = marginal_fidelity(train_df, synth_df)
            utility   = tstr_evaluation(train_df, synth_df, test_df, seed)
            diversity = diversity_metrics(synth_df)

            metrics = {**fidelity, **utility, **diversity}
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(str(synth_path))

            all_results[model_name] = metrics

            logger.info(
                f"{model_name} | JS: {fidelity['avg_js']} | "
                f"TSTR Acc: {utility['tstr_accuracy']} | "
                f"Unique combos: {diversity['unique_combination_rate']}"
            )

    comparison_df = save_comparison_table(all_results, out_dir)

    # Best model by TSTR accuracy
    best_fidelity = comparison_df.loc[comparison_df["Avg JS Divergence"].idxmin(), "Generator"]
    best_utility  = comparison_df.loc[comparison_df["TSTR Accuracy"].idxmax(),     "Generator"]
    logger.success(f"Best marginal fidelity : {best_fidelity}")
    logger.success(f"Best TSTR utility      : {best_utility}")
    logger.info("Check http://localhost:5000 for all runs")


if __name__ == "__main__":
    run_generator_comparison()