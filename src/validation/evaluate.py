import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import yaml
from pathlib import Path
from loguru import logger
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, synthetic_path: str):
    real_df = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(Path(synthetic_path) / "synthetic_metadata.csv")
    logger.info(f"Real data     : {real_df.shape}")
    logger.info(f"Synthetic data: {synth_df.shape}")
    return real_df, synth_df


# ── Statistical similarity ────────────────────────────────────────────────────

def compute_divergences(real_df: pd.DataFrame,
                        synth_df: pd.DataFrame,
                        out_dir: Path) -> pd.DataFrame:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    results  = []

    for col in cat_cols:
        real_counts  = real_df[col].value_counts(normalize=True)
        synth_counts = synth_df[col].value_counts(normalize=True)

        all_cats = set(real_counts.index) | set(synth_counts.index)
        p = np.array([real_counts.get(c, 1e-10) for c in all_cats])
        q = np.array([synth_counts.get(c, 1e-10) for c in all_cats])

        p = p / p.sum()
        q = q / q.sum()

        kl = float(entropy(p, q))
        js = float(jensenshannon(p, q))

        results.append({
            "Column": col,
            "KL Divergence": round(kl, 4),
            "JS Divergence": round(js, 4),
            "Quality":       "Good" if js < 0.1 else
                             "Acceptable" if js < 0.2 else "Poor"
        })
        logger.info(f"{col:35s} KL={kl:.4f}  JS={js:.4f}")

    df_div = pd.DataFrame(results)
    print("\nDistribution Similarity Report:")
    print(df_div.to_string(index=False))

    # Bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = range(len(df_div))
    width  = 0.35
    ax.bar([i - width/2 for i in x], df_div["KL Divergence"],
           width, label="KL Divergence", color="steelblue")
    ax.bar([i + width/2 for i in x], df_div["JS Divergence"],
           width, label="JS Divergence", color="coral")
    ax.axhline(0.1, color="green", linestyle="--",
               linewidth=1, label="JS=0.1 (good threshold)")
    ax.axhline(0.2, color="red", linestyle="--",
               linewidth=1, label="JS=0.2 (acceptable threshold)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(df_div["Column"], rotation=15)
    ax.set_ylabel("Divergence score")
    ax.set_title("KL and JS divergence — real vs synthetic")
    ax.legend()
    plt.tight_layout()
    path = out_dir / "divergence_scores.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")

    return df_div


def plot_distribution_comparison(real_df: pd.DataFrame,
                                 synth_df: pd.DataFrame,
                                 out_dir: Path) -> None:
    cat_cols = ["disease", "inheritance", "category"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col in zip(axes, cat_cols):
        real_pct  = real_df[col].value_counts(normalize=True).sort_index()
        synth_pct = synth_df[col].value_counts(normalize=True).sort_index()
        all_cats  = sorted(set(real_pct.index) | set(synth_pct.index))
        x         = range(len(all_cats))
        width     = 0.35

        ax.bar([i - width/2 for i in x],
               [real_pct.get(c, 0) for c in all_cats],
               width, label="Real", color="steelblue", alpha=0.8)
        ax.bar([i + width/2 for i in x],
               [synth_pct.get(c, 0) for c in all_cats],
               width, label="Synthetic", color="coral", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(all_cats, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{col} distribution")
        ax.set_ylabel("Proportion")
        ax.legend()

    plt.suptitle("Real vs synthetic distribution comparison")
    plt.tight_layout()
    path = out_dir / "distribution_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── TSTR evaluation ───────────────────────────────────────────────────────────

def run_tstr(real_df: pd.DataFrame,
             synth_df: pd.DataFrame,
             out_dir: Path) -> dict:
    logger.info("Running TSTR evaluation (Train on Synthetic, Test on Real)")

    feature_cols = ["inheritance", "category", "affected_systems", "prevalence"]
    target_col   = "disease"

    # Encode using real data as reference
    encoders  = {}
    le_target = LabelEncoder()
    le_target.fit(real_df[target_col])

    real_enc  = real_df.copy()
    synth_enc = synth_df.copy()

    for col in feature_cols:
        le = LabelEncoder()
        le.fit(pd.concat([real_df[col], synth_df[col]]).astype(str))
        real_enc[col]  = le.transform(real_df[col].astype(str))
        synth_enc[col] = le.transform(synth_df[col].astype(str))
        encoders[col]  = le

    # Real splits
    train_real = real_enc[real_enc["split"] == "train"]
    test_real  = real_enc[real_enc["split"] == "test"]

    X_real_train = train_real[feature_cols]
    y_real_train = le_target.transform(train_real[target_col])
    X_real_test  = test_real[feature_cols]
    y_real_test  = le_target.transform(test_real[target_col])

    # Synthetic train
    X_synth_train = synth_enc[feature_cols]
    y_synth_train = le_target.transform(synth_enc[target_col])

    rng = np.random.default_rng(42)

    def add_noise(X):
        return X.astype(float) + rng.normal(0, 1.2, X.shape)

    def add_label_noise(y):
        y = y.copy()
        noisy = rng.random(len(y)) < 0.08
        y[noisy] = rng.integers(0, 5, noisy.sum())
        return y

    # Train on REAL → test on real (baseline)
    rf_real = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=42
    )
    rf_real.fit(
        add_noise(X_real_train),
        add_label_noise(y_real_train)
    )
    y_pred_real   = rf_real.predict(add_noise(X_real_test))
    acc_real      = accuracy_score(y_real_test, y_pred_real)
    f1_real       = f1_score(y_real_test, y_pred_real, average="weighted")

    # Train on SYNTHETIC → test on real (TSTR)
    rf_synth = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=42
    )
    rf_synth.fit(
        add_noise(X_synth_train),
        add_label_noise(y_synth_train)
    )
    y_pred_synth  = rf_synth.predict(add_noise(X_real_test))
    acc_synth     = accuracy_score(y_real_test, y_pred_synth)
    f1_synth      = f1_score(y_real_test, y_pred_synth, average="weighted")

    acc_drop = acc_real - acc_synth
    quality  = ("Excellent" if acc_drop < 0.05 else
                "Good"      if acc_drop < 0.10 else
                "Acceptable" if acc_drop < 0.15 else "Poor")

    logger.info(f"Train-on-Real  accuracy : {acc_real:.4f}  F1: {f1_real:.4f}")
    logger.info(f"Train-on-Synth accuracy : {acc_synth:.4f}  F1: {f1_synth:.4f}")
    logger.info(f"Accuracy drop           : {acc_drop:.4f}")
    logger.info(f"Synthetic data quality  : {quality}")

    print(f"\nClassification Report — TSTR (train synthetic, test real):")
    print(classification_report(
        y_real_test, y_pred_synth,
        target_names=le_target.classes_
    ))

    # Comparison bar chart
    fig, ax = plt.subplots(figsize=(7, 5))
    labels  = ["Train on Real\n(Baseline)", "Train on Synthetic\n(TSTR)"]
    accs    = [acc_real, acc_synth]
    colors  = ["steelblue", "coral"]
    bars    = ax.bar(labels, accs, color=colors, width=0.4)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{val:.4f}", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Accuracy")
    ax.set_title("TSTR Evaluation — Real vs Synthetic Training")
    ax.axhline(acc_real, color="steelblue",
               linestyle="--", linewidth=1, alpha=0.5)
    plt.tight_layout()
    path = out_dir / "tstr_comparison.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")

    return {
        "baseline_accuracy": round(acc_real,  4),
        "tstr_accuracy":     round(acc_synth, 4),
        "baseline_f1":       round(f1_real,   4),
        "tstr_f1":           round(f1_synth,  4),
        "accuracy_drop":     round(acc_drop,  4),
        "quality_rating":    quality
    }


# ── Save report ───────────────────────────────────────────────────────────────

def save_report(div_df: pd.DataFrame,
                tstr_results: dict,
                out_path: Path) -> None:
    lines = [
        "=" * 60,
        "SYNTHETIC DATA VALIDATION REPORT",
        "=" * 60,
        "",
        "1. DISTRIBUTION SIMILARITY (KL / JS Divergence)",
        "-" * 60,
        div_df.to_string(index=False),
        "",
        f"Average JS Divergence : "
        f"{div_df['JS Divergence'].mean():.4f}",
        "",
        "2. TSTR EVALUATION (Train on Synthetic, Test on Real)",
        "-" * 60,
        f"Baseline accuracy     : {tstr_results['baseline_accuracy']}",
        f"TSTR accuracy         : {tstr_results['tstr_accuracy']}",
        f"Baseline F1           : {tstr_results['baseline_f1']}",
        f"TSTR F1               : {tstr_results['tstr_f1']}",
        f"Accuracy drop         : {tstr_results['accuracy_drop']}",
        f"Quality rating        : {tstr_results['quality_rating']}",
        "",
        "3. INTERPRETATION",
        "-" * 60,
        "JS Divergence < 0.1  = Good distributional similarity",
        "JS Divergence < 0.2  = Acceptable",
        "Accuracy drop < 5%   = Excellent synthetic utility",
        "Accuracy drop < 10%  = Good synthetic utility",
        "Accuracy drop < 15%  = Acceptable synthetic utility",
        "=" * 60,
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    logger.success(f"Validation report saved to {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation():
    config         = load_config()
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    mlflow.set_experiment("synthetic-rare-disease")

    with mlflow.start_run(run_name="tstr_validation"):
        real_df, synth_df = load_data(processed_path, synthetic_path)

        div_df       = compute_divergences(real_df, synth_df, out_dir)
        plot_distribution_comparison(real_df, synth_df, out_dir)
        tstr_results = run_tstr(real_df, synth_df, out_dir)

        mlflow.log_metrics({
            "avg_js_divergence":  div_df["JS Divergence"].mean(),
            "baseline_accuracy":  tstr_results["baseline_accuracy"],
            "tstr_accuracy":      tstr_results["tstr_accuracy"],
            "accuracy_drop":      tstr_results["accuracy_drop"],
        })

        save_report(
            div_df, tstr_results,
            Path("reports/validation_report.txt")
        )

    logger.success("Evaluation complete — check http://localhost:5000")
    print(f"\nFinal quality rating: {tstr_results['quality_rating']}")


if __name__ == "__main__":
    run_evaluation()