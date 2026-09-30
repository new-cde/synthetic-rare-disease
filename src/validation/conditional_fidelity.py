import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, synthetic_path: str):
    real_df  = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(Path(synthetic_path) / "synthetic_metadata.csv")
    real_train = real_df[real_df["split"] == "train"]
    return real_train, synth_df


def js_divergence(real_col: pd.Series,
                  synth_col: pd.Series) -> float:
    all_cats = set(real_col.unique()) | set(synth_col.unique())
    real_p   = real_col.value_counts(normalize=True)
    synth_p  = synth_col.value_counts(normalize=True)
    p = np.array([real_p.get(c, 1e-10)  for c in all_cats])
    q = np.array([synth_p.get(c, 1e-10) for c in all_cats])
    p /= p.sum(); q /= q.sum()
    return round(float(jensenshannon(p, q)), 4)


def disease_fidelity(real_df: pd.DataFrame,
                     synth_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    diseases     = sorted(real_df["disease"].unique())
    rows         = []

    for disease in diseases:
        real_d  = real_df[real_df["disease"] == disease]
        synth_d = synth_df[synth_df["disease"] == disease]

        if len(synth_d) == 0:
            logger.warning(f"No synthetic records for {disease}")
            continue

        row = {
            "Disease":          disease,
            "Real records":     len(real_d),
            "Synth records":    len(synth_d),
        }

        js_scores = []
        for col in feature_cols:
            js = js_divergence(real_d[col], synth_d[col])
            row[f"JS_{col}"] = js
            js_scores.append(js)

        row["Avg JS"] = round(np.mean(js_scores), 4)
        row["Max JS"] = round(np.max(js_scores),  4)
        row["Quality"] = ("Good"       if row["Avg JS"] < 0.1  else
                          "Acceptable" if row["Avg JS"] < 0.2  else "Poor")

        rows.append(row)
        logger.info(
            f"{disease:40s} | n_real={len(real_d):4d} "
            f"n_synth={len(synth_d):4d} | "
            f"Avg JS={row['Avg JS']:.4f} | {row['Quality']}"
        )

    return pd.DataFrame(rows)


def overall_fidelity(real_df: pd.DataFrame,
                     synth_df: pd.DataFrame) -> dict:
    feature_cols = ["inheritance", "category",
                    "affected_systems", "prevalence"]
    js_scores    = []
    for col in feature_cols:
        js = js_divergence(real_df[col], synth_df[col])
        js_scores.append(js)
    return {
        "overall_avg_js": round(np.mean(js_scores), 4),
        "overall_max_js": round(np.max(js_scores),  4),
    }


def plot_conditional_fidelity(df: pd.DataFrame,
                               out_dir: Path) -> None:
    feature_cols = ["JS_inheritance", "JS_category",
                    "JS_affected_systems", "JS_prevalence"]
    feature_labels = ["Inheritance", "Category",
                      "Affected systems", "Prevalence"]
    diseases = df["Disease"].tolist()
    x        = range(len(diseases))
    width    = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left — per-feature JS per disease
    colors = ["steelblue", "coral", "teal", "gold"]
    for i, (col, label, color) in enumerate(
        zip(feature_cols, feature_labels, colors)
    ):
        axes[0].bar(
            [xi + i * width for xi in x],
            df[col].values,
            width, label=label, color=color, alpha=0.85
        )
    axes[0].axhline(0.1, color="green", linestyle="--",
                    linewidth=1, label="Good threshold (JS=0.1)")
    axes[0].set_xticks([xi + 1.5 * width for xi in x])
    axes[0].set_xticklabels(
        [d.replace("_", "\n") for d in diseases],
        fontsize=8, rotation=15
    )
    axes[0].set_ylabel("JS Divergence")
    axes[0].set_title("Per-feature JS divergence by disease")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0, 0.35)

    # Right — avg JS per disease
    colors_avg = ["steelblue" if v < 0.1 else
                  "orange"    if v < 0.2 else
                  "red"       for v in df["Avg JS"]]
    axes[1].bar(range(len(diseases)), df["Avg JS"].values,
                color=colors_avg, alpha=0.85)
    axes[1].axhline(0.1, color="green", linestyle="--",
                    linewidth=1, label="Good (0.1)")
    axes[1].axhline(0.2, color="red", linestyle="--",
                    linewidth=1, label="Acceptable (0.2)")
    axes[1].set_xticks(range(len(diseases)))
    axes[1].set_xticklabels(
        [d.replace("_", "\n") for d in diseases],
        fontsize=8, rotation=15
    )
    axes[1].set_ylabel("Avg JS Divergence")
    axes[1].set_title("Average JS divergence per disease")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 0.35)
    for i, v in enumerate(df["Avg JS"].values):
        axes[1].text(i, v + 0.005, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=8)

    plt.suptitle("Disease-specific conditional fidelity — real vs synthetic")
    plt.tight_layout()
    path = out_dir / "conditional_fidelity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


def run():
    config         = load_config()
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df, synth_df = load_data(processed_path, synthetic_path)

    logger.info("=== Disease-Specific Conditional Fidelity ===")
    df_results = disease_fidelity(real_df, synth_df)

    logger.info("\n=== Overall Fidelity ===")
    overall = overall_fidelity(real_df, synth_df)
    logger.info(f"Overall avg JS : {overall['overall_avg_js']}")
    logger.info(f"Overall max JS : {overall['overall_max_js']}")

    # Save CSV
    csv_path = Path("reports/conditional_fidelity.csv")
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved: {csv_path}")

    plot_conditional_fidelity(df_results, out_dir)

    # Print table
    display_cols = ["Disease", "Real records", "Synth records",
                    "JS_inheritance", "JS_category",
                    "JS_affected_systems", "JS_prevalence",
                    "Avg JS", "Quality"]
    print("\n" + "="*100)
    print("DISEASE-SPECIFIC CONDITIONAL FIDELITY")
    print("="*100)
    print(df_results[display_cols].to_string(index=False))
    print("-"*100)
    print(f"Overall avg JS divergence : {overall['overall_avg_js']}")
    print(f"Overall max JS divergence : {overall['overall_max_js']}")
    print("="*100)

    logger.success("Conditional fidelity analysis complete")
    logger.info(f"Results saved to {csv_path}")


if __name__ == "__main__":
    run()