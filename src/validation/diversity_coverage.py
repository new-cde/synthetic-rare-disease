import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.stats import entropy as scipy_entropy


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, synthetic_path: str):
    real_df  = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(Path(synthetic_path) / "synthetic_metadata.csv")
    real_train = real_df[real_df["split"] == "train"]
    return real_train, synth_df


def combination_key(df: pd.DataFrame, cols: list) -> pd.Series:
    return df[cols].astype(str).apply(lambda r: "|".join(r), axis=1)


def diversity_metrics(real_df: pd.DataFrame,
                      synth_df: pd.DataFrame) -> dict:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    n = len(synth_df)

    # Unique combinations
    synth_combos = combination_key(synth_df, cat_cols)
    real_combos  = combination_key(real_df,  cat_cols)

    unique_synth     = synth_combos.nunique()
    unique_real      = real_combos.nunique()
    unique_rate      = round(unique_synth / n, 4)
    exact_dup_rate   = round((n - unique_synth) / n, 4)

    # Coverage — what fraction of real combinations appear in synthetic
    real_combo_set   = set(real_combos)
    synth_combo_set  = set(synth_combos)
    coverage         = round(
        len(real_combo_set & synth_combo_set) / len(real_combo_set), 4
    )

    # Near-duplicate rate (combos appearing more than 10 times)
    synth_counts     = synth_combos.value_counts()
    near_dup_rate    = round((synth_counts > 10).sum() / unique_synth, 4)

    # Per-class coverage
    class_coverage = {}
    for disease in real_df["disease"].unique():
        real_d   = real_df[real_df["disease"] == disease]
        synth_d  = synth_df[synth_df["disease"] == disease]
        if len(synth_d) == 0:
            class_coverage[disease] = 0.0
            continue
        r_combos = set(combination_key(real_d,  cat_cols))
        s_combos = set(combination_key(synth_d, cat_cols))
        class_coverage[disease] = round(
            len(r_combos & s_combos) / len(r_combos), 4
        )

    logger.info(f"Unique synthetic combinations : {unique_synth} / {n} ({unique_rate:.1%})")
    logger.info(f"Unique real combinations      : {unique_real}")
    logger.info(f"Coverage of real combinations : {coverage:.1%}")
    logger.info(f"Exact duplicate rate          : {exact_dup_rate:.1%}")
    logger.info(f"Near-duplicate rate (>10x)    : {near_dup_rate:.1%}")

    return {
        "n_synthetic":            n,
        "unique_synth_combos":    unique_synth,
        "unique_real_combos":     unique_real,
        "unique_rate":            unique_rate,
        "exact_duplicate_rate":   exact_dup_rate,
        "near_duplicate_rate":    near_dup_rate,
        "coverage_of_real":       coverage,
        **{f"coverage_{d}": v for d, v in class_coverage.items()}
    }


def entropy_diversity(real_df: pd.DataFrame,
                      synth_df: pd.DataFrame) -> dict:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    results  = {}
    for col in cat_cols:
        real_probs  = real_df[col].value_counts(normalize=True).values
        synth_probs = synth_df[col].value_counts(normalize=True).values
        real_h  = round(float(scipy_entropy(real_probs)),  4)
        synth_h = round(float(scipy_entropy(synth_probs)), 4)
        results[f"entropy_real_{col}"]  = real_h
        results[f"entropy_synth_{col}"] = synth_h
        results[f"entropy_delta_{col}"] = round(synth_h - real_h, 4)
        logger.info(f"Entropy '{col}': real={real_h:.4f} "
                    f"synth={synth_h:.4f} "
                    f"delta={synth_h - real_h:+.4f}")
    return results


def mode_collapse_check(synth_df: pd.DataFrame) -> dict:
    cat_cols   = ["disease", "inheritance", "category",
                  "affected_systems", "prevalence"]
    combos     = combination_key(synth_df, cat_cols)
    top_counts = combos.value_counts().head(5)
    top_pct    = round(top_counts.sum() / len(synth_df), 4)

    logger.info(f"Top-5 combinations cover {top_pct:.1%} of synthetic records")
    for combo, count in top_counts.items():
        logger.info(f"  {count:4d}x  {combo}")

    return {
        "top5_combo_coverage": top_pct,
        "mode_collapse_risk":  "High"   if top_pct > 0.8 else
                               "Medium" if top_pct > 0.5 else "Low"
    }


def plot_coverage(real_df: pd.DataFrame,
                  synth_df: pd.DataFrame,
                  class_results: dict,
                  out_dir: Path) -> None:
    diseases = real_df["disease"].unique()
    coverage_vals = [class_results.get(f"coverage_{d}", 0) for d in diseases]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-class coverage bar
    axes[0].bar(range(len(diseases)), coverage_vals,
                color="steelblue", alpha=0.8)
    axes[0].set_xticks(range(len(diseases)))
    axes[0].set_xticklabels(
        [d.replace("_", "\n") for d in diseases],
        fontsize=8, rotation=15
    )
    axes[0].set_ylim(0, 1.1)
    axes[0].set_ylabel("Coverage of real combinations")
    axes[0].set_title("Per-disease real combination coverage")
    for i, v in enumerate(coverage_vals):
        axes[0].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

    # Disease count comparison
    real_counts  = real_df["disease"].value_counts().sort_index()
    synth_counts = synth_df["disease"].value_counts().sort_index()
    x = range(len(real_counts))
    w = 0.35
    axes[1].bar([i - w/2 for i in x], real_counts.values,
                w, label="Real", color="steelblue")
    axes[1].bar([i + w/2 for i in x], synth_counts.values,
                w, label="Synthetic", color="coral")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(
        [d.replace("_", "\n") for d in real_counts.index],
        fontsize=8, rotation=15
    )
    axes[1].set_ylabel("Record count")
    axes[1].set_title("Disease distribution — real vs synthetic")
    axes[1].legend()

    plt.suptitle("Diversity and coverage analysis")
    plt.tight_layout()
    path = out_dir / "diversity_coverage.png"
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

    logger.info("=== Diversity and Coverage Analysis ===")
    div = diversity_metrics(real_df, synth_df)

    logger.info("\n=== Entropy Diversity ===")
    ent = entropy_diversity(real_df, synth_df)

    logger.info("\n=== Mode Collapse Check ===")
    mode = mode_collapse_check(synth_df)

    # Combine all results
    all_results = {**div, **ent, **mode}
    pd.DataFrame([all_results]).to_csv(
        "reports/diversity_coverage.csv", index=False
    )

    plot_coverage(real_df, synth_df, div, out_dir)

    print("\n" + "="*65)
    print("DIVERSITY AND COVERAGE SUMMARY")
    print("="*65)
    print(f"Unique combinations (synth) : {div['unique_synth_combos']} "
          f"/ {div['n_synthetic']} ({div['unique_rate']:.1%})")
    print(f"Unique combinations (real)  : {div['unique_real_combos']}")
    print(f"Coverage of real combos     : {div['coverage_of_real']:.1%}")
    print(f"Exact duplicate rate        : {div['exact_duplicate_rate']:.1%}")
    print(f"Near-duplicate rate (>10x)  : {div['near_duplicate_rate']:.1%}")
    print(f"Top-5 combo coverage        : {mode['top5_combo_coverage']:.1%}")
    print(f"Mode collapse risk          : {mode['mode_collapse_risk']}")
    print(f"\nPer-disease coverage:")
    for disease in real_df["disease"].unique():
        key = f"coverage_{disease}"
        print(f"  {disease:40s}: {div.get(key, 0):.1%}")
    print(f"\nEntropy (real vs synthetic):")
    cat_cols = ["disease","inheritance","category",
                "affected_systems","prevalence"]
    for col in cat_cols:
        print(f"  {col:20s}: real={ent[f'entropy_real_{col}']:.4f} "
              f"synth={ent[f'entropy_synth_{col}']:.4f} "
              f"delta={ent[f'entropy_delta_{col}']:+.4f}")
    print("="*65)

    logger.success("Diversity and coverage analysis complete")
    logger.info("Saved: reports/diversity_coverage.csv")


if __name__ == "__main__":
    run()