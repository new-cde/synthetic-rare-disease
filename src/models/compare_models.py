import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger


def run():
    reports_dir = Path("reports")
    out_dir     = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read individual metrics files
    metric_files = [
        "metrics_random_forest.csv",
        "metrics_svm.csv",
        "metrics_gradient_boosting.csv",
        "metrics_logistic_regression.csv",
    ]

    frames = []
    for fname in metric_files:
        fpath = reports_dir / fname
        if fpath.exists():
            frames.append(pd.read_csv(fpath))
            logger.info(f"Loaded: {fname}")
        else:
            logger.warning(f"Missing: {fname} — run the individual training script first")

    if not frames:
        logger.error("No metrics files found. Run all 4 training scripts first.")
        return

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={
        "model":          "Model",
        "val_accuracy":   "Val Accuracy",
        "val_f1":         "Val F1",
        "test_accuracy":  "Test Accuracy",
        "test_f1":        "Test F1",
    })
    df = df.sort_values("Test Accuracy", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON TABLE")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # Save CSV — same format as before
    csv_path = reports_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.success(f"Comparison table saved to {csv_path}")

    fig, ax = plt.subplots(figsize=(12, 6))
    x      = range(len(df))
    width  = 0.35
    bars1  = ax.bar(
        [i - width/2 for i in x], df["Val Accuracy"],
        width, label="Val Accuracy",
        color="#2196F3", edgecolor="white", linewidth=0.8
    )
    bars2  = ax.bar(
        [i + width/2 for i in x], df["Test Accuracy"],
        width, label="Test Accuracy",
        color="#FF5722", edgecolor="white", linewidth=0.8
    )
    for bar in list(bars1) + list(bars2):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold", color="#1a1a1a"
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        df["Model"], rotation=15, ha="right",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("Model", fontsize=14,
                  fontweight="bold", labelpad=12)
    ax.set_ylabel("Accuracy", fontsize=14,
                  fontweight="bold", labelpad=12)
    ax.set_title(
        "Baseline model comparison — validation vs test accuracy",
        fontsize=15, fontweight="bold", pad=15
    )
    ax.legend(fontsize=13, frameon=True)
    ax.tick_params(axis="y", labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    chart_path = out_dir / "model_comparison.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Chart saved to {chart_path}")

    best = df.iloc[0]["Model"]
    logger.success(f"Best model by test accuracy: {best}")


if __name__ == "__main__":
    run()