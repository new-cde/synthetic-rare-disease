import sys
import warnings
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.stats import chi2_contingency
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore", category=FutureWarning)

COLS = ["disease", "inheritance", "category", "affected_systems", "prevalence"]


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def cramers_v(x, y):
    tab = pd.crosstab(x, y)
    r, k = tab.shape
    n = tab.values.sum()
    if r < 2 or k < 2 or n < 2:
        return 0.0
    chi2 = chi2_contingency(tab.values, correction=False)[0]
    phi2 = chi2 / n
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rc = r - (r - 1) ** 2 / (n - 1)
    kc = k - (k - 1) ** 2 / (n - 1)
    denom = min(kc - 1, rc - 1)
    return float(np.sqrt(phi2c / denom)) if denom > 0 else 0.0


def matrix(df, func):
    m = pd.DataFrame(0.0, index=COLS, columns=COLS)
    for a in COLS:
        for b in COLS:
            m.loc[a, b] = func(df[a].astype(str), df[b].astype(str))
    return m


def upper_errors(a, b):
    idx = np.triu_indices(len(COLS), k=1)
    d = np.abs(a.values - b.values)[idx]
    return float(d.mean()), float(d.max())


def valid_rate(real, synth, cols):
    key = lambda d: d[cols].astype(str).apply(lambda r: "|".join(r), axis=1)
    return float(key(synth).isin(set(key(real))).mean())


def plot_matrices(v_r, v_s, mi_r, mi_s, tag, path):
    labels = [c.replace("_", " ") for c in COLS]
    vmax_mi = max(mi_r.values.max(), mi_s.values.max())
    rows = [("Cramer's V", v_r, v_s, 1.0),
            ("Mutual information (nats)", mi_r, mi_s, vmax_mi)]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for row, (name, r, s, vmax) in zip(axes, rows):
        panels = [(r, "Real (train)"), (s, "Synthetic"),
                  ((r - s).abs(), "|Real - Synthetic|")]
        for ax, (mat, title) in zip(row, panels):
            im = ax.imshow(mat.values, vmin=0, vmax=vmax, cmap="viridis")
            ax.set_xticks(range(len(COLS)))
            ax.set_yticks(range(len(COLS)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_yticklabels(labels, fontsize=8)
            for i in range(len(COLS)):
                for j in range(len(COLS)):
                    v = mat.values[i, j]
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="black" if v > 0.6 * vmax else "white")
            ax.set_title(f"{name}: {title}", fontsize=10)
            fig.colorbar(im, ax=ax, fraction=0.046)
    plt.suptitle(f"Association matrices: {tag}")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(synth_path=None):
    cfg = load_config()
    synth_path = (Path(synth_path) if synth_path
                  else Path(cfg["data"]["synthetic_path"]) / "synthetic_metadata.csv")
    tag = synth_path.stem
    real_all = pd.read_csv(Path(cfg["data"]["processed_path"]) / "metadata_clean.csv")
    real = real_all[real_all["split"] == "train"]
    synth = pd.read_csv(synth_path)
    logger.info(f"{tag}: {len(synth)} synthetic rows vs {len(real)} real train rows")

    v_r, v_s = matrix(real, cramers_v), matrix(synth, cramers_v)
    mi_r, mi_s = matrix(real, mutual_info_score), matrix(synth, mutual_info_score)
    v_mae, v_max = upper_errors(v_r, v_s)
    mi_mae, mi_max = upper_errors(mi_r, mi_s)

    cols8 = [c for c in synth.columns if c in real.columns and c != "split"]
    row = {
        "File": tag,
        "N synthetic": len(synth),
        "CramersV MAE": round(v_mae, 4),
        "CramersV max err": round(v_max, 4),
        "MI MAE (nats)": round(mi_mae, 4),
        "MI max err (nats)": round(mi_max, 4),
        "Valid combo rate (5 cols)": round(valid_rate(real, synth, COLS), 4),
        f"Valid combo rate ({len(cols8)} cols)": round(valid_rate(real, synth, cols8), 4),
    }

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    plot_matrices(v_r, v_s, mi_r, mi_s, tag,
                  Path("reports/figures") / f"association_{tag}.png")
    pd.concat({"cramers_v_real": v_r, "cramers_v_synth": v_s,
               "mi_real": mi_r, "mi_synth": mi_s}
              ).to_csv(f"reports/joint_matrices_{tag}.csv")

    summary_path = Path("reports/joint_summary.csv")
    new = pd.DataFrame([row])
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        old = old[old["File"] != tag]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(summary_path, index=False)

    print("\n" + "=" * 110)
    print("JOINT-DISTRIBUTION SUMMARY (real = training split)")
    print("=" * 110)
    print(new.to_string(index=False))
    logger.success(f"Saved figure and matrices for {tag}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else None)
