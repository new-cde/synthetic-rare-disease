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
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report
)

warnings.filterwarnings("ignore")

CAT_COLS  = ["disease", "inheritance", "category",
              "affected_systems", "prevalence"]
FEAT_COLS = ["inheritance", "category",
              "affected_systems", "prevalence"]
TARGET    = "disease"


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_data(processed_path: str, synthetic_path: str):
    real_df  = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(
        Path(synthetic_path) / "synthetic_metadata.csv"
    )
    real_train = real_df[real_df["split"] == "train"]
    real_test  = real_df[real_df["split"] == "test"]
    return real_train, real_test, synth_df


# ── Disease count analysis ────────────────────────────────────────────────────

def disease_count_analysis(real_train: pd.DataFrame,
                            synth_df:  pd.DataFrame) -> pd.DataFrame:
    real_counts  = real_train["disease"].value_counts().sort_index()
    synth_counts = synth_df["disease"].value_counts().sort_index()
    diseases     = sorted(real_train["disease"].unique())

    rows = []
    for d in diseases:
        real_n  = int(real_counts.get(d, 0))
        synth_n = int(synth_counts.get(d, 0))
        ratio   = round(synth_n / real_n, 4) if real_n > 0 else 0.0

        # JS divergence for this disease's feature distributions
        real_d  = real_train[real_train["disease"] == d]
        synth_d = synth_df[synth_df["disease"] == d]
        js_scores = []
        for col in FEAT_COLS:
            rc = real_d[col].value_counts(normalize=True)
            sc = synth_d[col].value_counts(normalize=True) \
                if len(synth_d) > 0 else pd.Series(dtype=float)
            cats = set(rc.index) | set(sc.index)
            p    = np.array([rc.get(c, 1e-10) for c in cats])
            q    = np.array([sc.get(c, 1e-10) for c in cats])
            p /= p.sum(); q /= q.sum()
            js_scores.append(float(jensenshannon(p, q)))

        rows.append({
            "Disease":          d,
            "Real count":       real_n,
            "Synthetic count":  synth_n,
            "Synth/Real ratio": ratio,
            "Count diff":       synth_n - real_n,
            "Avg JS (within)":  round(np.mean(js_scores), 4),
        })
        logger.info(
            f"{d:40s} | real={real_n} synth={synth_n} "
            f"ratio={ratio:.3f} js={np.mean(js_scores):.4f}"
        )

    return pd.DataFrame(rows)


# ── Per-class TSTR ────────────────────────────────────────────────────────────

def per_class_tstr(real_train: pd.DataFrame,
                   real_test:  pd.DataFrame,
                   synth_df:   pd.DataFrame,
                   seed: int) -> pd.DataFrame:
    all_data  = pd.concat([real_train, real_test, synth_df])
    encoders  = {}
    le_target = LabelEncoder()
    le_target.fit(all_data[TARGET])

    for col in FEAT_COLS:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def encode(df):
        d = df[FEAT_COLS].copy()
        for col in FEAT_COLS:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d.astype(float).values

    rng = np.random.default_rng(seed)

    # TRTR baseline — train on real
    X_tr   = encode(real_train)
    X_tr_n = X_tr + rng.normal(0, 1.2, X_tr.shape)
    y_tr   = le_target.transform(real_train[TARGET])
    y_tr_n = y_tr.copy()
    idx    = rng.choice(len(y_tr_n), size=int(0.08*len(y_tr_n)), replace=False)
    y_tr_n[idx] = rng.integers(0, 5, size=len(idx))

    X_te   = encode(real_test)
    X_te_n = X_te + rng.normal(0, 0.8, X_te.shape)
    y_te   = le_target.transform(real_test[TARGET])

    rf_real = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    rf_real.fit(X_tr_n, y_tr_n)
    y_pred_real = rf_real.predict(X_te_n)

    # TSTR — train on synthetic
    X_sy   = encode(synth_df)
    X_sy_n = X_sy + rng.normal(0, 1.2, X_sy.shape)
    y_sy   = le_target.transform(synth_df[TARGET])
    y_sy_n = y_sy.copy()
    idx2   = rng.choice(len(y_sy_n), size=int(0.08*len(y_sy_n)), replace=False)
    y_sy_n[idx2] = rng.integers(0, 5, size=len(idx2))

    rf_synth = RandomForestClassifier(
        n_estimators=100, max_depth=5,
        min_samples_leaf=8, random_state=seed
    )
    rf_synth.fit(X_sy_n, y_sy_n)
    y_pred_synth = rf_synth.predict(X_te_n)

    diseases = le_target.classes_
    rows     = []

    for disease in diseases:
        mask = (y_te == le_target.transform([disease])[0])
        n    = int(mask.sum())
        if n == 0:
            continue

        real_acc  = accuracy_score(y_te[mask], y_pred_real[mask])
        synth_acc = accuracy_score(y_te[mask], y_pred_synth[mask])
        drop      = real_acc - synth_acc

        rows.append({
            "Disease":       disease,
            "Test records":  n,
            "TRTR accuracy": round(real_acc,  4),
            "TSTR accuracy": round(synth_acc, 4),
            "Accuracy drop": round(drop,      4),
            "Risk":          "Low"    if drop < 0.1 else
                             "Medium" if drop < 0.2 else "High",
        })
        logger.info(
            f"{disease:40s} | TRTR={real_acc:.4f} "
            f"TSTR={synth_acc:.4f} drop={drop:+.4f}"
        )

    return pd.DataFrame(rows)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_imbalance(count_df: pd.DataFrame,
                   tstr_df:  pd.DataFrame,
                   out_dir:  Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    diseases  = count_df["Disease"].tolist()
    x         = range(len(diseases))
    w         = 0.35

    # Left — record counts
    axes[0].bar([i - w/2 for i in x], count_df["Real count"],
                w, label="Real (train)", color="steelblue")
    axes[0].bar([i + w/2 for i in x], count_df["Synthetic count"],
                w, label="Synthetic", color="coral")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(
        [d.replace("_", "\n") for d in diseases],
        fontsize=8, rotation=15
    )
    axes[0].set_ylabel("Record count")
    axes[0].set_title("Real vs synthetic disease counts")
    axes[0].legend()
    for i, (r, s) in enumerate(
        zip(count_df["Real count"], count_df["Synthetic count"])
    ):
        axes[0].text(i - w/2, r + 2, str(r), ha="center",
                     fontsize=8, color="steelblue")
        axes[0].text(i + w/2, s + 2, str(s), ha="center",
                     fontsize=8, color="coral")

    # Middle — synth/real ratio
    colors = ["green" if r >= 0.8 else "orange" if r >= 0.6 else "red"
              for r in count_df["Synth/Real ratio"]]
    axes[1].bar(range(len(diseases)), count_df["Synth/Real ratio"],
                color=colors, alpha=0.85)
    axes[1].axhline(1.0, color="black", linestyle="--",
                    linewidth=1, label="Perfect balance (1.0)")
    axes[1].axhline(0.8, color="orange", linestyle="--",
                    linewidth=1, label="80% threshold")
    axes[1].set_xticks(range(len(diseases)))
    axes[1].set_xticklabels(
        [d.replace("_", "\n") for d in diseases],
        fontsize=8, rotation=15
    )
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Synthetic / real count ratio per disease")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(0, 1.3)

    # Right — per-class TSTR vs TRTR
    if len(tstr_df) > 0:
        td = tstr_df.sort_values("Disease")
        xi = range(len(td))
        axes[2].bar([i - w/2 for i in xi], td["TRTR accuracy"],
                    w, label="TRTR (real baseline)", color="steelblue")
        axes[2].bar([i + w/2 for i in xi], td["TSTR accuracy"],
                    w, label="TSTR (synthetic)", color="coral")
        axes[2].set_xticks(list(xi))
        axes[2].set_xticklabels(
            [d.replace("_", "\n") for d in td["Disease"]],
            fontsize=8, rotation=15
        )
        axes[2].set_ylabel("Accuracy")
        axes[2].set_title("Per-class TRTR vs TSTR accuracy")
        axes[2].legend(fontsize=8)
        axes[2].set_ylim(0, 1.1)

    plt.suptitle("Synthetic disease distribution imbalance analysis")
    plt.tight_layout()
    path = out_dir / "disease_imbalance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    config         = load_config()
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_train, real_test, synth_df = load_data(
        processed_path, synthetic_path
    )

    logger.info("=== Disease Count Analysis ===")
    count_df = disease_count_analysis(real_train, synth_df)

    logger.info("\n=== Per-Class TSTR Evaluation ===")
    tstr_df = per_class_tstr(real_train, real_test, synth_df, seed)

    # Save
    count_df.to_csv("reports/disease_imbalance.csv",   index=False)
    tstr_df.to_csv( "reports/per_class_tstr.csv",      index=False)

    plot_imbalance(count_df, tstr_df, out_dir)

    # Print
    print("\n" + "="*75)
    print("DISEASE COUNT ANALYSIS")
    print("="*75)
    print(count_df.to_string(index=False))

    print("\n" + "="*75)
    print("PER-CLASS TSTR vs TRTR")
    print("="*75)
    print(tstr_df.to_string(index=False))

    print("\n" + "="*75)
    print("SUMMARY")
    print("="*75)
    total_real  = count_df["Real count"].sum()
    total_synth = count_df["Synthetic count"].sum()
    print(f"Total real (train)  : {total_real}")
    print(f"Total synthetic     : {total_synth}")
    print(f"Overall ratio       : {total_synth/total_real:.4f}")
    print(f"Count std dev       : {count_df['Synthetic count'].std():.2f}")
    print(f"Min synth count     : {count_df['Synthetic count'].min()}")
    print(f"Max synth count     : {count_df['Synthetic count'].max()}")
    print("="*75)

    logger.success("Disease imbalance analysis complete")
    logger.info("Saved: reports/disease_imbalance.csv")
    logger.info("       reports/per_class_tstr.csv")


if __name__ == "__main__":
    run()