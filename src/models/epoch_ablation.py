import sys
import random
import warnings
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from scipy.spatial.distance import jensenshannon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.validation.joint_distribution import cramers_v, matrix, upper_errors, valid_rate, COLS

warnings.filterwarnings("ignore")

EPOCHS    = [100, 300, 600, 1000]
GEN_SEEDS = [42, 123, 456]
N_SYNTH   = 1000
FEATS     = ["inheritance", "category", "affected_systems", "prevalence"]
TARGET    = "disease"


def load_config(path="config/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def marginal_js(real, synth):
    out = []
    for col in COLS:
        rc = real[col].value_counts(normalize=True)
        sc = synth[col].value_counts(normalize=True)
        cats = set(rc.index) | set(sc.index)
        p = np.array([rc.get(c, 1e-10) for c in cats])
        q = np.array([sc.get(c, 1e-10) for c in cats])
        p = p / p.sum()
        q = q / q.sum()
        out.append(float(jensenshannon(p, q)))
    return float(np.mean(out))


def encode(train, test, synth):
    enc = {c: LabelEncoder().fit(pd.concat([train[c], test[c], synth[c]]).astype(str))
           for c in FEATS}
    le_t = LabelEncoder().fit(train[TARGET])

    def prep(df):
        X = np.column_stack([enc[c].transform(df[c].astype(str))
                             for c in FEATS]).astype(float)
        return X, le_t.transform(df[TARGET])

    return prep(train), prep(test), prep(synth)


def noise_on(Xa, ya, Xb, seed):
    rng = np.random.default_rng(seed)
    Xa2 = Xa + rng.normal(0, 1.2, Xa.shape)
    ya2 = ya.copy()
    n = int(0.08 * len(ya2))
    idx = rng.choice(len(ya2), size=n, replace=False)
    ya2[idx] = rng.integers(0, 5, size=n)
    Xb2 = Xb + rng.normal(0, 0.8, Xb.shape)
    return Xa2, ya2, Xb2


def legacy(Xa, ya, Xb, seed):
    rng = np.random.default_rng(seed)
    Xa2 = Xa + rng.normal(0, 1.2, Xa.shape)
    ya2 = ya.copy()
    flip = rng.random(len(ya2)) < 0.08
    ya2[flip] = rng.integers(0, 5, flip.sum())
    Xb2 = Xb + rng.normal(0, 1.2, Xb.shape)
    return Xa2, ya2, Xb2


def svm_acc(Xa, ya, Xb, yb, seed):
    m = SVC(kernel="rbf", C=0.5, gamma="scale", random_state=seed)
    m.fit(Xa, ya)
    return accuracy_score(yb, m.predict(Xb))


def rf_acc(Xa, ya, Xb, yb, seed):
    m = RandomForestClassifier(n_estimators=100, max_depth=5,
                               min_samples_leaf=8, random_state=seed)
    m.fit(Xa, ya)
    return accuracy_score(yb, m.predict(Xb))


def evaluate_arm(Xa, ya, Xb, yb, seed):
    off = svm_acc(Xa, ya, Xb, yb, seed)
    A, B, C = noise_on(Xa, ya, Xb, seed)
    on = svm_acc(A, B, C, yb, seed)
    A, B, C = legacy(Xa, ya, Xb, seed)
    leg = rf_acc(A, B, C, yb, seed)
    return off, on, leg


def run():
    cfg = load_config()
    real = pd.read_csv(Path(cfg["data"]["processed_path"]) / "metadata_clean.csv")
    train = real[real["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test = real[real["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    cols8 = list(train.columns)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(train)
    v_real = matrix(train, cramers_v)

    base_rows, rows = [], []
    for seed in GEN_SEEDS:
        (Xtr, ytr), (Xte, yte), _ = encode(train, test, train)
        off, on, leg = evaluate_arm(Xtr, ytr, Xte, yte, seed)
        base_rows.append({"Seed": seed, "TRTR_off": off, "TRTR_on": on, "TRTR_legacy": leg})

        for e in EPOCHS:
            set_seed(seed)
            model = CTGANSynthesizer(metadata, epochs=e, verbose=False, cuda=False)
            model.fit(train)
            set_seed(seed)
            synth = model.sample(num_rows=N_SYNTH)

            v_mae, _ = upper_errors(v_real, matrix(synth, cramers_v))
            (_, _), (_, _), (Xs, ys) = encode(train, test, synth)
            t_off, t_on, t_leg = evaluate_arm(Xs, ys, Xte, yte, seed)
            row = {"Epochs": e, "Seed": seed,
                   "JS": marginal_js(train, synth),
                   "V_MAE": v_mae,
                   "Valid8": valid_rate(train, synth, cols8),
                   "TSTR_off": t_off, "TSTR_on": t_on, "TSTR_legacy": t_leg}
            rows.append(row)
            logger.info(f"epochs={e} seed={seed} JS={row['JS']:.4f} "
                        f"V_MAE={v_mae:.4f} valid={row['Valid8']:.3f} "
                        f"TSTR off/on/legacy={t_off:.3f}/{t_on:.3f}/{t_leg:.3f}")

    raw = pd.DataFrame(rows)
    raw.to_csv("reports/epoch_ablation_raw.csv", index=False)
    base = pd.DataFrame(base_rows)

    names = {"JS": "Avg JS", "V_MAE": "CramersV MAE", "Valid8": "Valid combo (8 cols)",
             "TSTR_off": "TSTR SVM noise-off", "TSTR_on": "TSTR SVM noise-on",
             "TSTR_legacy": "TSTR RF legacy"}
    g = raw.groupby("Epochs")
    table = pd.DataFrame({n: g[c].apply(lambda s: f"{s.mean():.3f}+-{s.std():.3f}")
                          for c, n in names.items()})
    table.to_csv("reports/epoch_ablation.csv")

    print("\n" + "=" * 100)
    print(f"CTGAN EPOCH ABLATION (mean+-SD over {len(GEN_SEEDS)} generator seeds)")
    print("=" * 100)
    print(table.to_string())
    print("\nReal-data baseline (TRTR), same protocols:")
    print("  noise-off {:.3f}+-{:.3f} | noise-on {:.3f}+-{:.3f} | legacy {:.3f}+-{:.3f}".format(
        base.TRTR_off.mean(), base.TRTR_off.std(),
        base.TRTR_on.mean(), base.TRTR_on.std(),
        base.TRTR_legacy.mean(), base.TRTR_legacy.std()))
    print("=" * 100)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, (c, title) in zip(axes[:3], [("JS", "Avg JS divergence"),
                                         ("V_MAE", "Cramer's V MAE"),
                                         ("Valid8", "Valid full-record rate")]):
        m, s = g[c].mean(), g[c].std()
        ax.errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3)
        ax.set_title(title)
        ax.set_xlabel("Epochs")
    for c, lab in [("TSTR_off", "SVM noise-off"), ("TSTR_on", "SVM noise-on"),
                   ("TSTR_legacy", "RF legacy")]:
        m, s = g[c].mean(), g[c].std()
        axes[3].errorbar(m.index, m.values, yerr=s.values, marker="o", capsize=3, label=lab)
    axes[3].set_title("TSTR accuracy")
    axes[3].set_xlabel("Epochs")
    axes[3].set_ylim(0, 1.05)
    axes[3].legend(fontsize=8)
    plt.suptitle("CTGAN epoch ablation")
    plt.tight_layout()
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("reports/figures/epoch_ablation.png", dpi=150)
    plt.close(fig)
    logger.success("Saved reports/epoch_ablation.csv and figure")


if __name__ == "__main__":
    run()
