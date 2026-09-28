import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from loguru import logger
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def encode_df(real_df: pd.DataFrame,
              synth_df: pd.DataFrame) -> tuple:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    all_data = pd.concat([real_df, synth_df])
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def enc(df):
        d = df[cat_cols].copy()
        for col in cat_cols:
            d[col] = encoders[col].transform(d[col].astype(str))
        return d.astype(float).values

    return enc(real_df), enc(synth_df)


# ── 1. Nearest-neighbour disclosure risk ──────────────────────────────────────

def nn_disclosure_risk(real_train: np.ndarray,
                       real_test:  np.ndarray,
                       synth:      np.ndarray) -> dict:
    """
    Compare distances from synthetic records to training records
    vs distances from synthetic records to held-out test records.
    If synthetic records are closer to training than to test,
    that is evidence of memorisation.
    """
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(real_train)
    dist_train, _ = nn.kneighbors(synth)

    nn2 = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn2.fit(real_test)
    dist_test, _  = nn2.kneighbors(synth)

    avg_train = float(np.mean(dist_train))
    avg_test  = float(np.mean(dist_test))
    ratio     = round(avg_train / avg_test, 4) if avg_test > 0 else 0.0

    # Exact matches (distance = 0) to training records
    exact_train = int(np.sum(dist_train == 0))
    exact_test  = int(np.sum(dist_test  == 0))

    # Near-duplicates (distance < 0.5)
    near_train = int(np.sum(dist_train < 0.5))
    near_test  = int(np.sum(dist_test  < 0.5))

    logger.info(f"NN avg distance to train : {avg_train:.4f}")
    logger.info(f"NN avg distance to test  : {avg_test:.4f}")
    logger.info(f"Train/test distance ratio: {ratio} "
                f"(< 1.0 = closer to train = memorisation risk)")
    logger.info(f"Exact matches — train: {exact_train} | test: {exact_test}")
    logger.info(f"Near-dupes (<0.5) — train: {near_train} | test: {near_test}")

    return {
        "avg_nn_dist_train":    round(avg_train, 4),
        "avg_nn_dist_test":     round(avg_test,  4),
        "train_test_ratio":     ratio,
        "exact_matches_train":  exact_train,
        "exact_matches_test":   exact_test,
        "near_dupes_train":     near_train,
        "near_dupes_test":      near_test,
        "memorisation_risk":    "Low" if ratio >= 0.9 else
                                "Medium" if ratio >= 0.7 else "High"
    }


# ── 2. Membership inference attack ───────────────────────────────────────────

def membership_inference_attack(real_train: np.ndarray,
                                real_test:  np.ndarray,
                                synth:      np.ndarray,
                                seed: int) -> dict:
    """
    Shadow-model membership inference:
    Train a classifier to distinguish real training records
    from synthetic records. AUC near 0.5 = safe (indistinguishable).
    AUC near 1.0 = synthetic records reveal training data membership.
    """
    n = min(len(real_train), len(synth))
    X = np.vstack([real_train[:n], synth[:n]])
    y = np.array([1] * n + [0] * n)

    clf = RandomForestClassifier(
        n_estimators=100, max_depth=5, random_state=seed
    )
    aucs = cross_val_score(clf, X, y, cv=5,
                           scoring="roc_auc", n_jobs=-1)
    mean_auc = round(float(aucs.mean()), 4)
    std_auc  = round(float(aucs.std()),  4)

    logger.info(f"Membership inference AUC: {mean_auc} ± {std_auc}")
    logger.info(f"  (0.5 = indistinguishable = safe, 1.0 = full leakage)")

    return {
        "membership_inference_auc":     mean_auc,
        "membership_inference_auc_std": std_auc,
        "membership_inference_risk":    "Low"    if mean_auc < 0.6 else
                                        "Medium" if mean_auc < 0.75 else "High"
    }


# ── 3. Attribute inference risk ───────────────────────────────────────────────

def attribute_inference_risk(real_train_df: pd.DataFrame,
                              synth_df:     pd.DataFrame,
                              seed: int) -> dict:
    """
    For each feature, test whether knowing other features
    allows predicting it better from synthetic than from real data.
    High predictability from synthetic = attribute leakage risk.
    """
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]
    results  = {}

    all_data = pd.concat([real_train_df, synth_df])
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(all_data[col].astype(str))
        encoders[col] = le

    def encode_col(df, cols):
        d = df[cols].copy()
        for c in cols:
            d[c] = encoders[c].transform(d[c].astype(str))
        return d.astype(float).values

    for target_col in cat_cols:
        other_cols = [c for c in cat_cols if c != target_col]
        X_real = encode_col(real_train_df, other_cols)
        y_real = encoders[target_col].transform(
            real_train_df[target_col].astype(str)
        )
        X_synth = encode_col(synth_df, other_cols)
        y_synth  = encoders[target_col].transform(
            synth_df[target_col].astype(str)
        )

        clf = RandomForestClassifier(
            n_estimators=50, max_depth=5, random_state=seed
        )
        real_aucs  = cross_val_score(clf, X_real,  y_real,
                                     cv=3, scoring="accuracy")
        synth_aucs = cross_val_score(clf, X_synth, y_synth,
                                     cv=3, scoring="accuracy")
        results[target_col] = {
            "real_acc":  round(float(real_aucs.mean()),  4),
            "synth_acc": round(float(synth_aucs.mean()), 4),
            "delta":     round(float(synth_aucs.mean() -
                               real_aucs.mean()), 4),
        }
        logger.info(f"Attribute inference '{target_col}': "
                    f"real={real_aucs.mean():.3f} "
                    f"synth={synth_aucs.mean():.3f} "
                    f"delta={synth_aucs.mean()-real_aucs.mean():.3f}")

    return results


# ── 4. Exact and near-duplicate analysis ──────────────────────────────────────

def duplicate_analysis(real_train_df: pd.DataFrame,
                       synth_df:      pd.DataFrame) -> dict:
    cat_cols = ["disease", "inheritance", "category",
                "affected_systems", "prevalence"]

    def key(df):
        return df[cat_cols].astype(str).apply(
            lambda r: "|".join(r), axis=1
        )

    real_keys  = set(key(real_train_df))
    synth_keys = key(synth_df)

    exact_match_rate = round(synth_keys.isin(real_keys).mean(), 4)
    synth_unique     = round(synth_keys.nunique() / len(synth_df), 4)
    exact_dup_rate   = round(
        (len(synth_df) - synth_keys.nunique()) / len(synth_df), 4
    )

    logger.info(f"Synthetic exact match rate (vs real train): {exact_match_rate}")
    logger.info(f"Synthetic unique record rate:               {synth_unique}")
    logger.info(f"Synthetic exact duplicate rate:             {exact_dup_rate}")

    return {
        "exact_match_rate_vs_real": exact_match_rate,
        "synth_unique_record_rate": synth_unique,
        "synth_exact_duplicate_rate": exact_dup_rate,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    config         = load_config()
    seed           = config["project"]["seed"]
    processed_path = config["data"]["processed_path"]
    synthetic_path = config["data"]["synthetic_path"]
    out_dir        = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    real_df  = pd.read_csv(Path(processed_path) / "metadata_clean.csv")
    synth_df = pd.read_csv(
        Path(synthetic_path) / "synthetic_metadata.csv"
    )

    real_train = real_df[real_df["split"] == "train"]
    real_test  = real_df[real_df["split"] == "test"]

    logger.info("Encoding features...")
    X_train, X_synth = encode_df(real_train, synth_df)
    X_test,  _       = encode_df(real_test,  synth_df)

    # ── Run all analyses
    logger.info("\n=== 1. Nearest-Neighbour Disclosure Risk ===")
    nn_results = nn_disclosure_risk(X_train, X_test, X_synth)

    logger.info("\n=== 2. Membership Inference Attack ===")
    mi_results = membership_inference_attack(X_train, X_test, X_synth, seed)

    logger.info("\n=== 3. Attribute Inference Risk ===")
    attr_results = attribute_inference_risk(real_train, synth_df, seed)

    logger.info("\n=== 4. Duplicate Analysis ===")
    dup_results = duplicate_analysis(real_train, synth_df)

    # ── Summary table
    summary = {**nn_results, **mi_results, **dup_results}
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv("reports/privacy_extended.csv", index=False)

    # ── Attribute inference table
    attr_df = pd.DataFrame(attr_results).T
    attr_df.index.name = "Feature"
    attr_df.to_csv("reports/attribute_inference.csv")

    # ── Print results
    print("\n" + "="*70)
    print("PRIVACY EVALUATION SUMMARY")
    print("="*70)
    print(f"1. NN Disclosure Risk")
    print(f"   Avg NN dist to train : {nn_results['avg_nn_dist_train']}")
    print(f"   Avg NN dist to test  : {nn_results['avg_nn_dist_test']}")
    print(f"   Train/test ratio     : {nn_results['train_test_ratio']}"
          f"  (< 1.0 = memorisation risk)")
    print(f"   Exact matches train  : {nn_results['exact_matches_train']}")
    print(f"   Near-dupes train     : {nn_results['near_dupes_train']}")
    print(f"   Memorisation risk    : {nn_results['memorisation_risk']}")
    print(f"\n2. Membership Inference Attack")
    print(f"   AUC: {mi_results['membership_inference_auc']} "
          f"± {mi_results['membership_inference_auc_std']}")
    print(f"   Risk: {mi_results['membership_inference_risk']}"
          f"  (0.5=safe, 1.0=full leakage)")
    print(f"\n3. Attribute Inference Risk")
    print(attr_df.to_string())
    print(f"\n4. Duplicate Analysis")
    print(f"   Exact match vs real  : {dup_results['exact_match_rate_vs_real']}")
    print(f"   Unique record rate   : {dup_results['synth_unique_record_rate']}")
    print(f"   Exact duplicate rate : {dup_results['synth_exact_duplicate_rate']}")
    print("="*70)

    logger.success("Privacy evaluation complete")
    logger.info("Results saved to reports/privacy_extended.csv")
    logger.info("           and reports/attribute_inference.csv")


if __name__ == "__main__":
    run()