"""Late Fusion with PSO for mRNA + miRNA + Clinical data (Fixed v2 - reads processed clinical)."""

from pathlib import Path
import argparse
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.ml.baseline import fit_preprocessor, transform_preprocessor
from src.luad.models.factory import create_model
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Late Fusion mRNA+miRNA+Clinical")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-mrna-experiment", type=str, default="pso_tumor_normal_v3")
    parser.add_argument("--mirna-pso-target", type=int, default=80)
    parser.add_argument("--mirna-pso-particles", type=int, default=30)
    parser.add_argument("--mirna-pso-iterations", type=int, default=50)
    parser.add_argument("--experiment-name", type=str, default="late_fusion_3modality_final")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_path = resolve_path(args.experiment) if args.experiment else None

    try:
        config = load_config(config_path=config_path, experiment_path=experiment_path)
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    logger = get_logger(
        name="luad.late_fusion_3mod",
        log_file=project_root / "artifacts" / "logs" / "late_fusion_3mod.log",
        level="INFO", console=True,
    )

    output_dir = ensure_dir(project_root / "artifacts" / "experiments" / args.experiment_name)

    # ================================================================
    # STEP 1: Loading Multi-Modal Data
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Multi-Modal Data")
    logger.info("=" * 60)

    tn_dir = project_root / "data" / "processed" / "mrna_mirna_tn"

    # Load Labels
    labels_df = pd.read_csv(tn_dir / "labels.csv").set_index("sample_id")

    # Load mRNA & miRNA
    mrna_df = pd.read_parquet(tn_dir / "mrna.parquet")
    mirna_df = pd.read_parquet(tn_dir / "mirna.parquet")

    # Drop label column if exists in feature matrices
    if "label" in mrna_df.columns:
        mrna_df = mrna_df.drop(columns=["label"])
    if "label" in mirna_df.columns:
        mirna_df = mirna_df.drop(columns=["label"])

    # FIXED v2: Load PROCESSED clinical data instead of raw file
    clin_processed_path = project_root / "data" / "processed" / "clinical_tn" / "clinical_processed.parquet"
    has_clinical = False
    clin_df = pd.DataFrame()

    if clin_processed_path.exists():
        clin_raw = pd.read_parquet(clin_processed_path)
        # Drop label column from clinical features
        if "label" in clin_raw.columns:
            clin_features = clin_raw.drop(columns=["label"])
        else:
            clin_features = clin_raw.copy()

        if not clin_features.empty:
            # Align patients across all modalities
            common_patients_all = sorted(
                set(mrna_df.index) & set(mirna_df.index) & set(clin_features.index)
            )
            if len(common_patients_all) > 0:
                clin_df = clin_features.loc[common_patients_all]
                has_clinical = True
                logger.info(
                    f"Clinical features loaded from processed file: "
                    f"{clin_df.shape[1]} features, {clin_df.shape[0]} patients"
                )
                logger.info(f"Clinical columns: {list(clin_df.columns)}")
            else:
                logger.warning("No common patients between clinical and molecular data.")
        else:
            logger.warning("Processed clinical file is empty.")
    else:
        logger.warning(
            f"Processed clinical file not found at {clin_processed_path}. "
            f"Run prepare_clinical_features.py first."
        )

    # Determine common patients
    if has_clinical:
        common_patients = sorted(
            set(mrna_df.index) & set(mirna_df.index) & set(clin_df.index)
        )
    else:
        common_patients = sorted(set(mrna_df.index) & set(mirna_df.index))

    logger.info(f"Common patients across modalities: {len(common_patients)}")

    # Subset data to common patients
    X_mrna_full = mrna_df.loc[common_patients]
    X_mirna_full = mirna_df.loc[common_patients]
    y_full = labels_df.loc[common_patients, "label"]

    if has_clinical:
        X_clin_full = clin_df.loc[common_patients]
        logger.info(
            f"Data shapes -> mRNA: {X_mrna_full.shape}, "
            f"miRNA: {X_mirna_full.shape}, Clinical: {X_clin_full.shape}"
        )
    else:
        logger.info(
            f"Data shapes -> mRNA: {X_mrna_full.shape}, "
            f"miRNA: {X_mirna_full.shape} (No Clinical)"
        )

    # ================================================================
    # Train/Test Split
    # ================================================================
    idx_train, idx_test = train_test_split(
        np.arange(len(common_patients)),
        test_size=0.2, random_state=42, stratify=y_full
    )

    y_train = y_full.iloc[idx_train].reset_index(drop=True)
    y_test = y_full.iloc[idx_test].reset_index(drop=True)

    X_mrna_tr = X_mrna_full.iloc[idx_train].reset_index(drop=True)
    X_mrna_te = X_mrna_full.iloc[idx_test].reset_index(drop=True)
    X_mirna_tr = X_mirna_full.iloc[idx_train].reset_index(drop=True)
    X_mirna_te = X_mirna_full.iloc[idx_test].reset_index(drop=True)

    if has_clinical:
        X_clin_tr = X_clin_full.iloc[idx_train].reset_index(drop=True)
        X_clin_te = X_clin_full.iloc[idx_test].reset_index(drop=True)

    # ================================================================
    # STEP 2: Feature Selection & Preprocessing
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 2: Feature Selection & Preprocessing")
    logger.info("=" * 60)

    # Load PSO genes for mRNA
    pso_genes_path = (
        project_root / "artifacts" / "experiments"
        / args.pso_mrna_experiment / "pso_selected_genes.csv"
    )
    pso_genes = pd.read_csv(pso_genes_path)["gene"].tolist()

    # Align mRNA columns to PSO genes
    avail_cols = {str(c): c for c in X_mrna_tr.columns}
    matched_mrna_cols = [avail_cols[g] for g in pso_genes if g in avail_cols]

    if len(matched_mrna_cols) == 0:
        logger.error("No PSO genes matched in mRNA data!")
        sys.exit(1)

    X_mrna_tr = X_mrna_tr[matched_mrna_cols]
    X_mrna_te = X_mrna_te[matched_mrna_cols]
    logger.info(f"mRNA aligned to {len(matched_mrna_cols)} PSO genes")

    # Preprocess mRNA
    prep_mrna = fit_preprocessor(
        X_mrna_tr, y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=len(matched_mrna_cols),
        feature_selection_method="variance",
        log1p=True,
    )
    X_mrna_tr_p = transform_preprocessor(X_mrna_tr, prep_mrna)
    X_mrna_te_p = transform_preprocessor(X_mrna_te, prep_mrna)

    # Preprocess miRNA (ANOVA top-k)
    prep_mirna = fit_preprocessor(
        X_mirna_tr, y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=args.mirna_pso_target,
        feature_selection_method="anova",
        log1p=True,
    )
    X_mirna_tr_p = transform_preprocessor(X_mirna_tr, prep_mirna)
    X_mirna_te_p = transform_preprocessor(X_mirna_te, prep_mirna)
    logger.info(f"miRNA reduced to {X_mirna_tr_p.shape[1]} features")

    # Preprocess Clinical (Scale only)
    scaler_clin = None
    X_clin_tr_p = None
    X_clin_te_p = None
    if has_clinical:
        scaler_clin = StandardScaler()
        X_clin_tr_p = scaler_clin.fit_transform(X_clin_tr.fillna(0))
        X_clin_te_p = scaler_clin.transform(X_clin_te.fillna(0))
        logger.info(f"Clinical scaled: {X_clin_tr_p.shape[1]} features")

    # ================================================================
    # STEP 3: Train Base Models
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 3: Training Base Models")
    logger.info("=" * 60)

    # mRNA Model
    model_mrna = create_model("xgboost", random_state=42, num_classes=2)
    model_mrna.fit(X_mrna_tr_p, y_train)
    prob_mrna_tr = model_mrna.predict_proba(X_mrna_tr_p)[:, 1]
    prob_mrna_te = model_mrna.predict_proba(X_mrna_te_p)[:, 1]
    ba_mrna = balanced_accuracy_score(y_test, model_mrna.predict(X_mrna_te_p))
    logger.info(f"mRNA Model BA: {ba_mrna:.4f}")

    # miRNA Model
    model_mirna = create_model("xgboost", random_state=42, num_classes=2)
    model_mirna.fit(X_mirna_tr_p, y_train)
    prob_mirna_tr = model_mirna.predict_proba(X_mirna_tr_p)[:, 1]
    prob_mirna_te = model_mirna.predict_proba(X_mirna_te_p)[:, 1]
    ba_mirna = balanced_accuracy_score(y_test, model_mirna.predict(X_mirna_te_p))
    logger.info(f"miRNA Model BA: {ba_mirna:.4f}")

    # Clinical Model (if available)
    prob_clin_tr = None
    prob_clin_te = None
    ba_clin = None
    if has_clinical and X_clin_tr_p is not None:
        model_clin = LogisticRegression(
            max_iter=5000, class_weight="balanced",
            solver="lbfgs", random_state=42
        )
        model_clin.fit(X_clin_tr_p, y_train)
        prob_clin_tr = model_clin.predict_proba(X_clin_tr_p)[:, 1]
        prob_clin_te = model_clin.predict_proba(X_clin_te_p)[:, 1]
        ba_clin = balanced_accuracy_score(y_test, model_clin.predict(X_clin_te_p))
        logger.info(f"Clinical Model BA: {ba_clin:.4f}")

    # ================================================================
    # STEP 4: Late Fusion Meta-Learner
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 4: Late Fusion Meta-Learner")
    logger.info("=" * 60)

    if has_clinical and prob_clin_tr is not None:
        meta_train = np.column_stack([prob_mrna_tr, prob_mirna_tr, prob_clin_tr])
        meta_test = np.column_stack([prob_mrna_te, prob_mirna_te, prob_clin_te])
        fusion_name = "Late Fusion (mRNA+miRNA+Clinical)"
    else:
        meta_train = np.column_stack([prob_mrna_tr, prob_mirna_tr])
        meta_test = np.column_stack([prob_mrna_te, prob_mirna_te])
        fusion_name = "Late Fusion (mRNA+miRNA)"
        logger.warning("Clinical data missing. Running 2-modality fusion instead.")

    meta_model = LogisticRegression(
        max_iter=5000, class_weight="balanced",
        solver="lbfgs", random_state=42
    )
    meta_model.fit(meta_train, y_train)

    meta_pred = meta_model.predict(meta_test)
    meta_prob = meta_model.predict_proba(meta_test)[:, 1]

    ba_fusion = balanced_accuracy_score(y_test, meta_pred)
    auc_fusion = roc_auc_score(y_test, meta_prob)

    logger.info(f"{fusion_name}: BA={ba_fusion:.4f}, AUC={auc_fusion:.4f}")

    # ================================================================
    # Save Results
    # ================================================================
    results = {
        "experiment": args.experiment_name,
        "modalities": ["mRNA", "miRNA"] + (["Clinical"] if has_clinical else []),
        "n_patients": len(common_patients),
        "results": {
            "mrna_only": {"test_ba": round(ba_mrna, 4)},
            "mirna_only": {"test_ba": round(ba_mirna, 4)},
            fusion_name: {
                "test_ba": round(ba_fusion, 4),
                "test_auc": round(auc_fusion, 4),
            },
        },
    }
    if has_clinical and ba_clin is not None:
        results["results"]["clinical_only"] = {"test_ba": round(ba_clin, 4)}

    with (output_dir / "late_fusion_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print(f"RESULTS: {fusion_name}")
    print("=" * 60)
    print(f"mRNA Only BA:      {ba_mrna:.4f}")
    print(f"miRNA Only BA:     {ba_mirna:.4f}")
    if has_clinical and ba_clin is not None:
        print(f"Clinical Only BA:  {ba_clin:.4f}")
    print(f"{fusion_name} BA: {ba_fusion:.4f} | AUC: {auc_fusion:.4f}")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()