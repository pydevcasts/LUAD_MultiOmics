"""Late Fusion with PSO for mRNA + miRNA Tumor vs Normal (v3 - fixed label loading)."""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.feature_selection.pso import BinaryPSOFeatureSelector
from src.luad.ml.baseline import fit_preprocessor, transform_preprocessor
from src.luad.models.factory import create_model
from src.luad.evaluation.metrics import compute_classification_metrics
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Late Fusion mRNA+miRNA with PSO.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-mrna-experiment", type=str, default="pso_tumor_normal_v3")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--mirna-top-anova", type=int, default=500)
    parser.add_argument("--mirna-pso-target", type=int, default=80)
    parser.add_argument("--mirna-pso-particles", type=int, default=30)
    parser.add_argument("--mirna-pso-iterations", type=int, default=50)
    parser.add_argument("--max-shap-samples", type=int, default=200)
    parser.add_argument("--experiment-name", type=str, default="late_fusion_mrna_mirna_pso")
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
        print(f"[FAILED] {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    logger = get_logger(
        name="luad.late_fusion_pso",
        log_file=project_root / "artifacts" / "logs" / "late_fusion_pso.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    output_dir = project_root / "artifacts" / "experiments" / args.experiment_name
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    data_dir = resolve_path(args.data_dir) if args.data_dir else project_root / "data" / "processed" / "mrna_mirna_tn"

    # ================================================================
    # STEP 1: Load labels from labels.csv (NOT from parquet)
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading labels and data")
    logger.info("=" * 60)

    labels_path = data_dir / "labels.csv"
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path)
    logger.info(f"Labels loaded: {labels_df.shape}, columns: {list(labels_df.columns)}")

    # Determine label column name
    if "label" in labels_df.columns:
        y_all = labels_df.set_index("sample_id")["label"].astype(int)
    elif "binary_label" in labels_df.columns:
        y_all = labels_df.set_index("sample_id")["binary_label"].astype(int)
    else:
        logger.error(f"No label column found in {labels_path}")
        sys.exit(1)

    logger.info(f"Label distribution: {y_all.value_counts().to_dict()}")

    n_tumor = int((y_all == 1).sum())
    n_normal = int((y_all == 0).sum())

    if n_normal == 0:
        logger.error("CRITICAL: No normal samples in labels!")
        sys.exit(1)

    # Load mRNA features
    mrna_path = data_dir / "mrna.parquet"
    mrna_df = pd.read_parquet(mrna_path)
    if "label" in mrna_df.columns:
        mrna_df = mrna_df.drop(columns=["label"])
    if "sample_id" in mrna_df.columns:
        mrna_df = mrna_df.set_index("sample_id")
    mrna_df.index = mrna_df.index.astype(str)

    # Load miRNA features
    mirna_path = data_dir / "mirna.parquet"
    mirna_df = pd.read_parquet(mirna_path)
    if "label" in mirna_df.columns:
        mirna_df = mirna_df.drop(columns=["label"])
    if "sample_id" in mirna_df.columns:
        mirna_df = mirna_df.set_index("sample_id")
    mirna_df.index = mirna_df.index.astype(str)

    logger.info(f"mRNA shape: {mrna_df.shape}")
    logger.info(f"miRNA shape: {mirna_df.shape}")

    # Align all to common patients from labels
    common_patients = sorted(set(y_all.index) & set(mrna_df.index) & set(mirna_df.index))
    logger.info(f"Common patients (labels ∩ mRNA ∩ miRNA): {len(common_patients)}")

    y = y_all.loc[common_patients].reset_index(drop=True)
    X_mrna = mrna_df.loc[common_patients].reset_index(drop=True)
    X_mirna = mirna_df.loc[common_patients].reset_index(drop=True)

    logger.info(f"Final dataset: {len(common_patients)} patients")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    n_tumor_final = int((y == 1).sum())
    n_normal_final = int((y == 0).sum())

    if n_normal_final == 0:
        logger.error("CRITICAL: No normal samples after alignment!")
        sys.exit(1)

    # ================================================================
    # STEP 2: Train/Test Split
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Train/Test Split")
    logger.info("=" * 60)

    idx_train, idx_test = train_test_split(
        np.arange(len(common_patients)),
        test_size=0.20,
        random_state=42,
        stratify=y.values,
    )

    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    X_mrna_train = X_mrna.iloc[idx_train].reset_index(drop=True)
    X_mrna_test = X_mrna.iloc[idx_test].reset_index(drop=True)
    X_mirna_train = X_mirna.iloc[idx_train].reset_index(drop=True)
    X_mirna_test = X_mirna.iloc[idx_test].reset_index(drop=True)

    logger.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    logger.info(f"Train dist: {y_train.value_counts().to_dict()}")
    logger.info(f"Test dist: {y_test.value_counts().to_dict()}")

    # ================================================================
    # STEP 3: Load mRNA PSO genes
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Loading mRNA PSO genes")
    logger.info("=" * 60)

    pso_mrna_dir = project_root / "artifacts" / "experiments" / args.pso_mrna_experiment
    pso_mrna_genes_path = pso_mrna_dir / "pso_selected_genes.csv"

    if not pso_mrna_genes_path.exists():
        logger.error(f"mRNA PSO genes not found: {pso_mrna_genes_path}")
        sys.exit(1)

    mrna_pso_genes = pd.read_csv(pso_mrna_genes_path)["gene"].tolist()
    logger.info(f"Loaded {len(mrna_pso_genes)} mRNA PSO genes")

    # Align mRNA features to PSO genes
    available_columns = list(X_mrna.columns)
    stripped_col_map = {}
    for col in available_columns:
        name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        if "|" in name:
            name = name.split("|", 1)[0]
        stripped_col_map[name] = col

    matched_cols = []
    matched_names = []
    for gene in mrna_pso_genes:
        if gene in stripped_col_map:
            matched_cols.append(stripped_col_map[gene])
            matched_names.append(gene)

    logger.info(f"mRNA PSO genes matched: {len(matched_cols)}/{len(mrna_pso_genes)}")

    X_mrna_train = X_mrna_train[matched_cols].copy()
    X_mrna_train.columns = matched_names
    X_mrna_test = X_mrna_test[matched_cols].copy()
    X_mrna_test.columns = matched_names

    # ================================================================
    # STEP 4: Preprocess mRNA
    # ================================================================
    logger.info("\nPreprocessing mRNA (log1p + scale)...")
    preproc_mrna = fit_preprocessor(
        X_train=X_mrna_train, y_train=y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=len(matched_names),
        feature_selection_method="variance",
        log1p=True,
    )
    X_mrna_train_p = transform_preprocessor(X_mrna_train, preproc_mrna)
    X_mrna_test_p = transform_preprocessor(X_mrna_test, preproc_mrna)

    # ================================================================
    # STEP 5: ANOVA + PSO on miRNA
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: ANOVA + PSO on miRNA")
    logger.info("=" * 60)

    # Strip miRNA column prefixes
    mirna_col_map = {}
    for col in X_mirna.columns:
        name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        if "|" in name:
            name = name.split("|", 1)[0]
        mirna_col_map[name] = col

    X_mirna_renamed = X_mirna.rename(columns={orig: stripped for stripped, orig in mirna_col_map.items()})
    mirna_feature_names = list(X_mirna_renamed.columns)

    X_mirna_train_r = X_mirna_renamed.iloc[idx_train].reset_index(drop=True)
    X_mirna_test_r = X_mirna_renamed.iloc[idx_test].reset_index(drop=True)

    logger.info(f"Running ANOVA on miRNA (top-{args.mirna_top_anova})...")
    preproc_mirna_anova = fit_preprocessor(
        X_train=X_mirna_train_r, y_train=y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=args.mirna_top_anova,
        feature_selection_method="anova",
        log1p=True,
    )
    X_mirna_train_filtered = transform_preprocessor(X_mirna_train_r, preproc_mirna_anova)
    X_mirna_test_filtered = transform_preprocessor(X_mirna_test_r, preproc_mirna_anova)

    mirna_anova_features = preproc_mirna_anova["selected_features"]
    logger.info(f"After ANOVA: {len(mirna_anova_features)} miRNA features")

    logger.info(f"\nRunning PSO on miRNA (target={args.mirna_pso_target})...")
    pso_mirna = BinaryPSOFeatureSelector(
        n_particles=args.mirna_pso_particles,
        n_iterations=args.mirna_pso_iterations,
        target_n_features=args.mirna_pso_target,
        model_name="random_forest",
        cv_folds=5,
        random_state=42,
        logger=logger,
    )
    pso_mirna.fit(
        X=X_mirna_train_filtered,
        y=y_train.values,
        feature_names=mirna_anova_features,
    )

    mirna_pso_indices = pso_mirna.get_selected_indices()
    mirna_pso_names = pso_mirna.get_selected_feature_names(mirna_anova_features)
    logger.info(f"PSO selected {len(mirna_pso_names)} miRNA features")

    X_mirna_train_pso = X_mirna_train_filtered[:, mirna_pso_indices]
    X_mirna_test_pso = X_mirna_test_filtered[:, mirna_pso_indices]

    # Save miRNA PSO genes
    mirna_pso_path = output_dir / "mirna_pso_selected_genes.csv"
    pd.DataFrame({"gene": mirna_pso_names}).to_csv(mirna_pso_path, index=False, encoding="utf-8-sig")

    # ================================================================
    # STEP 6: Train Modality-Specific Models
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training modality-specific models")
    logger.info("=" * 60)

    # mRNA model
    logger.info("Training mRNA XGBoost...")
    model_mrna = create_model(model_name="xgboost", random_state=42, num_classes=2)
    model_mrna.fit(X_mrna_train_p, y_train)

    mrna_test_pred = model_mrna.predict(X_mrna_test_p)
    mrna_test_proba = model_mrna.predict_proba(X_mrna_test_p)[:, 1]
    mrna_ba = balanced_accuracy_score(y_test, mrna_test_pred)
    mrna_auc = roc_auc_score(y_test, mrna_test_proba)
    logger.info(f"  mRNA alone: Test BA={mrna_ba:.4f}, AUC={mrna_auc:.4f}")

    mrna_train_proba = model_mrna.predict_proba(X_mrna_train_p)[:, 1]

    # miRNA model
    logger.info("Training miRNA XGBoost...")
    model_mirna = create_model(model_name="xgboost", random_state=42, num_classes=2)
    model_mirna.fit(X_mirna_train_pso, y_train)

    mirna_test_pred = model_mirna.predict(X_mirna_test_pso)
    mirna_test_proba = model_mirna.predict_proba(X_mirna_test_pso)[:, 1]
    mirna_ba = balanced_accuracy_score(y_test, mirna_test_pred)
    mirna_auc = roc_auc_score(y_test, mirna_test_proba)
    logger.info(f"  miRNA alone: Test BA={mirna_ba:.4f}, AUC={mirna_auc:.4f}")

    mirna_train_proba = model_mirna.predict_proba(X_mirna_train_pso)[:, 1]

    # ================================================================
    # STEP 7: Meta-Learner (Late Fusion)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Late Fusion Meta-Learner")
    logger.info("=" * 60)

    meta_train = np.column_stack([mrna_train_proba, mirna_train_proba])
    meta_test = np.column_stack([mrna_test_proba, mirna_test_proba])

    meta_model = LogisticRegression(
        max_iter=5000, class_weight="balanced", solver="lbfgs", random_state=42
    )
    meta_model.fit(meta_train, y_train)

    meta_test_pred = meta_model.predict(meta_test)
    meta_test_proba = meta_model.predict_proba(meta_test)[:, 1]

    meta_ba = balanced_accuracy_score(y_test, meta_test_pred)
    meta_auc = roc_auc_score(y_test, meta_test_proba)
    logger.info(f"  Late Fusion: Test BA={meta_ba:.4f}, AUC={meta_auc:.4f}")

    # ================================================================
    # STEP 8: SHAP on Meta-Learner
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 8: SHAP Analysis")
    logger.info("=" * 60)

    mean_abs_shap = [0.0, 0.0]
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        explainer = shap.LinearExplainer(meta_model, meta_train)
        shap_values = explainer.shap_values(meta_test)

        mean_abs_shap = np.mean(np.abs(shap_values), axis=0).tolist()
        modality_names = [f"mRNA ({len(mrna_pso_genes)} genes)", f"miRNA ({len(mirna_pso_names)} genes)"]

        logger.info("SHAP importance per modality:")
        for name, val in zip(modality_names, mean_abs_shap):
            logger.info(f"  {name}: mean|SHAP|={val:.6f}")

        plt.figure(figsize=(8, 5))
        plt.barh(modality_names, mean_abs_shap, color=["#2196F3", "#FF9800"])
        plt.xlabel("Mean |SHAP Value|")
        plt.title("Late Fusion: Modality Importance (SHAP)")
        plt.tight_layout()
        bar_path = figures_dir / "late_fusion_shap_bar.png"
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP bar plot saved: {bar_path}")

        plt.figure(figsize=(8, 5))
        shap.summary_plot(shap_values, meta_test, feature_names=modality_names, show=False)
        summary_path = figures_dir / "late_fusion_shap_summary.png"
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved: {summary_path}")

    except Exception as e:
        logger.warning(f"SHAP failed: {e}")

    # ================================================================
    # STEP 9: Save Results
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 9: Saving Results")
    logger.info("=" * 60)

    results = {
        "experiment": args.experiment_name,
        "task": "Tumor vs Normal",
        "modalities": ["mRNA", "miRNA"],
        "mrna_pso_genes": len(mrna_pso_genes),
        "mirna_pso_genes": len(mirna_pso_names),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_tumor_train": int((y_train == 1).sum()),
        "n_normal_train": int((y_train == 0).sum()),
        "results": {
            "mrna_only": {"test_balanced_accuracy": round(mrna_ba, 4), "test_roc_auc": round(mrna_auc, 4)},
            "mirna_only": {"test_balanced_accuracy": round(mirna_ba, 4), "test_roc_auc": round(mirna_auc, 4)},
            "late_fusion": {"test_balanced_accuracy": round(meta_ba, 4), "test_roc_auc": round(meta_auc, 4)},
        },
        "shap_modality_importance": {
            "mRNA": round(float(mean_abs_shap[0]), 6),
            "miRNA": round(float(mean_abs_shap[1]), 6),
        },
        "mirna_pso_selected": mirna_pso_names,
    }

    with (output_dir / "late_fusion_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    comp_rows = [
        {"model": f"mRNA_only ({len(mrna_pso_genes)} PSO genes)", "test_ba": round(mrna_ba, 4), "test_auc": round(mrna_auc, 4)},
        {"model": f"miRNA_only ({len(mirna_pso_names)} PSO genes)", "test_ba": round(mirna_ba, 4), "test_auc": round(mirna_auc, 4)},
        {"model": "Late Fusion (mRNA+miRNA)", "test_ba": round(meta_ba, 4), "test_auc": round(meta_auc, 4)},
    ]
    pd.DataFrame(comp_rows).to_csv(output_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 70)
    print("LATE FUSION RESULTS: mRNA + miRNA (PSO)")
    print("=" * 70)
    print(f"{'Model':40s} | {'Test BA':>8s} | {'Test AUC':>8s}")
    print("-" * 70)
    for row in comp_rows:
        print(f"{row['model']:40s} | {row['test_ba']:8.4f} | {row['test_auc']:8.4f}")
    print("=" * 70)

    improvement_ba = meta_ba - mrna_ba
    improvement_auc = meta_auc - mrna_auc
    print(f"\nΔ BA  (Late Fusion vs mRNA only): {improvement_ba:+.4f}")
    print(f"Δ AUC (Late Fusion vs mRNA only): {improvement_auc:+.4f}")

    dominant = "mRNA" if mean_abs_shap[0] > mean_abs_shap[1] else "miRNA"
    print(f"\nDominant modality (SHAP): {dominant}")
    print(f"  mRNA SHAP:  {mean_abs_shap[0]:.6f}")
    print(f"  miRNA SHAP: {mean_abs_shap[1]:.6f}")
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()