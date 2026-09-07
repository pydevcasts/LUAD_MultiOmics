"""Train survival prediction using pathway-based feature selection."""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.feature_selection.pathway_genes import (
    PATHWAY_GENE_SETS,
    get_all_pathway_genes,
    get_gene_to_pathways,
    get_pathway_gene_counts,
)
from src.luad.ml.baseline import (
    fit_preprocessor,
    transform_preprocessor,
    compute_balanced_sample_weights,
    _aggregate_fold_metrics,
)
from src.luad.models.factory import create_model
from src.luad.evaluation.metrics import compute_classification_metrics
from src.luad.fusion.late import run_late_fusion
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Train pathway-based survival prediction.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--top-features", type=int, default=200)
    parser.add_argument("--experiment-name", type=str, default="survival_pathway")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def extract_pathway_features(mrna_matrix, pathway_genes, logger):
    """Extract only pathway genes from mRNA matrix."""
    mrna_columns = list(mrna_matrix.columns)

    stripped_col_map = {}
    for col in mrna_columns:
        name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        if "|" in name:
            name = name.split("|", 1)[0]
        stripped_col_map[name] = col

    matched_features = {}
    unmatched_genes = []

    for gene in pathway_genes:
        if gene in stripped_col_map:
            matched_features[gene] = stripped_col_map[gene]
        else:
            unmatched_genes.append(gene)

    logger.info(f"Pathway genes requested: {len(pathway_genes)}")
    logger.info(f"Pathway genes found in data: {len(matched_features)}")
    logger.info(f"Pathway genes NOT found: {len(unmatched_genes)}")

    if len(unmatched_genes) <= 20:
        logger.info(f"  Missing genes: {unmatched_genes}")

    if len(matched_features) == 0:
        raise ValueError("No pathway genes found in mRNA matrix!")

    original_cols = list(matched_features.values())
    pathway_matrix = mrna_matrix[original_cols].copy()

    rename_map = {orig: gene for gene, orig in matched_features.items()}
    pathway_matrix = pathway_matrix.rename(columns=rename_map)

    return pathway_matrix, matched_features, unmatched_genes


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_path = None
    if args.experiment is not None:
        experiment_path = resolve_path(args.experiment)

    try:
        config = load_config(config_path=config_path, experiment_path=experiment_path)
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))

    logger = get_logger(
        name="luad.pathway_survival",
        log_file=project_root / "artifacts" / "logs" / "pathway_survival.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    if args.data_dir is not None:
        data_dir = resolve_path(args.data_dir)
    else:
        data_dir = project_root / "data" / "processed" / "survival"

    # ===== Load Labels =====
    survival_labels_path = data_dir / "survival_labels.csv"
    if not survival_labels_path.exists():
        logger.error(f"Survival labels not found: {survival_labels_path}")
        sys.exit(1)

    survival_df = pd.read_csv(survival_labels_path)
    y = survival_df["survival_label"].astype(int)
    patient_ids = survival_df["patient_id"].astype(str)

    logger.info(f"Survival dataset: {len(patient_ids)} patients")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    # ===== Load mRNA =====
    mrna_path = data_dir / "mrna_rsem.parquet"
    if not mrna_path.exists():
        logger.error(f"mRNA matrix not found: {mrna_path}")
        sys.exit(1)

    mrna_matrix = pd.read_parquet(mrna_path)
    if "patient_id" in mrna_matrix.columns:
        mrna_matrix = mrna_matrix.set_index("patient_id")
    mrna_matrix.index = mrna_matrix.index.astype(str)

    logger.info(f"mRNA matrix: {mrna_matrix.shape}")

    # ===== Load Clinical =====
    clinical_path = data_dir / "clinical_features.parquet"
    clinical_matrix = None
    if clinical_path.exists():
        clinical_matrix = pd.read_parquet(clinical_path)
        if "patient_id" in clinical_matrix.columns:
            clinical_matrix = clinical_matrix.set_index("patient_id")
        clinical_matrix.index = clinical_matrix.index.astype(str)
        logger.info(f"Clinical features: {clinical_matrix.shape}")

    # ===== Print Pathway Info =====
    pathway_counts = get_pathway_gene_counts()
    logger.info("\n=== Pathway Gene Counts ===")
    for pw, count in pathway_counts.items():
        logger.info(f"  {pw}: {count} genes")

    all_pathway_genes = get_all_pathway_genes()
    logger.info(f"\nTotal unique pathway genes: {len(all_pathway_genes)}")

    # ===== Extract Pathway Features =====
    pathway_matrix, matched, unmatched = extract_pathway_features(
        mrna_matrix, all_pathway_genes, logger
    )

    logger.info(f"Pathway matrix shape: {pathway_matrix.shape}")

    # ===== Per-pathway matrices =====
    pathway_matrices = {}
    for pw_name, pw_genes in PATHWAY_GENE_SETS.items():
        pw_matched = {g: matched[g] for g in pw_genes if g in matched}
        if len(pw_matched) > 0:
            orig_cols = list(pw_matched.values())
            pw_mat = mrna_matrix[orig_cols].copy()
            rename_map = {orig: gene for gene, orig in pw_matched.items()}
            pw_mat = pw_mat.rename(columns=rename_map)
            pathway_matrices[pw_name] = pw_mat
            logger.info(f"  {pw_name}: {len(pw_matched)} genes matched")

    # ===== Find Common Patients =====
    common_patients = set(patient_ids) & set(pathway_matrix.index)
    if clinical_matrix is not None:
        common_patients &= set(clinical_matrix.index)
    common_patients = sorted(common_patients)
    logger.info(f"Common patients: {len(common_patients)}")

    y = y[patient_ids.isin(common_patients)].reset_index(drop=True)
    patient_ids = patient_ids[patient_ids.isin(common_patients)].reset_index(drop=True)
    pathway_matrix = pathway_matrix.loc[common_patients]

    if clinical_matrix is not None:
        clinical_matrix = clinical_matrix.loc[common_patients]

    # ===== Train/Test Split =====
    idx_train, idx_test = train_test_split(
        np.arange(len(common_patients)),
        test_size=0.20,
        random_state=42,
        stratify=y.values,
    )

    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    logger.info(f"Train: {len(y_train)}, Test: {len(y_test)}")

    # ===== Training =====
    baseline_config = config.get("baseline", {})
    model_names = baseline_config.get(
        "models", ["logistic_regression", "random_forest", "svm", "xgboost"]
    )
    cv_folds = int(baseline_config.get("cv_folds", 5))
    max_missing_ratio = float(baseline_config.get("max_missing_ratio", 0.2))
    random_state = int(baseline_config.get("random_state", 42))

    output_dir = project_root / "artifacts" / "experiments" / args.experiment_name
    output_dir = ensure_dir(output_dir)

    all_results = {}

    # ===== 1. All Pathway Genes Combined =====
    logger.info(f"\n{'='*60}")
    logger.info("Training with ALL pathway genes combined")
    logger.info(f"{'='*60}")

    X_train_pw = pathway_matrix.iloc[idx_train]
    X_test_pw = pathway_matrix.iloc[idx_test]

    for model_name in model_names:
        logger.info(f"  Training {model_name} on pathway genes...")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fold_metrics = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_pw, y_train)):
            X_tr = X_train_pw.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_val = X_train_pw.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            preproc = fit_preprocessor(
                X_train=X_tr, y_train=y_tr.values,
                max_missing_ratio=max_missing_ratio,
                top_variance_features=min(args.top_features, X_tr.shape[1]),
                feature_selection_method="anova",
                log1p=True,
            )
            X_tr_p = transform_preprocessor(X_tr, preproc)
            X_val_p = transform_preprocessor(X_val, preproc)

            model = create_model(model_name=model_name, random_state=random_state, num_classes=2)
            fit_params = {}
            if model_name == "xgboost":
                fit_params["sample_weight"] = compute_balanced_sample_weights(y_tr.values)
            model.fit(X_tr_p, y_tr, **fit_params)

            y_pred = model.predict(X_val_p)
            y_proba = model.predict_proba(X_val_p) if hasattr(model, "predict_proba") else None

            fm = compute_classification_metrics(
                y_true=y_val.values, y_pred=y_pred, y_proba=y_proba, labels=[0, 1]
            )
            fold_metrics.append(fm)

        cv_metrics = _aggregate_fold_metrics(fold_metrics)

        final_preproc = fit_preprocessor(
            X_train=X_train_pw, y_train=y_train.values,
            max_missing_ratio=max_missing_ratio,
            top_variance_features=min(args.top_features, X_train_pw.shape[1]),
            feature_selection_method="anova",
            log1p=True,
        )
        X_train_p = transform_preprocessor(X_train_pw, final_preproc)
        X_test_p = transform_preprocessor(X_test_pw, final_preproc)

        final_model = create_model(model_name=model_name, random_state=random_state, num_classes=2)
        fit_params = {}
        if model_name == "xgboost":
            fit_params["sample_weight"] = compute_balanced_sample_weights(y_train.values)
        final_model.fit(X_train_p, y_train, **fit_params)

        y_test_pred = final_model.predict(X_test_p)
        y_test_proba = final_model.predict_proba(X_test_p) if hasattr(final_model, "predict_proba") else None

        test_metrics = compute_classification_metrics(
            y_true=y_test.values, y_pred=y_test_pred, y_proba=y_test_proba, labels=[0, 1]
        )

        result_key = f"all_pathways_{model_name}"
        all_results[result_key] = {
            "modality": "all_pathway_genes",
            "model": model_name,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }

        cv_ba = cv_metrics.get("balanced_accuracy", {})
        logger.info(
            f"    {result_key}: "
            f"CV BA={cv_ba.get('mean',0):.4f}, "
            f"Test BA={test_metrics['balanced_accuracy']:.4f}, "
            f"Test AUC={test_metrics['roc_auc']:.4f}"
        )

    # ===== 2. Individual Pathway Models =====
    logger.info(f"\n{'='*60}")
    logger.info("Training individual pathway models")
    logger.info(f"{'='*60}")

    for pw_name, pw_mat in pathway_matrices.items():
        pw_mat_common = pw_mat.loc[common_patients]
        X_train_ind = pw_mat_common.iloc[idx_train]
        X_test_ind = pw_mat_common.iloc[idx_test]

        best_model = "random_forest"
        logger.info(f"  Training {best_model} on {pw_name} ({X_train_ind.shape[1]} genes)...")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        fold_metrics = []

        for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_ind, y_train)):
            X_tr = X_train_ind.iloc[tr_idx]
            y_tr = y_train.iloc[tr_idx]
            X_val = X_train_ind.iloc[val_idx]
            y_val = y_train.iloc[val_idx]

            top_k = min(args.top_features, X_tr.shape[1])
            preproc = fit_preprocessor(
                X_train=X_tr, y_train=y_tr.values,
                max_missing_ratio=max_missing_ratio,
                top_variance_features=top_k,
                feature_selection_method="anova",
                log1p=True,
            )
            X_tr_p = transform_preprocessor(X_tr, preproc)
            X_val_p = transform_preprocessor(X_val, preproc)

            model = create_model(model_name=best_model, random_state=random_state, num_classes=2)
            model.fit(X_tr_p, y_tr)

            y_pred = model.predict(X_val_p)
            y_proba = model.predict_proba(X_val_p) if hasattr(model, "predict_proba") else None

            fm = compute_classification_metrics(
                y_true=y_val.values, y_pred=y_pred, y_proba=y_proba, labels=[0, 1]
            )
            fold_metrics.append(fm)

        cv_metrics = _aggregate_fold_metrics(fold_metrics)

        top_k = min(args.top_features, X_train_ind.shape[1])
        final_preproc = fit_preprocessor(
            X_train=X_train_ind, y_train=y_train.values,
            max_missing_ratio=max_missing_ratio,
            top_variance_features=top_k,
            feature_selection_method="anova",
            log1p=True,
        )
        X_train_p = transform_preprocessor(X_train_ind, final_preproc)
        X_test_p = transform_preprocessor(X_test_ind, final_preproc)

        final_model = create_model(model_name=best_model, random_state=random_state, num_classes=2)
        final_model.fit(X_train_p, y_train)

        y_test_pred = final_model.predict(X_test_p)
        y_test_proba = final_model.predict_proba(X_test_p) if hasattr(final_model, "predict_proba") else None

        test_metrics = compute_classification_metrics(
            y_true=y_test.values, y_pred=y_test_pred, y_proba=y_test_proba, labels=[0, 1]
        )

        result_key = f"{pw_name}_{best_model}"
        all_results[result_key] = {
            "modality": pw_name,
            "model": best_model,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
        }

        cv_ba = cv_metrics.get("balanced_accuracy", {})
        logger.info(
            f"    {result_key}: "
            f"CV BA={cv_ba.get('mean',0):.4f}, "
            f"Test BA={test_metrics['balanced_accuracy']:.4f}, "
            f"Test AUC={test_metrics['roc_auc']:.4f}"
        )

    # ===== 3. Pathway + Clinical =====
    if clinical_matrix is not None:
        logger.info(f"\n{'='*60}")
        logger.info("Training Pathway + Clinical (combined)")
        logger.info(f"{'='*60}")

        X_train_combined = pd.concat(
            [pathway_matrix.iloc[idx_train], clinical_matrix.iloc[idx_train]],
            axis=1,
        )
        X_test_combined = pd.concat(
            [pathway_matrix.iloc[idx_test], clinical_matrix.iloc[idx_test]],
            axis=1,
        )

        for model_name in model_names:
            logger.info(f"  Training {model_name} on pathway + clinical...")

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            fold_metrics = []

            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_combined, y_train)):
                X_tr = X_train_combined.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_val = X_train_combined.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                preproc = fit_preprocessor(
                    X_train=X_tr, y_train=y_tr.values,
                    max_missing_ratio=max_missing_ratio,
                    top_variance_features=min(args.top_features, X_tr.shape[1]),
                    feature_selection_method="anova",
                    log1p=True,
                )
                X_tr_p = transform_preprocessor(X_tr, preproc)
                X_val_p = transform_preprocessor(X_val, preproc)

                model = create_model(model_name=model_name, random_state=random_state, num_classes=2)
                fit_params = {}
                if model_name == "xgboost":
                    fit_params["sample_weight"] = compute_balanced_sample_weights(y_tr.values)
                model.fit(X_tr_p, y_tr, **fit_params)

                y_pred = model.predict(X_val_p)
                y_proba = model.predict_proba(X_val_p) if hasattr(model, "predict_proba") else None

                fm = compute_classification_metrics(
                    y_true=y_val.values, y_pred=y_pred, y_proba=y_proba, labels=[0, 1]
                )
                fold_metrics.append(fm)

            cv_metrics = _aggregate_fold_metrics(fold_metrics)

            final_preproc = fit_preprocessor(
                X_train=X_train_combined, y_train=y_train.values,
                max_missing_ratio=max_missing_ratio,
                top_variance_features=min(args.top_features, X_train_combined.shape[1]),
                feature_selection_method="anova",
                log1p=True,
            )
            X_train_p = transform_preprocessor(X_train_combined, final_preproc)
            X_test_p = transform_preprocessor(X_test_combined, final_preproc)

            final_model = create_model(model_name=model_name, random_state=random_state, num_classes=2)
            fit_params = {}
            if model_name == "xgboost":
                fit_params["sample_weight"] = compute_balanced_sample_weights(y_train.values)
            final_model.fit(X_train_p, y_train, **fit_params)

            y_test_pred = final_model.predict(X_test_p)
            y_test_proba = final_model.predict_proba(X_test_p) if hasattr(final_model, "predict_proba") else None

            test_metrics = compute_classification_metrics(
                y_true=y_test.values, y_pred=y_test_pred, y_proba=y_test_proba, labels=[0, 1]
            )

            result_key = f"pathway_clinical_{model_name}"
            all_results[result_key] = {
                "modality": "pathway + clinical",
                "model": model_name,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
            }

            cv_ba = cv_metrics.get("balanced_accuracy", {})
            logger.info(
                f"    {result_key}: "
                f"CV BA={cv_ba.get('mean',0):.4f}, "
                f"Test BA={test_metrics['balanced_accuracy']:.4f}, "
                f"Test AUC={test_metrics['roc_auc']:.4f}"
            )

    # ===== Save Results =====
    results_path = output_dir / "pathway_survival_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)

    comparison_rows = []
    for key, res in all_results.items():
        cv_ba = res["cv_metrics"].get("balanced_accuracy", {})
        tm = res["test_metrics"]
        comparison_rows.append({
            "experiment": key,
            "modality": res["modality"],
            "model": res["model"],
            "cv_balanced_acc_mean": cv_ba.get("mean"),
            "cv_balanced_acc_std": cv_ba.get("std"),
            "test_accuracy": tm.get("accuracy"),
            "test_balanced_accuracy": tm.get("balanced_accuracy"),
            "test_f1_macro": tm.get("f1_macro"),
            "test_roc_auc": tm.get("roc_auc"),
            "test_mcc": tm.get("mcc"),
        })

    comp_df = pd.DataFrame(comparison_rows)
    comp_path = output_dir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

    logger.info(f"\nResults saved to: {output_dir}")

    print("\n" + "="*70)
    print("PATHWAY-BASED SURVIVAL PREDICTION RESULTS")
    print("="*70)
    for key, res in all_results.items():
        cv_ba = res["cv_metrics"].get("balanced_accuracy", {})
        tm = res["test_metrics"]
        print(
            f"{key:50s} | "
            f"CV BA={cv_ba.get('mean',0):.4f} | "
            f"Test BA={tm['balanced_accuracy']:.4f} | "
            f"Test AUC={tm['roc_auc']:.4f}"
        )
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()