"""Train survival prediction models with clinical + molecular late fusion."""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
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
    parser = argparse.ArgumentParser(description="Train survival prediction.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--top-features", type=int, default=500)
    parser.add_argument("--experiment-name", type=str, default="survival_prediction_v2")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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
        name="luad.survival_train",
        log_file=project_root / "artifacts" / "logs" / "survival_train.log",
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

    # ===== Load Modalities =====
    modality_keys = ["mrna_rsem", "mirna", "cna_raw", "mutations", "clinical"]
    modality_matrices = {}

    for modality_key in modality_keys:
        modality_path = data_dir / f"{modality_key}.parquet"
        if not modality_path.exists():
            if modality_key == "clinical":
                modality_path = data_dir / "clinical_features.parquet"
                if not modality_path.exists():
                    logger.warning(f"Clinical features not found: {modality_path}")
                    continue
            else:
                logger.warning(f"Modality {modality_key} not found: {modality_path}")
                continue

        matrix = pd.read_parquet(modality_path)
        if "patient_id" in matrix.columns:
            matrix = matrix.set_index("patient_id")
        matrix.index = matrix.index.astype(str)
        modality_matrices[modality_key] = matrix
        logger.info(f"Loaded {modality_key}: {matrix.shape}")

    # ===== Find Common Patients =====
    common_patients = set(patient_ids)
    for mk, mat in modality_matrices.items():
        common_patients &= set(mat.index)
    common_patients = sorted(common_patients)
    logger.info(f"Common patients across all modalities: {len(common_patients)}")

    y = y[patient_ids.isin(common_patients)].reset_index(drop=True)
    patient_ids = patient_ids[patient_ids.isin(common_patients)].reset_index(drop=True)

    for mk in list(modality_matrices.keys()):
        modality_matrices[mk] = modality_matrices[mk].loc[common_patients]

    # ===== Train/Test Split =====
    idx_train, idx_test = train_test_split(
        np.arange(len(common_patients)),
        test_size=0.20,
        random_state=42,
        stratify=y.values,
    )

    X_train_dict = {}
    X_test_dict = {}

    for mk, mat in modality_matrices.items():
        X_train_dict[mk] = mat.iloc[idx_train]
        X_test_dict[mk] = mat.iloc[idx_test]

    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_test = y.iloc[idx_test].reset_index(drop=True)

    logger.info(f"Train: {len(y_train)}, Test: {len(y_test)}")
    logger.info(f"Train distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test distribution: {y_test.value_counts().to_dict()}")

    # ===== Training =====
    baseline_config = config.get("baseline", {})
    model_names = baseline_config.get(
        "models", ["logistic_regression", "random_forest", "svm", "xgboost"]
    )
    if args.models is not None:
        model_names = [m.strip() for m in args.models.split(",") if m.strip()]

    cv_folds = int(baseline_config.get("cv_folds", 5))
    max_missing_ratio = float(baseline_config.get("max_missing_ratio", 0.2))
    top_features = args.top_features
    random_state = int(baseline_config.get("random_state", 42))

    output_dir = project_root / "artifacts" / "experiments" / args.experiment_name
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(output_dir / "models")

    logger.info(f"Models: {model_names}")
    logger.info(f"Top features: {top_features}")
    logger.info(f"Output: {output_dir}")

    all_results = {}

    # ===== Single Modality Models =====
    for modality_key in modality_keys:
        if modality_key not in X_train_dict:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Training single modality: {modality_key}")
        logger.info(f"{'='*60}")

        X_train_mod = X_train_dict[modality_key]
        X_test_mod = X_test_dict[modality_key]

        use_log1p = modality_key in ["mrna_rsem", "mirna"]
        use_anova = modality_key != "clinical"

        for model_name in model_names:
            logger.info(f"  Training {model_name} on {modality_key}...")

            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            fold_metrics = []

            for fold_idx, (tr_idx, val_idx) in enumerate(skf.split(X_train_mod, y_train)):
                X_tr = X_train_mod.iloc[tr_idx]
                y_tr = y_train.iloc[tr_idx]
                X_val = X_train_mod.iloc[val_idx]
                y_val = y_train.iloc[val_idx]

                if use_anova:
                    preproc = fit_preprocessor(
                        X_train=X_tr, y_train=y_tr.values,
                        max_missing_ratio=max_missing_ratio,
                        top_variance_features=top_features,
                        feature_selection_method="anova",
                        log1p=use_log1p,
                    )
                    X_tr_p = transform_preprocessor(X_tr, preproc)
                    X_val_p = transform_preprocessor(X_val, preproc)
                else:
                    preproc = fit_preprocessor(
                        X_train=X_tr, y_train=y_tr.values,
                        max_missing_ratio=max_missing_ratio,
                        top_variance_features=top_features,
                        feature_selection_method="variance",
                        log1p=False,
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
                fm["fold"] = fold_idx + 1
                fold_metrics.append(fm)

            cv_metrics = _aggregate_fold_metrics(fold_metrics)

            if use_anova:
                final_preproc = fit_preprocessor(
                    X_train=X_train_mod, y_train=y_train.values,
                    max_missing_ratio=max_missing_ratio,
                    top_variance_features=top_features,
                    feature_selection_method="anova",
                    log1p=use_log1p,
                )
            else:
                final_preproc = fit_preprocessor(
                    X_train=X_train_mod, y_train=y_train.values,
                    max_missing_ratio=max_missing_ratio,
                    top_variance_features=top_features,
                    feature_selection_method="variance",
                    log1p=False,
                )

            X_train_p = transform_preprocessor(X_train_mod, final_preproc)
            X_test_p = transform_preprocessor(X_test_mod, final_preproc)

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

            import joblib
            model_path = models_dir / f"{modality_key}_{model_name}.joblib"
            joblib.dump({"model": final_model, "preprocessor": final_preproc}, model_path)

            result_key = f"{modality_key}_{model_name}"
            all_results[result_key] = {
                "modality": modality_key,
                "model": model_name,
                "cv_metrics": cv_metrics,
                "test_metrics": test_metrics,
            }

            cv_ba = cv_metrics.get("balanced_accuracy", {})
            logger.info(
                f"    {result_key}: "
                f"CV BA={cv_ba.get('mean',0):.4f}±{cv_ba.get('std',0):.4f}, "
                f"Test BA={test_metrics['balanced_accuracy']:.4f}, "
                f"Test AUC={test_metrics['roc_auc']}"
            )

    # ===== Late Fusion =====
    logger.info(f"\n{'='*60}")
    logger.info("Training Late Fusion (all modalities including clinical)")
    logger.info(f"{'='*60}")

    log1p_keys = ["mrna_rsem", "mirna"]

    late_result = run_late_fusion(
        X_train_dict=X_train_dict,
        y_train=y_train,
        X_test_dict=X_test_dict,
        y_test=y_test,
        output_dir=output_dir / "late_fusion",
        base_model_name="random_forest",
        log1p_keys=log1p_keys,
        cv_folds=cv_folds,
        max_missing_ratio=max_missing_ratio,
        top_features=top_features,
        feature_selection_method="anova",
        random_state=random_state,
    )

    all_results["late_fusion_random_forest"] = {
        "modality": "all_including_clinical",
        "model": "late_fusion_rf",
        "cv_metrics": late_result["meta_cv_metrics"],
        "test_metrics": late_result["test_metrics"],
    }

    meta_cv_ba = late_result["meta_cv_metrics"].get("balanced_accuracy", {})
    logger.info(
        f"  Late Fusion RF: "
        f"CV BA={meta_cv_ba.get('mean',0):.4f}±{meta_cv_ba.get('std',0):.4f}, "
        f"Test BA={late_result['test_metrics']['balanced_accuracy']:.4f}, "
        f"Test AUC={late_result['test_metrics']['roc_auc']}"
    )

    # ===== Late Fusion without Clinical (for comparison) =====
    logger.info(f"\n{'='*60}")
    logger.info("Training Late Fusion (molecular only, no clinical)")
    logger.info(f"{'='*60}")

    molecular_only_keys = [k for k in modality_keys if k != "clinical" and k in X_train_dict]
    X_train_mol = {k: X_train_dict[k] for k in molecular_only_keys}
    X_test_mol = {k: X_test_dict[k] for k in molecular_only_keys}

    late_result_mol = run_late_fusion(
        X_train_dict=X_train_mol,
        y_train=y_train,
        X_test_dict=X_test_mol,
        y_test=y_test,
        output_dir=output_dir / "late_fusion_molecular_only",
        base_model_name="random_forest",
        log1p_keys=log1p_keys,
        cv_folds=cv_folds,
        max_missing_ratio=max_missing_ratio,
        top_features=top_features,
        feature_selection_method="anova",
        random_state=random_state,
    )

    all_results["late_fusion_molecular_only"] = {
        "modality": "molecular_only_no_clinical",
        "model": "late_fusion_rf",
        "cv_metrics": late_result_mol["meta_cv_metrics"],
        "test_metrics": late_result_mol["test_metrics"],
    }

    meta_cv_ba_mol = late_result_mol["meta_cv_metrics"].get("balanced_accuracy", {})
    logger.info(
        f"  Late Fusion RF (molecular only): "
        f"CV BA={meta_cv_ba_mol.get('mean',0):.4f}±{meta_cv_ba_mol.get('std',0):.4f}, "
        f"Test BA={late_result_mol['test_metrics']['balanced_accuracy']:.4f}, "
        f"Test AUC={late_result_mol['test_metrics']['roc_auc']}"
    )

    # ===== Save Results =====
    results_path = output_dir / "survival_results.json"
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
    print("SURVIVAL PREDICTION RESULTS (with Clinical Features)")
    print("="*70)
    for key, res in all_results.items():
        cv_ba = res["cv_metrics"].get("balanced_accuracy", {})
        tm = res["test_metrics"]
        print(
            f"{key:45s} | "
            f"CV BA={cv_ba.get('mean',0):.4f} | "
            f"Test BA={tm['balanced_accuracy']:.4f} | "
            f"Test AUC={tm['roc_auc']:.4f}"
        )
    print("="*70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()