"""Train tumor vs normal classification models."""

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
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train tumor vs normal classification."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config YAML file.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment overlay.",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing tumor_normal_combined.parquet.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names.",
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=1000,
        help="Number of top features to select.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="tumor_vs_normal",
        help="Experiment name for output directory.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)
    experiment_path = None

    if args.experiment is not None:
        experiment_path = resolve_path(args.experiment)

    try:
        config = load_config(
            config_path=config_path,
            experiment_path=experiment_path,
        )
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))

    logger = get_logger(
        name="luad.tumor_normal_train",
        log_file=project_root / "artifacts" / "logs" / "tumor_normal_train.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    if args.data_dir is not None:
        data_dir = resolve_path(args.data_dir)
    else:
        data_dir = project_root / "data" / "processed" / "tumor_normal"

    combined_path = data_dir / "tumor_normal_combined.parquet"

    if not combined_path.exists():
        logger.error(f"Combined dataset not found: {combined_path}")
        logger.error("Run prepare_tumor_normal.py first.")
        sys.exit(1)

    logger.info(f"Loading combined dataset from: {combined_path}")
    combined_df = pd.read_parquet(combined_path)

    if "sample_id" in combined_df.columns:
        combined_df = combined_df.set_index("sample_id")

    y = combined_df["label"].astype(str)
    X = combined_df.drop(columns=["label"])

    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    label_encoder = LabelEncoder()
    y_encoded = pd.Series(
        label_encoder.fit_transform(y),
        index=y.index,
        name="label_encoded",
    )

    original_classes = list(label_encoder.classes_)
    num_classes = len(original_classes)

    logger.info(f"Classes: {original_classes}")

    X_train, X_test, y_train_enc, y_test_enc = train_test_split(
        X,
        y_encoded,
        test_size=0.20,
        random_state=42,
        stratify=y_encoded,
    )

    logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    split_info = {
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "train_distribution": y_train_enc.value_counts().to_dict(),
        "test_distribution": y_test_enc.value_counts().to_dict(),
        "random_state": 42,
        "stratified": True,
    }

    baseline_config = config.get("baseline", {})

    model_names = baseline_config.get(
        "models",
        ["logistic_regression", "random_forest", "svm", "xgboost"],
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

    logger.info(f"Training models: {model_names}")
    logger.info(f"Top features: {top_features}")
    logger.info(f"Output directory: {output_dir}")

    results = []

    for model_name in model_names:
        logger.info(f"Training {model_name}...")

        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        fold_metrics = []

        for fold_index, (train_idx, valid_idx) in enumerate(
            skf.split(X_train, y_train_enc)
        ):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y_train_enc.iloc[train_idx]

            X_fold_valid = X_train.iloc[valid_idx]
            y_fold_valid = y_train_enc.iloc[valid_idx]

            fold_preprocessor = fit_preprocessor(
                X_train=X_fold_train,
                y_train=y_fold_train.values,
                max_missing_ratio=max_missing_ratio,
                top_variance_features=top_features,
                feature_selection_method="anova",
                log1p=True,
            )

            X_fold_train_processed = transform_preprocessor(
                X_fold_train, fold_preprocessor
            )
            X_fold_valid_processed = transform_preprocessor(
                X_fold_valid, fold_preprocessor
            )

            model = create_model(
                model_name=model_name,
                random_state=random_state,
                num_classes=num_classes,
            )

            fit_params = {}
            if model_name == "xgboost":
                fit_params["sample_weight"] = compute_balanced_sample_weights(
                    y_fold_train.values
                )

            model.fit(X_fold_train_processed, y_fold_train, **fit_params)

            y_valid_pred = model.predict(X_fold_valid_processed)
            y_valid_proba = None
            if hasattr(model, "predict_proba"):
                y_valid_proba = model.predict_proba(X_fold_valid_processed)

            fold_metric = compute_classification_metrics(
                y_true=y_fold_valid.values,
                y_pred=y_valid_pred,
                y_proba=y_valid_proba,
                labels=list(range(num_classes)),
            )
            fold_metric["fold"] = fold_index + 1
            fold_metrics.append(fold_metric)

        cv_metrics = _aggregate_fold_metrics(fold_metrics)

        final_preprocessor = fit_preprocessor(
            X_train=X_train,
            y_train=y_train_enc.values,
            max_missing_ratio=max_missing_ratio,
            top_variance_features=top_features,
            feature_selection_method="anova",
            log1p=True,
        )

        X_train_processed = transform_preprocessor(X_train, final_preprocessor)
        X_test_processed = transform_preprocessor(X_test, final_preprocessor)

        final_model = create_model(
            model_name=model_name,
            random_state=random_state,
            num_classes=num_classes,
        )

        final_fit_params = {}
        if model_name == "xgboost":
            final_fit_params["sample_weight"] = compute_balanced_sample_weights(
                y_train_enc.values
            )

        final_model.fit(X_train_processed, y_train_enc, **final_fit_params)

        y_test_pred_enc = final_model.predict(X_test_processed)
        y_test_pred = label_encoder.inverse_transform(y_test_pred_enc)
        y_test_true = label_encoder.inverse_transform(y_test_enc.values)

        y_test_proba = None
        if hasattr(final_model, "predict_proba"):
            y_test_proba = final_model.predict_proba(X_test_processed)

        test_metrics = compute_classification_metrics(
            y_true=y_test_true,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            labels=original_classes,
        )

        import joblib
        model_path = models_dir / f"{model_name}.joblib"
        joblib.dump(
            {
                "model": final_model,
                "preprocessor": final_preprocessor,
                "label_encoder": label_encoder,
                "model_name": model_name,
            },
            model_path,
        )

        result = {
            "model": model_name,
            "cv_metrics": cv_metrics,
            "test_metrics": test_metrics,
            "selected_features": len(final_preprocessor["selected_features"]),
            "model_path": str(model_path),
        }
        results.append(result)

        cv_ba = cv_metrics.get("balanced_accuracy", {})
        logger.info(
            f"  {model_name}: "
            f"CV BA={cv_ba.get('mean', 0):.4f}±{cv_ba.get('std', 0):.4f}, "
            f"Test BA={test_metrics['balanced_accuracy']:.4f}, "
            f"Test AUC={test_metrics['roc_auc']}"
        )

    results_path = output_dir / "tumor_normal_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    comparison_rows = []
    for r in results:
        cv_ba = r["cv_metrics"].get("balanced_accuracy", {})
        cv_f1 = r["cv_metrics"].get("f1_macro", {})
        cv_auc = r["cv_metrics"].get("roc_auc", {})
        tm = r["test_metrics"]

        comparison_rows.append({
            "model": r["model"],
            "selected_features": r["selected_features"],
            "cv_balanced_acc_mean": cv_ba.get("mean"),
            "cv_balanced_acc_std": cv_ba.get("std"),
            "cv_f1_macro_mean": cv_f1.get("mean"),
            "cv_roc_auc_mean": cv_auc.get("mean"),
            "test_accuracy": tm.get("accuracy"),
            "test_balanced_accuracy": tm.get("balanced_accuracy"),
            "test_precision_macro": tm.get("precision_macro"),
            "test_recall_macro": tm.get("recall_macro"),
            "test_f1_macro": tm.get("f1_macro"),
            "test_roc_auc": tm.get("roc_auc"),
            "test_mcc": tm.get("mcc"),
        })

    comparison_df = pd.DataFrame(comparison_rows)
    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    split_path = output_dir / "split_info.json"
    with split_path.open("w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to: {output_dir}")

    print("\n=== Tumor vs Normal Results ===")
    for r in results:
        cv_ba = r["cv_metrics"].get("balanced_accuracy", {})
        tm = r["test_metrics"]
        print(
            f"- {r['model']}: "
            f"CV BA={cv_ba.get('mean', 0):.4f}±{cv_ba.get('std', 0):.4f} | "
            f"Test BA={tm['balanced_accuracy']:.4f} | "
            f"Test AUC={tm['roc_auc']:.4f} | "
            f"Test F1={tm['f1_macro']:.4f}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()