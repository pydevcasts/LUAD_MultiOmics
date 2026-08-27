"""Leakage-safe baseline training utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.luad.evaluation.metrics import compute_classification_metrics
from src.luad.models.factory import create_model
from src.luad.utils.io import ensure_dir


def compute_balanced_sample_weights(y_encoded: Any) -> np.ndarray:
    """
    Compute balanced sample weights for imbalanced classes.
    """
    y_array = np.asarray(y_encoded, dtype=int)

    classes = np.unique(y_array)
    n_samples = len(y_array)
    n_classes = len(classes)

    counts = np.bincount(y_array, minlength=n_classes)
    counts = np.maximum(counts, 1)

    class_weights = n_samples / (n_classes * counts)

    return class_weights[y_array]


def fit_preprocessor(
    X_train: pd.DataFrame,
    y_train: Optional[Any] = None,
    max_missing_ratio: float = 0.2,
    top_variance_features: int = 1000,
    feature_selection_method: str = "anova",
    log1p: bool = False,
) -> Dict[str, Any]:
    """
    Fit leakage-safe preprocessing on training data only.

    Steps:
    1. Missingness filter
    2. Median imputation
    3. Optional log1p transformation
    4. Feature selection:
       - variance-based, or
       - supervised ANOVA F-test inside training fold
    5. Standard scaling
    """
    missing_ratio = X_train.isna().mean()

    kept_columns = missing_ratio[missing_ratio <= max_missing_ratio].index.tolist()

    if len(kept_columns) == 0:
        raise ValueError("No features remained after missingness filtering.")

    X_train_kept = X_train[kept_columns]

    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train_kept)

    log1p_applied = False

    if log1p and X_train_imputed.size > 0:
        min_value = float(np.min(X_train_imputed))

        if min_value >= 0:
            X_train_imputed = np.log1p(X_train_imputed)
            log1p_applied = True

    variances = np.var(X_train_imputed, axis=0)
    valid_positions = np.where(variances > 0)[0]

    if len(valid_positions) == 0:
        raise ValueError("No features with positive variance remained after imputation.")

    selected_positions: np.ndarray

    if feature_selection_method == "anova":
        if y_train is None:
            raise ValueError(
                "ANOVA feature selection requires y_train inside the training fold."
            )

        y_array = np.asarray(y_train)

        if len(y_array) != X_train_imputed.shape[0]:
            raise ValueError(
                "y_train length does not match X_train number of rows during "
                "ANOVA feature selection."
            )

        X_valid = X_train_imputed[:, valid_positions]

        f_scores, _ = f_classif(X_valid, y_array)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)

        if top_variance_features and top_variance_features < len(valid_positions):
            top_local_positions = np.argsort(f_scores)[::-1][:top_variance_features]
            selected_positions = valid_positions[top_local_positions]
        else:
            selected_positions = valid_positions

    else:
        if top_variance_features and top_variance_features < len(valid_positions):
            valid_variances = variances[valid_positions]
            top_local_positions = np.argsort(valid_variances)[::-1][
                :top_variance_features
            ]
            selected_positions = valid_positions[top_local_positions]
        else:
            selected_positions = valid_positions

    selected_positions = np.sort(selected_positions)

    selected_features = [kept_columns[position] for position in selected_positions]

    scaler = StandardScaler()
    scaler.fit(X_train_imputed[:, selected_positions])

    return {
        "kept_columns": kept_columns,
        "imputer": imputer,
        "log1p_applied": log1p_applied,
        "feature_selection_method": feature_selection_method,
        "selected_positions": selected_positions.tolist(),
        "selected_features": selected_features,
        "scaler": scaler,
        "max_missing_ratio": max_missing_ratio,
        "top_features": top_variance_features,
    }


def transform_preprocessor(
    X: pd.DataFrame,
    preprocessor: Dict[str, Any],
) -> np.ndarray:
    """
    Transform data using a fitted preprocessing object.
    """
    kept_columns = preprocessor["kept_columns"]
    selected_positions = np.array(preprocessor["selected_positions"])

    X_kept = X[kept_columns]

    X_imputed = preprocessor["imputer"].transform(X_kept)

    if preprocessor.get("log1p_applied", False):
        X_imputed = np.log1p(np.clip(X_imputed, 0, None))

    X_selected = X_imputed[:, selected_positions]
    X_scaled = preprocessor["scaler"].transform(X_selected)

    return X_scaled


def _aggregate_fold_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate fold metrics into mean/std for scalar metrics.
    """
    aggregated: Dict[str, Any] = {}

    if not metrics_list:
        return aggregated

    scalar_keys = [
        key
        for key in metrics_list[0].keys()
        if key != "per_class"
    ]

    for key in scalar_keys:
        values = []

        for metrics in metrics_list:
            value = metrics.get(key)

            if isinstance(value, (int, float)) and not np.isnan(value):
                values.append(float(value))

        if values:
            aggregated[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return aggregated


def run_baseline_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_names: List[str],
    output_dir: Path,
    cv_folds: int = 5,
    max_missing_ratio: float = 0.2,
    top_variance_features: int = 1000,
    feature_selection_method: str = "anova",
    log1p: bool = False,
    random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Run leakage-safe baseline experiments for multiple models.
    """
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(output_dir / "models")

    y_train = y_train.astype(str)
    y_test = y_test.astype(str)

    label_encoder = LabelEncoder()

    train_classes = sorted(set(y_train.tolist()))
    label_encoder.fit(train_classes)

    missing_test_classes = set(y_test.tolist()) - set(label_encoder.classes_)

    if missing_test_classes:
        raise ValueError(
            "Test set contains classes that were not present in training set: "
            f"{sorted(missing_test_classes)}"
        )

    y_train_enc = pd.Series(
        label_encoder.transform(y_train),
        index=y_train.index,
        name="target_encoded",
    )

    y_test_enc = pd.Series(
        label_encoder.transform(y_test),
        index=y_test.index,
        name="target_encoded",
    )

    original_classes = list(label_encoder.classes_)
    num_classes = len(original_classes)

    results: List[Dict[str, Any]] = []

    for model_name in model_names:
        skf = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )

        fold_metrics: List[Dict[str, Any]] = []

        for fold_index, (train_index, valid_index) in enumerate(
            skf.split(X_train, y_train_enc)
        ):
            X_fold_train = X_train.iloc[train_index]
            y_fold_train_enc = y_train_enc.iloc[train_index]

            X_fold_valid = X_train.iloc[valid_index]
            y_fold_valid_enc = y_train_enc.iloc[valid_index]

            fold_preprocessor = fit_preprocessor(
                X_train=X_fold_train,
                y_train=y_fold_train_enc.values,
                max_missing_ratio=max_missing_ratio,
                top_variance_features=top_variance_features,
                feature_selection_method=feature_selection_method,
                log1p=log1p,
            )

            X_fold_train_processed = transform_preprocessor(
                X_fold_train,
                fold_preprocessor,
            )

            X_fold_valid_processed = transform_preprocessor(
                X_fold_valid,
                fold_preprocessor,
            )

            model = create_model(
                model_name=model_name,
                random_state=random_state,
                num_classes=num_classes,
            )

            fit_params = {}

            if model_name == "xgboost":
                fit_params["sample_weight"] = compute_balanced_sample_weights(
                    y_fold_train_enc.values
                )

            model.fit(X_fold_train_processed, y_fold_train_enc, **fit_params)

            y_valid_pred_enc = model.predict(X_fold_valid_processed)

            y_valid_pred = label_encoder.inverse_transform(y_valid_pred_enc)

            y_fold_valid_original = label_encoder.inverse_transform(
                y_fold_valid_enc.values
            )

            y_valid_proba = None

            if hasattr(model, "predict_proba"):
                y_valid_proba = model.predict_proba(X_fold_valid_processed)

            fold_metric = compute_classification_metrics(
                y_true=y_fold_valid_original,
                y_pred=y_valid_pred,
                y_proba=y_valid_proba,
                labels=original_classes,
            )

            fold_metric["fold"] = fold_index + 1

            fold_metrics.append(fold_metric)

        cv_metrics = _aggregate_fold_metrics(fold_metrics)

        final_preprocessor = fit_preprocessor(
            X_train=X_train,
            y_train=y_train_enc.values,
            max_missing_ratio=max_missing_ratio,
            top_variance_features=top_variance_features,
            feature_selection_method=feature_selection_method,
            log1p=log1p,
        )

        X_train_processed = transform_preprocessor(
            X_train,
            final_preprocessor,
        )

        X_test_processed = transform_preprocessor(
            X_test,
            final_preprocessor,
        )

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

        y_test_proba = None

        if hasattr(final_model, "predict_proba"):
            y_test_proba = final_model.predict_proba(X_test_processed)

        test_metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_test_pred,
            y_proba=y_test_proba,
            labels=original_classes,
        )

        model_artifact_path = models_dir / f"{model_name}.joblib"

        joblib.dump(
            {
                "model_name": model_name,
                "model": final_model,
                "preprocessor": final_preprocessor,
                "label_encoder": label_encoder,
                "classes": original_classes,
                "random_state": random_state,
                "feature_selection_method": feature_selection_method,
                "log1p": log1p,
            },
            model_artifact_path,
        )

        result = {
            "model": model_name,
            "cv_metrics": cv_metrics,
            "fold_metrics": fold_metrics,
            "test_metrics": test_metrics,
            "selected_feature_count": len(final_preprocessor["selected_features"]),
            "feature_selection_method": feature_selection_method,
            "log1p": log1p,
            "model_artifact": str(model_artifact_path),
        }

        results.append(result)

    return results


def save_baseline_results(
    results: List[Dict[str, Any]],
    output_dir: Path,
) -> None:
    """
    Save baseline results as JSON and comparison CSV.
    """
    output_dir = ensure_dir(output_dir)

    results_path = output_dir / "baseline_results.json"

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    comparison_rows = []

    for result in results:
        cv_balanced_accuracy = result["cv_metrics"].get("balanced_accuracy", {})
        cv_f1_macro = result["cv_metrics"].get("f1_macro", {})
        cv_roc_auc = result["cv_metrics"].get("roc_auc", {})

        test_metrics = result["test_metrics"]

        comparison_rows.append(
            {
                "model": result["model"],
                "selected_features": result["selected_feature_count"],
                "feature_selection_method": result.get("feature_selection_method"),
                "log1p": result.get("log1p"),
                "cv_balanced_accuracy_mean": cv_balanced_accuracy.get("mean"),
                "cv_balanced_accuracy_std": cv_balanced_accuracy.get("std"),
                "cv_f1_macro_mean": cv_f1_macro.get("mean"),
                "cv_f1_macro_std": cv_f1_macro.get("std"),
                "cv_roc_auc_mean": cv_roc_auc.get("mean"),
                "cv_roc_auc_std": cv_roc_auc.get("std"),
                "test_accuracy": test_metrics.get("accuracy"),
                "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
                "test_precision_macro": test_metrics.get("precision_macro"),
                "test_recall_macro": test_metrics.get("recall_macro"),
                "test_f1_macro": test_metrics.get("f1_macro"),
                "test_roc_auc": test_metrics.get("roc_auc"),
                "test_mcc": test_metrics.get("mcc"),
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")