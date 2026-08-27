"""Late fusion utilities for multi-omics classification.

This module implements leakage-safe late fusion:
- Each modality trains a base model independently.
- Out-of-fold probabilities are generated for training data.
- A meta learner is trained on base-model probabilities.
- The locked test set is evaluated only once.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from src.luad.evaluation.metrics import compute_classification_metrics
from src.luad.ml.baseline import (
    _aggregate_fold_metrics,
    compute_balanced_sample_weights,
    fit_preprocessor,
    transform_preprocessor,
)
from src.luad.models.factory import create_model
from src.luad.utils.io import ensure_dir


def _map_proba_to_full(
    proba: np.ndarray,
    model_classes: Any,
    num_classes: int,
) -> np.ndarray:
    """
    Map model probability columns to full class-index columns.
    """
    full_proba = np.zeros((proba.shape[0], num_classes), dtype=float)

    for column_index, class_label in enumerate(model_classes):
        full_proba[:, int(class_label)] = proba[:, column_index]

    return full_proba


def generate_modality_oof_test_proba(
    X_train: pd.DataFrame,
    y_train_enc: pd.Series,
    X_test: pd.DataFrame,
    model_name: str,
    num_classes: int,
    cv_folds: int = 5,
    max_missing_ratio: float = 0.2,
    top_features: int = 500,
    feature_selection_method: str = "anova",
    log1p: bool = False,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Generate out-of-fold train probabilities and test probabilities for one modality.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    oof_proba = np.zeros((n_train, num_classes), dtype=float)

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics: List[Dict[str, Any]] = []

    for train_index, valid_index in skf.split(X_train, y_train_enc):
        X_fold_train = X_train.iloc[train_index]
        y_fold_train_enc = y_train_enc.iloc[train_index]

        X_fold_valid = X_train.iloc[valid_index]
        y_fold_valid_enc = y_train_enc.iloc[valid_index]

        fold_preprocessor = fit_preprocessor(
            X_train=X_fold_train,
            y_train=y_fold_train_enc.values,
            max_missing_ratio=max_missing_ratio,
            top_variance_features=top_features,
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

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                f"Model '{model_name}' does not support predict_proba for late fusion."
            )

        valid_proba_raw = model.predict_proba(X_fold_valid_processed)

        valid_proba = _map_proba_to_full(
            proba=valid_proba_raw,
            model_classes=model.classes_,
            num_classes=num_classes,
        )

        oof_proba[valid_index, :] = valid_proba

        valid_pred_enc = np.argmax(valid_proba, axis=1)

        fold_metric = compute_classification_metrics(
            y_true=y_fold_valid_enc.values,
            y_pred=valid_pred_enc,
            y_proba=valid_proba,
            labels=list(range(num_classes)),
        )

        fold_metrics.append(fold_metric)

    final_preprocessor = fit_preprocessor(
        X_train=X_train,
        y_train=y_train_enc.values,
        max_missing_ratio=max_missing_ratio,
        top_variance_features=top_features,
        feature_selection_method=feature_selection_method,
        log1p=log1p,
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

    test_proba_raw = final_model.predict_proba(X_test_processed)

    test_proba = _map_proba_to_full(
        proba=test_proba_raw,
        model_classes=final_model.classes_,
        num_classes=num_classes,
    )

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "fold_metrics": fold_metrics,
        "final_preprocessor": final_preprocessor,
        "final_model": final_model,
    }


def cross_validate_meta_learner(
    meta_train: np.ndarray,
    y_train_enc: pd.Series,
    label_encoder: LabelEncoder,
    cv_folds: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Cross-validate the meta learner on out-of-fold base probabilities.
    """
    original_classes = list(label_encoder.classes_)

    skf = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    fold_metrics: List[Dict[str, Any]] = []

    for train_index, valid_index in skf.split(meta_train, y_train_enc):
        X_meta_train = meta_train[train_index]
        y_meta_train_enc = y_train_enc.iloc[train_index]

        X_meta_valid = meta_train[valid_index]
        y_meta_valid_enc = y_train_enc.iloc[valid_index]

        meta_model = LogisticRegression(
            max_iter=5000,
            random_state=random_state,
            class_weight="balanced",
        )

        meta_model.fit(X_meta_train, y_meta_train_enc)

        y_meta_valid_pred_enc = meta_model.predict(X_meta_valid)
        y_meta_valid_proba = meta_model.predict_proba(X_meta_valid)

        y_meta_valid_pred = label_encoder.inverse_transform(y_meta_valid_pred_enc)
        y_meta_valid_original = label_encoder.inverse_transform(
            y_meta_valid_enc.values
        )

        fold_metric = compute_classification_metrics(
            y_true=y_meta_valid_original,
            y_pred=y_meta_valid_pred,
            y_proba=y_meta_valid_proba,
            labels=original_classes,
        )

        fold_metrics.append(fold_metric)

    return _aggregate_fold_metrics(fold_metrics)


def run_late_fusion(
    X_train_dict: Dict[str, pd.DataFrame],
    y_train: pd.Series,
    X_test_dict: Dict[str, pd.DataFrame],
    y_test: pd.Series,
    output_dir: Path,
    base_model_name: str = "random_forest",
    log1p_keys: Optional[List[str]] = None,
    cv_folds: int = 5,
    max_missing_ratio: float = 0.2,
    top_features: int = 500,
    feature_selection_method: str = "anova",
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Run leakage-safe late fusion across modalities.
    """
    output_dir = ensure_dir(output_dir)
    models_dir = ensure_dir(output_dir / "models")

    if log1p_keys is None:
        log1p_keys = []

    log1p_set = set(log1p_keys)

    modality_order = list(X_train_dict.keys())

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

    train_meta_parts: List[np.ndarray] = []
    test_meta_parts: List[np.ndarray] = []

    base_models: Dict[str, Any] = {}
    base_preprocessors: Dict[str, Any] = {}
    base_summaries: List[Dict[str, Any]] = []

    for modality_key in modality_order:
        X_train_modality = X_train_dict[modality_key]
        X_test_modality = X_test_dict[modality_key]

        use_log1p = modality_key in log1p_set

        modality_output = generate_modality_oof_test_proba(
            X_train=X_train_modality,
            y_train_enc=y_train_enc,
            X_test=X_test_modality,
            model_name=base_model_name,
            num_classes=num_classes,
            cv_folds=cv_folds,
            max_missing_ratio=max_missing_ratio,
            top_features=top_features,
            feature_selection_method=feature_selection_method,
            log1p=use_log1p,
            random_state=random_state,
        )

        train_meta_parts.append(modality_output["oof_proba"])
        test_meta_parts.append(modality_output["test_proba"])

        base_models[modality_key] = modality_output["final_model"]
        base_preprocessors[modality_key] = modality_output["final_preprocessor"]

        base_cv_metrics = _aggregate_fold_metrics(modality_output["fold_metrics"])

        base_summaries.append(
            {
                "modality": modality_key,
                "base_model": base_model_name,
                "log1p": use_log1p,
                "cv_metrics_encoded_labels": base_cv_metrics,
            }
        )

    meta_train = np.hstack(train_meta_parts)
    meta_test = np.hstack(test_meta_parts)

    meta_cv_metrics = cross_validate_meta_learner(
        meta_train=meta_train,
        y_train_enc=y_train_enc,
        label_encoder=label_encoder,
        cv_folds=cv_folds,
        random_state=random_state,
    )

    meta_model = LogisticRegression(
        max_iter=5000,
        random_state=random_state,
        class_weight="balanced",
    )

    meta_model.fit(meta_train, y_train_enc)

    y_test_pred_enc = meta_model.predict(meta_test)
    y_test_pred = label_encoder.inverse_transform(y_test_pred_enc)
    y_test_proba = meta_model.predict_proba(meta_test)

    test_metrics = compute_classification_metrics(
        y_true=y_test,
        y_pred=y_test_pred,
        y_proba=y_test_proba,
        labels=original_classes,
    )

    artifact_path = models_dir / "late_fusion_model.joblib"

    joblib.dump(
        {
            "base_model_name": base_model_name,
            "meta_learner": "logistic_regression",
            "modality_order": modality_order,
            "base_models": base_models,
            "base_preprocessors": base_preprocessors,
            "meta_model": meta_model,
            "label_encoder": label_encoder,
            "classes": original_classes,
            "log1p_keys": sorted(log1p_set),
            "cv_folds": cv_folds,
            "top_features": top_features,
            "feature_selection_method": feature_selection_method,
            "random_state": random_state,
        },
        artifact_path,
    )

    result = {
        "fusion": "late",
        "base_model": base_model_name,
        "meta_learner": "logistic_regression",
        "modalities": modality_order,
        "meta_cv_metrics": meta_cv_metrics,
        "test_metrics": test_metrics,
        "base_summaries": base_summaries,
        "meta_train_shape": list(meta_train.shape),
        "meta_test_shape": list(meta_test.shape),
        "model_artifact": str(artifact_path),
    }

    results_path = output_dir / "late_fusion_results.json"

    with results_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    comparison_row = {
        "model": f"late_fusion_{base_model_name}",
        "modalities": "+".join(modality_order),
        "meta_cv_balanced_accuracy_mean": meta_cv_metrics.get(
            "balanced_accuracy",
            {},
        ).get("mean"),
        "meta_cv_balanced_accuracy_std": meta_cv_metrics.get(
            "balanced_accuracy",
            {},
        ).get("std"),
        "meta_cv_f1_macro_mean": meta_cv_metrics.get("f1_macro", {}).get("mean"),
        "meta_cv_f1_macro_std": meta_cv_metrics.get("f1_macro", {}).get("std"),
        "meta_cv_roc_auc_mean": meta_cv_metrics.get("roc_auc", {}).get("mean"),
        "meta_cv_roc_auc_std": meta_cv_metrics.get("roc_auc", {}).get("std"),
        "test_accuracy": test_metrics.get("accuracy"),
        "test_balanced_accuracy": test_metrics.get("balanced_accuracy"),
        "test_precision_macro": test_metrics.get("precision_macro"),
        "test_recall_macro": test_metrics.get("recall_macro"),
        "test_f1_macro": test_metrics.get("f1_macro"),
        "test_roc_auc": test_metrics.get("roc_auc"),
        "test_mcc": test_metrics.get("mcc"),
    }

    comparison_df = pd.DataFrame([comparison_row])

    comparison_path = output_dir / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    return result