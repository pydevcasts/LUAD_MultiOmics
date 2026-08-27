"""Classification metrics for multiclass staging evaluation."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true: Any,
    y_pred: Any,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute robust multiclass classification metrics.

    Metrics include:
    - accuracy
    - balanced_accuracy
    - macro/weighted precision/recall/f1
    - MCC
    - ROC-AUC if probabilities are available
    - per-class metrics
    """
    y_true_list = [str(value) for value in list(y_true)]
    y_pred_list = [str(value) for value in list(y_pred)]

    if labels is None:
        labels = sorted(set(y_true_list) | set(y_pred_list))

    labels = [str(label) for label in labels]

    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true_list, y_pred_list)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true_list, y_pred_list)
        ),
        "precision_macro": float(
            precision_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true_list,
                y_pred_list,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(y_true_list, y_pred_list)),
        "roc_auc": None,
    }

    per_class_metrics: Dict[str, Dict[str, float]] = {}

    precision_per_class = precision_score(
        y_true_list,
        y_pred_list,
        labels=labels,
        average=None,
        zero_division=0,
    )

    recall_per_class = recall_score(
        y_true_list,
        y_pred_list,
        labels=labels,
        average=None,
        zero_division=0,
    )

    f1_per_class = f1_score(
        y_true_list,
        y_pred_list,
        labels=labels,
        average=None,
        zero_division=0,
    )

    for index, label in enumerate(labels):
        per_class_metrics[label] = {
            "precision": float(precision_per_class[index]),
            "recall": float(recall_per_class[index]),
            "f1": float(f1_per_class[index]),
        }

    metrics["per_class"] = per_class_metrics

    if y_proba is not None and len(labels) > 1:
        try:
            y_proba_array = np.asarray(y_proba)

            if len(labels) == 2:
                positive_label = labels[1]

                y_binary = np.array(
                    [str(value) == positive_label for value in y_true_list],
                    dtype=int,
                )

                metrics["roc_auc"] = float(
                    roc_auc_score(y_binary, y_proba_array[:, 1])
                )
            else:
                y_true_bin = label_binarize(y_true_list, classes=labels)

                metrics["roc_auc"] = float(
                    roc_auc_score(
                        y_true_bin,
                        y_proba_array,
                        average="macro",
                    )
                )
        except Exception:
            metrics["roc_auc"] = None

    return metrics