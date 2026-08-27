"""Model factory for baseline classifiers."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def create_model(model_name: str, random_state: int = 42, num_classes: int = 4):
    """
    Create a baseline classifier by name.
    """
    model_name = str(model_name).lower()

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=5000,
            random_state=random_state,
            class_weight="balanced",
            solver="lbfgs",
        )

    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
            n_jobs=-1,
        )

    if model_name == "svm":
        return SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        )

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required. Install it with: pip install xgboost"
            ) from exc

        if num_classes <= 2:
            objective = "binary:logistic"
            eval_metric = "logloss"
        else:
            objective = "multi:softprob"
            eval_metric = "mlogloss"

        return XGBClassifier(
            objective=objective,
            eval_metric=eval_metric,
            n_estimators=400,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.5,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )

    raise ValueError(f"Unknown model name: {model_name}")