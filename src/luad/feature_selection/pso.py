"""Particle Swarm Optimization for feature selection (v3 - imbalanced-safe).

Binary PSO with multi-objective fitness, designed for highly imbalanced datasets.
Works directly with NumPy arrays.

Fitness = α × BalancedAccuracy - β × Sparsity + γ × Target_Proximity
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, List, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score


class BinaryPSOFeatureSelector:
    """
    Binary Particle Swarm Optimization for feature selection.
    Optimized for imbalanced cancer datasets.
    """

    def __init__(
        self,
        n_particles: int = 30,
        n_iterations: int = 50,
        w_min: float = 0.4,
        w_max: float = 0.9,
        c1: float = 1.5,
        c2: float = 1.5,
        alpha: float = 0.8,
        beta: float = 0.1,
        gamma: float = 0.1,
        target_n_features: int = 100,
        model_name: str = "random_forest",
        cv_folds: int = 5,
        random_state: int = 42,
        logger: Optional[Any] = None,
    ):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w_min = w_min
        self.w_max = w_max
        self.c1 = c1
        self.c2 = c2
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.target_n_features = target_n_features
        self.model_name = model_name
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.logger = logger

        self.best_position: Optional[np.ndarray] = None
        self.best_fitness: float = -np.inf
        self.history: List[Dict[str, Any]] = []

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.info(message)
        else:
            print(message)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _create_model(self, n_classes: int):
        name = self.model_name.lower()
        if name == "random_forest":
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight="balanced", random_state=self.random_state, n_jobs=-1,
            )
        elif name == "xgboost":
            try:
                from xgboost import XGBClassifier
                obj = "binary:logistic" if n_classes == 2 else "multi:softprob"
                metric = "logloss" if n_classes == 2 else "mlogloss"
                return XGBClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1,
                    objective=obj, eval_metric=metric, use_label_encoder=False,
                    random_state=self.random_state, verbosity=0,
                )
            except ImportError:
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=self.random_state)
        elif name == "svm":
            from sklearn.svm import SVC
            return SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=self.random_state)
        else:
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", random_state=self.random_state)

    def _compute_fitness(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
        n_selected = int(np.sum(mask))
        n_total = X.shape[1]

        # Only reject empty subsets
        if n_selected < 5:
            return -1.0

        X_selected = X[:, mask]

        # Handle NaN
        if np.any(np.isnan(X_selected)):
            col_medians = np.nanmedian(X_selected, axis=0)
            nan_mask = np.isnan(X_selected)
            X_selected = np.where(nan_mask, col_medians, X_selected)

        try:
            classes = np.unique(y)
            n_classes = len(classes)

            # Use minimum of cv_folds and min class count to avoid stratification errors
            min_class_count = min(np.bincount(y.astype(int)))
            safe_folds = min(self.cv_folds, max(2, min_class_count))

            skf = StratifiedKFold(n_splits=safe_folds, shuffle=True, random_state=self.random_state)

            fold_scores = []
            for train_idx, val_idx in skf.split(X_selected, y):
                X_tr, y_tr = X_selected[train_idx], y[train_idx]
                X_val, y_val = X_selected[val_idx], y[val_idx]

                if len(np.unique(y_val)) < 2 or len(np.unique(y_tr)) < 2:
                    continue

                model = self._create_model(n_classes)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_val)
                score = balanced_accuracy_score(y_val, y_pred)
                fold_scores.append(score)

            if len(fold_scores) == 0:
                return -1.0

            mean_accuracy = float(np.mean(fold_scores))

        except Exception:
            return -1.0

        # Softer sparsity: penalize only when far from target
        ratio = n_selected / n_total
        sparsity_penalty = ratio  # Simple linear penalty

        # Target proximity: reward being close to target_n_features
        distance = abs(n_selected - self.target_n_features)
        max_distance = max(self.target_n_features, n_total - self.target_n_features)
        target_proximity = max(0.0, 1.0 - distance / max_distance)

        fitness = (
            self.alpha * mean_accuracy
            - self.beta * sparsity_penalty
            + self.gamma * target_proximity
        )

        return float(fitness)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        self._log(
            f"PSO v3 starting: {n_features} features, target={self.target_n_features}, "
            f"particles={self.n_particles}, iterations={self.n_iterations}, "
            f"α={self.alpha}, β={self.beta}, γ={self.gamma}"
        )

        # Initialize with bias toward fewer features (sparse initialization)
        positions = rng.uniform(-4, 2, size=(self.n_particles, n_features))
        velocities = rng.uniform(-1, 1, size=(self.n_particles, n_features))

        personal_best_positions = positions.copy()
        personal_best_fitness = np.full(self.n_particles, -np.inf)

        self.best_position = None
        self.best_fitness = -np.inf
        self.history = []

        for iteration in range(self.n_iterations):
            w = self.w_max - (self.w_max - self.w_min) * (iteration / max(1, self.n_iterations - 1))

            valid_evaluations = 0

            for p in range(self.n_particles):
                binary_mask = (self._sigmoid(positions[p]) > rng.rand(n_features)).astype(int)
                fitness = self._compute_fitness(X, y, binary_mask)

                if fitness > -1.0:
                    valid_evaluations += 1

                if fitness > personal_best_fitness[p]:
                    personal_best_fitness[p] = fitness
                    personal_best_positions[p] = positions[p].copy()

                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_position = binary_mask.copy()

            r1 = rng.rand(self.n_particles, n_features)
            r2 = rng.rand(self.n_particles, n_features)

            cognitive = self.c1 * r1 * (personal_best_positions - positions)
            social = self.c2 * r2 * (self.best_position.astype(float) - positions)

            velocities = w * velocities + cognitive + social
            positions = positions + velocities

            n_selected = int(np.sum(self.best_position)) if self.best_position is not None else 0

            iter_record = {
                "iteration": iteration + 1,
                "best_fitness": round(self.best_fitness, 6),
                "n_selected": n_selected,
                "inertia_weight": round(w, 4),
                "valid_evaluations": valid_evaluations,
            }
            self.history.append(iter_record)

            if (iteration + 1) % 5 == 0 or iteration == 0:
                self._log(
                    f"  Iter {iteration+1}/{self.n_iterations}: "
                    f"fitness={self.best_fitness:.4f}, "
                    f"selected={n_selected}/{n_features}, "
                    f"valid={valid_evaluations}/{self.n_particles}"
                )

        final_selected = int(np.sum(self.best_position)) if self.best_position is not None else 0
        self._log(f"PSO finished: best_fitness={self.best_fitness:.4f}, selected={final_selected}")

        return self

    def get_selected_indices(self) -> np.ndarray:
        if self.best_position is None:
            raise ValueError("PSO has not been fitted yet.")
        return np.where(self.best_position == 1)[0]

    def get_selected_feature_names(self, feature_names: List[str]) -> List[str]:
        indices = self.get_selected_indices()
        return [feature_names[i] for i in indices]

    def get_history(self) -> List[Dict[str, Any]]:
        return self.history