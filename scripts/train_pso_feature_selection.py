"""Run PSO feature selection on mRNA for tumor vs normal classification."""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
    parser = argparse.ArgumentParser(description="PSO Feature Selection for Tumor vs Normal.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--top-anova", type=int, default=1000, help="Features entering PSO after ANOVA")
    parser.add_argument("--pso-target", type=int, default=100, help="Target number of features for PSO")
    parser.add_argument("--pso-particles", type=int, default=30)
    parser.add_argument("--pso-iterations", type=int, default=50)
    parser.add_argument("--experiment-name", type=str, default="pso_tumor_normal")
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
        name="luad.pso_fs",
        log_file=project_root / "artifacts" / "logs" / "pso_feature_selection.log",
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

    logger.info(f"Loading tumor vs normal data from: {combined_path}")
    combined_df = pd.read_parquet(combined_path)

    if "sample_id" in combined_df.columns:
        combined_df = combined_df.set_index("sample_id")

    # Convert string labels to binary integers
    label_map = {"NORMAL": 0, "TUMOR": 1}
    y = combined_df["label"].map(label_map).astype(int)
    X = combined_df.drop(columns=["label"])

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    # ===== Train/Test Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ===== Step 1: Preprocessing + ANOVA on Train Only =====
    logger.info(f"\nStep 1: ANOVA filtering on train set (top-{args.top_anova})...")

    preproc_anova = fit_preprocessor(
        X_train=X_train,
        y_train=y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=args.top_anova,
        feature_selection_method="anova",
        log1p=True,
    )

    X_train_filtered = transform_preprocessor(X_train, preproc_anova)
    X_test_filtered = transform_preprocessor(X_test, preproc_anova)

    anova_feature_names = preproc_anova["selected_features"]
    logger.info(f"After ANOVA: {X_train_filtered.shape[1]} features")

    # ===== Step 2: PSO on Filtered Train Data =====
    logger.info(f"\nStep 2: Running PSO (target={args.pso_target} features)...")

    pso = BinaryPSOFeatureSelector(
        n_particles=args.pso_particles,
        n_iterations=args.pso_iterations,
        target_n_features=args.pso_target,
        model_name="random_forest",
        cv_folds=3,
        random_state=42,
        logger=logger,
    )

    pso.fit(
        X=X_train_filtered,
        y=y_train.values,
        feature_names=anova_feature_names,
    )

    selected_indices = pso.get_selected_indices()
    selected_names = pso.get_selected_feature_names(anova_feature_names)

    logger.info(f"\nPSO selected {len(selected_names)} features:")
    for i, name in enumerate(selected_names[:20]):
        logger.info(f"  {i+1}. {name}")
    if len(selected_names) > 20:
        logger.info(f"  ... and {len(selected_names) - 20} more")

    # ===== Step 3: Evaluate Final Model with PSO Features =====
    logger.info("\nStep 3: Evaluating final models with PSO-selected features...")

    X_train_pso = X_train_filtered[:, selected_indices]
    X_test_pso = X_test_filtered[:, selected_indices]

    output_dir = project_root / "artifacts" / "experiments" / args.experiment_name
    output_dir = ensure_dir(output_dir)

    model_names = ["logistic_regression", "random_forest", "svm", "xgboost"]
    results = []

    for model_name in model_names:
        logger.info(f"  Training {model_name} on {len(selected_indices)} PSO features...")

        model = create_model(model_name=model_name, random_state=42, num_classes=2)
        model.fit(X_train_pso, y_train)

        y_pred = model.predict(X_test_pso)
        y_proba = model.predict_proba(X_test_pso) if hasattr(model, "predict_proba") else None

        metrics = compute_classification_metrics(
            y_true=y_test.values,
            y_pred=y_pred,
            y_proba=y_proba,
            labels=[0, 1],
        )

        results.append({
            "model": model_name,
            "n_pso_features": len(selected_indices),
            "test_accuracy": metrics["accuracy"],
            "test_balanced_accuracy": metrics["balanced_accuracy"],
            "test_f1_macro": metrics["f1_macro"],
            "test_roc_auc": metrics["roc_auc"],
            "test_mcc": metrics["mcc"],
        })

        logger.info(
            f"    {model_name}: "
            f"BA={metrics['balanced_accuracy']:.4f}, "
            f"AUC={metrics['roc_auc']:.4f}, "
            f"F1={metrics['f1_macro']:.4f}"
        )

    # ===== Save Results =====
    pso_results = {
        "pso_config": {
            "n_particles": args.pso_particles,
            "n_iterations": args.pso_iterations,
            "target_n_features": args.pso_target,
            "anova_top_k": args.top_anova,
        },
        "pso_history": pso.get_history(),
        "selected_features": selected_names,
        "n_selected": len(selected_names),
        "model_results": results,
    }

    results_path = output_dir / "pso_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(pso_results, f, ensure_ascii=False, indent=2, default=str)

    comp_df = pd.DataFrame(results)
    comp_path = output_dir / "model_comparison.csv"
    comp_df.to_csv(comp_path, index=False, encoding="utf-8-sig")

    selected_path = output_dir / "pso_selected_genes.csv"
    pd.DataFrame({"gene": selected_names}).to_csv(selected_path, index=False, encoding="utf-8-sig")

    logger.info(f"\nResults saved to: {output_dir}")

    print("\n" + "=" * 70)
    print("PSO FEATURE SELECTION RESULTS (Tumor vs Normal)")
    print("=" * 70)
    print(f"ANOVA input: {args.top_anova} features")
    print(f"PSO output:  {len(selected_names)} features")
    print("-" * 70)
    for r in results:
        print(
            f"{r['model']:25s} | "
            f"BA={r['test_balanced_accuracy']:.4f} | "
            f"AUC={r['test_roc_auc']:.4f} | "
            f"F1={r['test_f1_macro']:.4f} | "
            f"MCC={r['test_mcc']:.4f}"
        )
    print("=" * 70)
    print(f"\nSelected genes saved to: {selected_path}")
    print(f"Full results saved to: {output_dir}")


if __name__ == "__main__":
    main()