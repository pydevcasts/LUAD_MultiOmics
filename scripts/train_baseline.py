"""Train leakage-safe baseline models on locked train/test split."""

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.ml.baseline import run_baseline_experiment, save_baseline_results
from src.luad.ml.datasets import build_modality_dataset, load_clinical_target
from src.luad.utils.logger import get_logger


STAGE_TO_BINARY_EARLY_LATE = {
    "STAGE I": "EARLY",
    "STAGE II": "EARLY",
    "STAGE III": "LATE",
    "STAGE IV": "LATE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train leakage-safe baseline models."
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
        help="Optional experiment overlay YAML file.",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default="baseline_mrna",
        help="Experiment name used for output directory.",
    )

    parser.add_argument(
        "--split-cohort",
        type=str,
        default=None,
        help="Split cohort name. Defaults to cohort.primary in config.",
    )

    parser.add_argument(
        "--modality-keys",
        type=str,
        default="mrna_rsem",
        help="Comma-separated modality keys, e.g. mrna_rsem or mrna_rsem,mirna.",
    )

    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated model names. Defaults to baseline.models in config.",
    )

    parser.add_argument(
        "--feature-selection",
        type=str,
        choices=["variance", "anova"],
        default=None,
        help="Feature selection method inside CV.",
    )

    parser.add_argument(
        "--transform",
        type=str,
        choices=["none", "log1p"],
        default=None,
        help="Optional numeric transformation.",
    )

    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["multiclass", "binary_early_late"],
        default="multiclass",
        help="Target formulation.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_split_patient_ids(split_dir: Path, file_name: str) -> list:
    file_path = split_dir / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Split file not found: {file_path}. Run make_split.py first."
        )

    split_df = pd.read_csv(file_path)

    if "patient_id" not in split_df.columns:
        raise ValueError(f"{file_path} must contain patient_id column.")

    return split_df["patient_id"].astype(str).tolist()


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
        name="luad.baseline",
        log_file=project_root / "artifacts" / "logs" / "baseline_training.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    baseline_config = config.get("baseline", {})
    cohort_config = config.get("cohort", {})

    split_cohort = args.split_cohort or cohort_config.get("primary", "core_mutation")

    split_dir = project_root / "artifacts" / "splits" / split_cohort

    train_ids = load_split_patient_ids(split_dir, "train_patient_ids.csv")
    test_ids = load_split_patient_ids(split_dir, "test_patient_ids.csv")

    interim_dir = project_root / config.get("paths", {}).get(
        "interim_dir",
        "data/interim",
    )

    modality_keys = [
        key.strip()
        for key in args.modality_keys.split(",")
        if key.strip()
    ]

    logger.info(f"Experiment: {args.experiment_name}")
    logger.info(f"Split cohort: {split_cohort}")
    logger.info(f"Modality keys: {modality_keys}")
    logger.info(f"Target mode: {args.target_mode}")
    logger.info(f"Train patients: {len(train_ids)}")
    logger.info(f"Test patients: {len(test_ids)}")

    X_train = build_modality_dataset(
        interim_dir=interim_dir,
        modality_keys=modality_keys,
        patient_ids=train_ids,
    )

    X_test = build_modality_dataset(
        interim_dir=interim_dir,
        modality_keys=modality_keys,
        patient_ids=test_ids,
    )

    X_test = X_test.reindex(columns=X_train.columns)

    target_df = load_clinical_target(interim_dir)
    target_df = target_df.set_index("patient_id")

    y_train = target_df.loc[X_train.index, "stage_group"]
    y_test = target_df.loc[X_test.index, "stage_group"]

    missing_train_target = int(y_train.isna().sum())
    missing_test_target = int(y_test.isna().sum())

    if missing_train_target > 0 or missing_test_target > 0:
        raise ValueError(
            f"Missing target values found. train_missing={missing_train_target}, "
            f"test_missing={missing_test_target}"
        )

    if args.target_mode == "binary_early_late":
        y_train = y_train.map(STAGE_TO_BINARY_EARLY_LATE)
        y_test = y_test.map(STAGE_TO_BINARY_EARLY_LATE)

        train_valid_mask = y_train.notna()
        test_valid_mask = y_test.notna()

        X_train = X_train.loc[train_valid_mask]
        y_train = y_train.loc[train_valid_mask]

        X_test = X_test.loc[test_valid_mask]
        y_test = y_test.loc[test_valid_mask]

        logger.info("Converted target to binary early/late.")
        logger.info(f"Train binary target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Test binary target distribution: {y_test.value_counts().to_dict()}")

    model_names = baseline_config.get(
        "models",
        [
            "logistic_regression",
            "random_forest",
            "svm",
            "xgboost",
        ],
    )

    if args.models is not None:
        model_names = [
            model_name.strip()
            for model_name in args.models.split(",")
            if model_name.strip()
        ]

    feature_selection_method = (
        args.feature_selection
        or baseline_config.get("feature_selection_method", "variance")
    )

    log1p_modalities = baseline_config.get("log1p_modalities", [])

    if args.transform == "log1p":
        use_log1p = True
    elif args.transform == "none":
        use_log1p = False
    else:
        use_log1p = any(key in log1p_modalities for key in modality_keys)

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "artifacts" / "experiments" / args.experiment_name

    logger.info(f"Training models: {model_names}")
    logger.info(f"Feature selection method: {feature_selection_method}")
    logger.info(f"log1p transform: {use_log1p}")
    logger.info(f"Output directory: {output_dir}")

    results = run_baseline_experiment(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_names=model_names,
        output_dir=output_dir,
        cv_folds=int(baseline_config.get("cv_folds", 5)),
        max_missing_ratio=float(baseline_config.get("max_missing_ratio", 0.2)),
        top_variance_features=int(baseline_config.get("top_variance_features", 1000)),
        feature_selection_method=feature_selection_method,
        log1p=use_log1p,
        random_state=int(baseline_config.get("random_state", 42)),
    )

    save_baseline_results(results=results, output_dir=output_dir)

    print("\n=== Baseline Results ===")

    for result in results:
        cv_balanced_accuracy = result["cv_metrics"].get("balanced_accuracy", {})
        test_metrics = result["test_metrics"]

        print(
            f"- model={result['model']} | "
            f"cv_balanced_acc={cv_balanced_accuracy.get('mean'):.4f} "
            f"± {cv_balanced_accuracy.get('std'):.4f} | "
            f"test_balanced_acc={test_metrics.get('balanced_accuracy'):.4f} | "
            f"test_macro_f1={test_metrics.get('f1_macro'):.4f} | "
            f"test_mcc={test_metrics.get('mcc'):.4f} | "
            f"test_roc_auc={test_metrics.get('roc_auc')}"
        )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()