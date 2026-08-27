"""Train leakage-safe late fusion models on locked train/test split."""

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.fusion.late import run_late_fusion
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
        description="Train leakage-safe late fusion models."
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
        default="late_fusion",
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
        default="mrna_rsem,mirna,cna_raw,mutations",
        help="Comma-separated modality keys for late fusion.",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model for each modality. Defaults to late_fusion.base_model.",
    )

    parser.add_argument(
        "--target-mode",
        type=str,
        choices=["multiclass", "binary_early_late"],
        default="multiclass",
        help="Target formulation.",
    )

    parser.add_argument(
        "--top-features",
        type=int,
        default=None,
        help="Top features selected inside each modality model.",
    )

    parser.add_argument(
        "--no-log1p",
        action="store_true",
        help="Disable log1p for expression modalities.",
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
        name="luad.late_fusion",
        log_file=project_root / "artifacts" / "logs" / "late_fusion_training.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    late_fusion_config = config.get("late_fusion", {})
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

    X_train_dict = {}
    X_test_dict = {}

    for modality_key in modality_keys:
        logger.info(f"Loading modality: {modality_key}")

        X_train_modality = build_modality_dataset(
            interim_dir=interim_dir,
            modality_keys=[modality_key],
            patient_ids=train_ids,
        )

        X_test_modality = build_modality_dataset(
            interim_dir=interim_dir,
            modality_keys=[modality_key],
            patient_ids=test_ids,
        )

        X_train_dict[modality_key] = X_train_modality
        X_test_dict[modality_key] = X_test_modality

    common_train_patients = None
    common_test_patients = None

    for modality_key in modality_keys:
        train_index_set = set(X_train_dict[modality_key].index)
        test_index_set = set(X_test_dict[modality_key].index)

        if common_train_patients is None:
            common_train_patients = train_index_set
        else:
            common_train_patients = common_train_patients.intersection(train_index_set)

        if common_test_patients is None:
            common_test_patients = test_index_set
        else:
            common_test_patients = common_test_patients.intersection(test_index_set)

    common_train_patients = sorted(common_train_patients)
    common_test_patients = sorted(common_test_patients)

    logger.info(f"Common train patients across modalities: {len(common_train_patients)}")
    logger.info(f"Common test patients across modalities: {len(common_test_patients)}")

    for modality_key in modality_keys:
        X_train_dict[modality_key] = X_train_dict[modality_key].loc[common_train_patients]
        X_test_dict[modality_key] = X_test_dict[modality_key].loc[common_test_patients]

    target_df = load_clinical_target(interim_dir)
    target_df = target_df.set_index("patient_id")

    y_train = target_df.loc[common_train_patients, "stage_group"]
    y_test = target_df.loc[common_test_patients, "stage_group"]

    if args.target_mode == "binary_early_late":
        y_train = y_train.map(STAGE_TO_BINARY_EARLY_LATE)
        y_test = y_test.map(STAGE_TO_BINARY_EARLY_LATE)

        train_valid_mask = y_train.notna()
        test_valid_mask = y_test.notna()

        y_train = y_train.loc[train_valid_mask]
        y_test = y_test.loc[test_valid_mask]

        common_train_patients = y_train.index.tolist()
        common_test_patients = y_test.index.tolist()

        for modality_key in modality_keys:
            X_train_dict[modality_key] = X_train_dict[modality_key].loc[
                common_train_patients
            ]
            X_test_dict[modality_key] = X_test_dict[modality_key].loc[
                common_test_patients
            ]

        logger.info("Converted target to binary early/late.")
        logger.info(
            f"Train binary target distribution: {y_train.value_counts().to_dict()}"
        )
        logger.info(
            f"Test binary target distribution: {y_test.value_counts().to_dict()}"
        )

    base_model_name = args.base_model or late_fusion_config.get(
        "base_model",
        "random_forest",
    )

    top_features = args.top_features or int(late_fusion_config.get("top_features", 500))

    if args.no_log1p:
        log1p_keys = []
    else:
        log1p_keys = [
            key
            for key in baseline_config.get("log1p_modalities", [])
            if key in modality_keys
        ]

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "artifacts" / "experiments" / args.experiment_name

    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Top features per modality: {top_features}")
    logger.info(f"log1p modalities: {log1p_keys}")
    logger.info(f"Output directory: {output_dir}")

    result = run_late_fusion(
        X_train_dict=X_train_dict,
        y_train=y_train,
        X_test_dict=X_test_dict,
        y_test=y_test,
        output_dir=output_dir,
        base_model_name=base_model_name,
        log1p_keys=log1p_keys,
        cv_folds=int(late_fusion_config.get("cv_folds", 5)),
        max_missing_ratio=float(late_fusion_config.get("max_missing_ratio", 0.2)),
        top_features=top_features,
        feature_selection_method=late_fusion_config.get(
            "feature_selection_method",
            "anova",
        ),
        random_state=int(late_fusion_config.get("random_state", 42)),
    )

    meta_cv_balanced = result["meta_cv_metrics"].get("balanced_accuracy", {})
    test_metrics = result["test_metrics"]

    print("\n=== Late Fusion Result ===")
    print(f"Experiment: {args.experiment_name}")
    print(f"Base model: {result['base_model']}")
    print(f"Modalities: {', '.join(result['modalities'])}")

    print(
        f"- meta_cv_balanced_acc={meta_cv_balanced.get('mean'):.4f} "
        f"± {meta_cv_balanced.get('std'):.4f} | "
        f"test_balanced_acc={test_metrics.get('balanced_accuracy'):.4f} | "
        f"test_macro_f1={test_metrics.get('f1_macro'):.4f} | "
        f"test_mcc={test_metrics.get('mcc'):.4f} | "
        f"test_roc_auc={test_metrics.get('roc_auc')}"
    )

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()