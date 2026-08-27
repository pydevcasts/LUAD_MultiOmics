"""Prepare survival prediction dataset with clinical features."""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare survival prediction dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def encode_categorical_column(series, max_categories=10):
    """Encode categorical column with one-hot encoding, limiting to top categories."""
    if series.isna().sum() == len(series):
        return pd.DataFrame(index=series.index)

    filled = series.fillna("MISSING")
    top_categories = filled.value_counts().head(max_categories).index.tolist()

    result = {}
    for cat in top_categories:
        col_name = f"{series.name}_{cat}"
        result[col_name] = (filled == cat).astype(float)

    return pd.DataFrame(result, index=series.index)


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
        name="luad.survival_prep",
        log_file=project_root / "artifacts" / "logs" / "survival_prep.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    interim_dir = project_root / config.get("paths", {}).get("interim_dir", "data/interim")

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "data" / "processed" / "survival"

    output_dir = ensure_dir(output_dir)

    logger.info("Starting survival prediction dataset preparation (with clinical).")

    # ===== Load Clinical Data =====
    clinical_path = interim_dir / "clinical_stage_cohort.csv"
    if not clinical_path.exists():
        logger.error(f"Clinical file not found: {clinical_path}")
        sys.exit(1)

    clinical_df = pd.read_csv(clinical_path)
    logger.info(f"Clinical data loaded: {clinical_df.shape}")

    # ===== Extract Survival Label =====
    if "OS_STATUS" not in clinical_df.columns:
        logger.error("OS_STATUS column not found in clinical data.")
        sys.exit(1)

    survival_df = clinical_df[["patient_id", "OS_STATUS"]].copy()
    survival_df = survival_df.dropna(subset=["OS_STATUS"])
    survival_df["survival_label"] = survival_df["OS_STATUS"].apply(
        lambda x: 1 if "DECEASED" in str(x).upper() else 0
    )

    logger.info(f"Survival label distribution: {survival_df['survival_label'].value_counts().to_dict()}")
    logger.info(f"Patients with survival label: {len(survival_df)}")

    # ===== Extract Clinical Features =====
    logger.info("Extracting clinical features...")

    clinical_features_list = []

    # AGE (continuous)
    if "AGE" in clinical_df.columns:
        age_col = clinical_df["AGE"].copy()
        age_median = age_col.median()
        age_col = age_col.fillna(age_median)
        age_df = pd.DataFrame({"AGE": age_col}, index=clinical_df.index)
        clinical_features_list.append(age_df)
        logger.info(f"  Added AGE (median imputation: {age_median})")

    # SEX (binary)
    if "SEX" in clinical_df.columns:
        sex_encoded = encode_categorical_column(clinical_df["SEX"])
        clinical_features_list.append(sex_encoded)
        logger.info(f"  Added SEX: {list(sex_encoded.columns)}")

    # AJCC_PATHOLOGIC_TUMOR_STAGE (multi-class)
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in clinical_df.columns:
        stage_col = clinical_df["AJCC_PATHOLOGIC_TUMOR_STAGE"].copy()

        stage_mapping = {
            "STAGE IA": "STAGE_I",
            "STAGE IB": "STAGE_I",
            "STAGE I": "STAGE_I",
            "STAGE IIA": "STAGE_II",
            "STAGE IIB": "STAGE_II",
            "STAGE II": "STAGE_II",
            "STAGE IIIA": "STAGE_III",
            "STAGE IIIB": "STAGE_III",
            "STAGE IIIC": "STAGE_III",
            "STAGE III": "STAGE_III",
            "STAGE IV": "STAGE_IV",
            "STAGE IVA": "STAGE_IV",
            "STAGE IVB": "STAGE_IV",
        }

        stage_mapped = stage_col.map(stage_mapping).fillna("MISSING")
        stage_encoded = encode_categorical_column(
            pd.Series(stage_mapped, name="STAGE_GROUP", index=clinical_df.index)
        )
        clinical_features_list.append(stage_encoded)
        logger.info(f"  Added STAGE_GROUP: {list(stage_encoded.columns)}")

    # ICD_O_3_HISTOLOGY (multi-class)
    if "ICD_O_3_HISTOLOGY" in clinical_df.columns:
        histology_encoded = encode_categorical_column(clinical_df["ICD_O_3_HISTOLOGY"], max_categories=8)
        clinical_features_list.append(histology_encoded)
        logger.info(f"  Added ICD_O_3_HISTOLOGY: {list(histology_encoded.columns)}")

    # RACE
    if "RACE" in clinical_df.columns:
        race_encoded = encode_categorical_column(clinical_df["RACE"], max_categories=5)
        clinical_features_list.append(race_encoded)
        logger.info(f"  Added RACE: {list(race_encoded.columns)}")

    # ===== Combine Clinical Features =====
    if clinical_features_list:
        clinical_features = pd.concat(clinical_features_list, axis=1)
        clinical_features.index = clinical_df.index
        logger.info(f"Clinical features matrix shape: {clinical_features.shape}")
    else:
        logger.warning("No clinical features could be extracted.")
        clinical_features = pd.DataFrame(index=clinical_df.index)

    # ===== Merge with Survival Labels =====
    survival_with_clinical = survival_df.merge(
        clinical_features,
        left_index=True,
        right_index=True,
        how="left",
    )

    survival_patients = set(survival_with_clinical["patient_id"].astype(str))

    # ===== Load Molecular Modalities =====
    modality_keys = ["mrna_rsem", "mirna", "cna_raw", "mutations"]
    modality_matrices = {}

    for modality_key in modality_keys:
        modality_path = interim_dir / f"{modality_key}.patient_features.parquet"
        if not modality_path.exists():
            logger.warning(f"Modality {modality_key} not found: {modality_path}")
            continue
        logger.info(f"Loading {modality_key}...")
        matrix = pd.read_parquet(modality_path)
        if "patient_id" in matrix.columns:
            matrix = matrix.set_index("patient_id")
        matrix.index = matrix.index.astype(str)
        available_patients = set(matrix.index) & survival_patients
        matrix = matrix.loc[matrix.index.isin(survival_patients)]
        logger.info(f"  {modality_key}: {matrix.shape[0]} patients, {matrix.shape[1]} features")
        modality_matrices[modality_key] = matrix

    # ===== Find Common Patients =====
    common_patients = survival_patients.copy()
    for modality_key, matrix in modality_matrices.items():
        common_patients = common_patients & set(matrix.index)
    common_patients = sorted(common_patients)
    logger.info(f"Common patients across all modalities: {len(common_patients)}")

    # ===== Filter to Common Patients =====
    survival_with_clinical = survival_with_clinical[
        survival_with_clinical["patient_id"].astype(str).isin(common_patients)
    ].copy()

    for modality_key in list(modality_matrices.keys()):
        modality_matrices[modality_key] = modality_matrices[modality_key].loc[common_patients]

    # ===== Save Clinical Features =====
    clinical_cols = [c for c in survival_with_clinical.columns if c not in ["patient_id", "OS_STATUS", "survival_label"]]
    clinical_matrix = survival_with_clinical.set_index("patient_id")[clinical_cols]
    clinical_path_out = output_dir / "clinical_features.parquet"
    clinical_matrix.to_parquet(clinical_path_out)
    logger.info(f"Clinical features saved: {clinical_matrix.shape}")

    # ===== Save Labels =====
    survival_path = output_dir / "survival_labels.csv"
    survival_with_clinical[["patient_id", "survival_label"]].to_csv(survival_path, index=False, encoding="utf-8-sig")

    # ===== Save Molecular Modalities =====
    for modality_key, matrix in modality_matrices.items():
        matrix_path = output_dir / f"{modality_key}.parquet"
        matrix.to_parquet(matrix_path)
        logger.info(f"Saved {modality_key} to: {matrix_path}")

    # ===== Metadata =====
    metadata = {
        "total_patients": len(common_patients),
        "survival_distribution": survival_with_clinical["survival_label"].value_counts().to_dict(),
        "modalities": list(modality_matrices.keys()),
        "clinical_features": clinical_cols,
        "modality_shapes": {
            k: {"patients": v.shape[0], "features": v.shape[1]}
            for k, v in modality_matrices.items()
        },
    }

    metadata_path = output_dir / "survival_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Survival dataset saved to: {output_dir}")

    print("\n=== Survival Dataset Summary (with Clinical) ===")
    print(f"Total patients: {len(common_patients)}")
    print(f"Survival distribution: {survival_with_clinical['survival_label'].value_counts().to_dict()}")
    print(f"Molecular modalities: {list(modality_matrices.keys())}")
    print(f"Clinical features: {clinical_cols}")

    for k, v in modality_matrices.items():
        print(f"  {k}: {v.shape[0]} patients x {v.shape[1]} features")

    print(f"  clinical: {clinical_matrix.shape[0]} patients x {clinical_matrix.shape[1]} features")


if __name__ == "__main__":
    main()