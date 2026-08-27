"""Patient-level train/test splitting utilities.

This module creates locked patient-level splits for downstream modeling.
The test set must remain untouched until final evaluation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from src.luad.utils.io import ensure_dir, save_json


DEFAULT_COHORT_DEFINITIONS: Dict[str, List[str]] = {
    "stage_only": [],
    "mrna_only": ["has_mrna_rsem"],
    "clinical_mrna": ["has_mrna_rsem"],
    "mrna_mirna": ["has_mrna_rsem", "has_mirna"],
    "mrna_mirna_methylation": [
        "has_mrna_rsem",
        "has_mirna",
        "has_methylation",
    ],
    "core_omics": [
        "has_mrna_rsem",
        "has_mirna",
        "has_methylation",
        "has_cna_raw",
    ],
    "core_mutation": [
        "has_mrna_rsem",
        "has_mirna",
        "has_methylation",
        "has_cna_raw",
        "has_mutations",
    ],
    "core_rppa": [
        "has_mrna_rsem",
        "has_mirna",
        "has_methylation",
        "has_cna_raw",
        "has_rppa_zscores",
    ],
    "all_main": [
        "has_mrna_rsem",
        "has_mirna",
        "has_methylation",
        "has_cna_raw",
        "has_rppa_zscores",
        "has_mutations",
    ],
}


def load_patient_manifest(project_root: Path) -> pd.DataFrame:
    """
    Load patient manifest created by build_cohort.py.
    """
    manifest_path = Path(project_root) / "artifacts" / "cohort" / "patient_manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patient manifest not found: {manifest_path}. "
            "Run scripts/build_cohort.py first."
        )

    manifest_df = pd.read_csv(manifest_path)

    if "patient_id" not in manifest_df.columns:
        raise ValueError("patient_manifest.csv must contain a 'patient_id' column.")

    manifest_df["patient_id"] = manifest_df["patient_id"].astype(str)

    return manifest_df


def filter_cohort(
    manifest_df: pd.DataFrame,
    cohort_name: str,
    cohort_definitions: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    """
    Filter manifest to a cohort definition.
    """
    if cohort_definitions is None:
        cohort_definitions = DEFAULT_COHORT_DEFINITIONS

    if cohort_name not in cohort_definitions:
        raise ValueError(
            f"Cohort '{cohort_name}' not found in cohort definitions. "
            f"Available cohorts: {list(cohort_definitions.keys())}"
        )

    required_columns = cohort_definitions[cohort_name]

    cohort_df = manifest_df.copy()

    for column in required_columns:
        if column not in cohort_df.columns:
            raise ValueError(
                f"Cohort '{cohort_name}' requires column '{column}', "
                "but it was not found in patient_manifest.csv."
            )

        cohort_df = cohort_df[
            cohort_df[column].fillna(False).astype(bool)
        ]

    return cohort_df


def stratified_patient_split(
    cohort_df: pd.DataFrame,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[List[str], List[str]]:
    """
    Split patients into train/test using stratified sampling on target.
    """
    if target_column not in cohort_df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in cohort dataframe."
        )

    split_df = cohort_df.dropna(subset=[target_column]).copy()

    if split_df.empty:
        raise ValueError(
            "Cohort is empty after removing rows with missing target."
        )

    class_counts = split_df[target_column].value_counts(dropna=True)

    if (class_counts < 2).any():
        problematic_classes = class_counts[class_counts < 2].to_dict()

        raise ValueError(
            "Cannot create a stratified split because some classes have fewer than 2 samples: "
            f"{problematic_classes}"
        )

    patient_ids = split_df["patient_id"].astype(str)
    target_values = split_df[target_column]

    train_ids, test_ids = train_test_split(
        patient_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=target_values,
        shuffle=True,
    )

    return sorted(train_ids.tolist()), sorted(test_ids.tolist())


def summarize_target_distribution(
    cohort_df: pd.DataFrame,
    patient_ids: List[str],
    target_column: str,
) -> Dict[str, int]:
    """
    Summarize target distribution for a list of patient IDs.
    """
    subset_df = cohort_df[cohort_df["patient_id"].isin(patient_ids)]

    value_counts = subset_df[target_column].value_counts(dropna=True)

    return {str(key): int(count) for key, count in value_counts.items()}


def make_split(
    config: Dict[str, Any],
    cohort_name: Optional[str] = None,
    output_root: Optional[str] = None,
    logger: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Create and save a locked patient-level train/test split.
    """
    project_root = Path(config.get("runtime", {}).get("project_root", Path.cwd()))

    manifest_df = load_patient_manifest(project_root)

    split_config = config.get("split", {})
    cohort_config = config.get("cohort", {})

    if cohort_name is None:
        cohort_name = cohort_config.get("primary", "stage_only")

    cohort_definitions = cohort_config.get("definitions", DEFAULT_COHORT_DEFINITIONS)

    target_column = cohort_config.get("target_column")

    if target_column is None:
        target_column = (
            config.get("task", {})
            .get("target_preprocessing", {})
            .get("grouped_target_column", "stage_group")
        )

    test_size = float(split_config.get("test_size", 0.20))
    random_state = int(
        split_config.get(
            "random_state",
            config.get("project", {}).get("random_state", 42),
        )
    )

    cohort_df = filter_cohort(
        manifest_df=manifest_df,
        cohort_name=cohort_name,
        cohort_definitions=cohort_definitions,
    )

    if cohort_df.empty:
        raise ValueError(
            f"Cohort '{cohort_name}' is empty. Check cohort definition and manifest."
        )

    if logger is not None:
        logger.info(
            f"Creating split for cohort '{cohort_name}' with {cohort_df.shape[0]} patients."
        )

    train_ids, test_ids = stratified_patient_split(
        cohort_df=cohort_df,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    if output_root is None:
        split_dir = project_root / "artifacts" / "splits" / cohort_name
    else:
        split_dir = Path(output_root) / cohort_name

    split_dir = ensure_dir(split_dir)

    train_df = pd.DataFrame({"patient_id": train_ids})
    test_df = pd.DataFrame({"patient_id": test_ids})

    train_path = split_dir / "train_patient_ids.csv"
    test_path = split_dir / "test_patient_ids.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8-sig")
    test_df.to_csv(test_path, index=False, encoding="utf-8-sig")

    split_manifest_df = cohort_df.copy()

    split_manifest_df["split"] = split_manifest_df["patient_id"].map(
        lambda patient_id: "train" if patient_id in set(train_ids) else "test"
    )

    split_manifest_path = split_dir / "split_manifest.csv"
    split_manifest_df.to_csv(split_manifest_path, index=False, encoding="utf-8-sig")

    summary = {
        "cohort_name": cohort_name,
        "target_column": target_column,
        "cohort_size": int(cohort_df.shape[0]),
        "train_size": int(len(train_ids)),
        "test_size": int(len(test_ids)),
        "test_fraction": round(len(test_ids) / len(train_ids + test_ids), 6),
        "random_state": random_state,
        "split_level": "patient",
        "stratified": True,
        "lock_test_set": True,
        "required_columns": cohort_definitions.get(cohort_name, []),
        "train_target_distribution": summarize_target_distribution(
            cohort_df=cohort_df,
            patient_ids=train_ids,
            target_column=target_column,
        ),
        "test_target_distribution": summarize_target_distribution(
            cohort_df=cohort_df,
            patient_ids=test_ids,
            target_column=target_column,
        ),
        "outputs": {
            "train_patient_ids": str(train_path),
            "test_patient_ids": str(test_path),
            "split_manifest": str(split_manifest_path),
        },
    }

    summary_path = split_dir / "split_summary.json"
    save_json(summary_path, summary)

    if logger is not None:
        logger.info(
            f"Split saved to: {split_dir} | "
            f"train={len(train_ids)}, test={len(test_ids)}"
        )

    return summary