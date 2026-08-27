"""Dataset loading utilities for interim prepared matrices."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import pandas as pd


def load_clinical_target(interim_dir: Path) -> pd.DataFrame:
    """
    Load clinical stage cohort containing patient_id and stage_group.
    """
    clinical_path = Path(interim_dir) / "clinical_stage_cohort.csv"

    if not clinical_path.exists():
        raise FileNotFoundError(
            f"Clinical stage cohort not found: {clinical_path}. "
            "Run prepare_data.py first."
        )

    clinical_df = pd.read_csv(clinical_path)

    if "patient_id" not in clinical_df.columns:
        raise ValueError("clinical_stage_cohort.csv must contain 'patient_id'.")

    if "stage_group" not in clinical_df.columns:
        raise ValueError("clinical_stage_cohort.csv must contain 'stage_group'.")

    clinical_df["patient_id"] = clinical_df["patient_id"].astype(str)

    return clinical_df[["patient_id", "stage_group"]]


def load_modality_matrix(interim_dir: Path, modality_key: str) -> pd.DataFrame:
    """
    Load a patient-by-feature parquet matrix for one modality.
    """
    matrix_path = Path(interim_dir) / f"{modality_key}.patient_features.parquet"

    if not matrix_path.exists():
        raise FileNotFoundError(
            f"Modality matrix not found: {matrix_path}. "
            "Run prepare_data.py first."
        )

    matrix_df = pd.read_parquet(matrix_path)

    if "patient_id" in matrix_df.columns:
        matrix_df = matrix_df.set_index("patient_id")

    if matrix_df.index.name != "patient_id":
        matrix_df.index.name = "patient_id"

    matrix_df.index = matrix_df.index.astype(str)

    return matrix_df


def build_modality_dataset(
    interim_dir: Path,
    modality_keys: List[str],
    patient_ids: List[str],
) -> pd.DataFrame:
    """
    Build a patient-by-feature matrix for one or more modalities.

    For multiple modalities, patients must exist in all matrices.
    """
    interim_dir = Path(interim_dir)
    patient_ids = [str(patient_id) for patient_id in patient_ids]

    matrices: List[pd.DataFrame] = []

    for modality_key in modality_keys:
        if modality_key == "clinical":
            raise ValueError(
                "Clinical baseline is not supported by this minimal baseline loader yet."
            )

        if modality_key == "methylation":
            raise ValueError(
                "Methylation is stored as feature_by_patient and will be supported "
                "in a dedicated high-dimensional loader."
            )

        matrix_df = load_modality_matrix(
            interim_dir=interim_dir,
            modality_key=modality_key,
        )

        matrix_df = matrix_df.loc[matrix_df.index.intersection(patient_ids)]

        matrices.append(matrix_df)

    if not matrices:
        raise ValueError("No modality matrices were loaded.")

    if len(matrices) == 1:
        combined_df = matrices[0]
    else:
        combined_df = pd.concat(matrices, axis=1, join="inner")

    combined_df = combined_df.loc[combined_df.index.isin(patient_ids)]

    return combined_df