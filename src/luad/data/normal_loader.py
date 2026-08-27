"""Load and prepare normal samples for tumor vs normal classification."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.luad.data.harmonizer import read_header_columns


def load_normals_matrix(
    normals_path: Path,
    logger=None,
) -> Tuple[pd.DataFrame, str]:
    """
    Load normal samples matrix.

    Returns
    -------
    tuple of (normals_matrix, feature_id_column)
        normals_matrix: samples x features DataFrame (with unique feature columns)
        feature_id_column: name of the feature identifier column used
    """
    normals_path = Path(normals_path)

    if not normals_path.exists():
        raise FileNotFoundError(f"Normals file not found: {normals_path}")

    if logger is not None:
        logger.info(f"Loading normals from: {normals_path}")

    header_columns, separator = read_header_columns(str(normals_path))

    normals_raw = pd.read_csv(
        normals_path,
        sep=separator,
        comment="#",
        encoding="utf-8-sig",
        low_memory=False,
    )

    if logger is not None:
        logger.info(f"Normals raw shape: {normals_raw.shape}")
        logger.info(f"First 5 columns: {list(normals_raw.columns[:5])}")

    non_sample_columns = []
    sample_columns = []

    for col in normals_raw.columns:
        col_str = str(col)
        if col_str.startswith("TCGA-") or col_str.startswith("tcga-"):
            sample_columns.append(col)
        else:
            non_sample_columns.append(col)

    if logger is not None:
        logger.info(f"Non-sample columns (feature metadata): {non_sample_columns}")
        logger.info(f"Sample columns count: {len(sample_columns)}")

    feature_id_column = non_sample_columns[0] if non_sample_columns else normals_raw.columns[0]

    if logger is not None:
        logger.info(f"Using feature ID column: {feature_id_column}")

    normals_matrix = normals_raw.set_index(feature_id_column)[sample_columns].T
    normals_matrix.index.name = "sample_id"
    normals_matrix.columns.name = None

    normals_matrix = normals_matrix.apply(pd.to_numeric, errors="coerce")

    duplicate_features = normals_matrix.columns[normals_matrix.columns.duplicated()]

    if len(duplicate_features) > 0:
        if logger is not None:
            logger.info(
                f"Found {len(duplicate_features)} duplicate feature columns in normals. "
                f"Aggregating by mean."
            )

        normals_matrix = normals_matrix.T
        normals_matrix.index = normals_matrix.index.astype(str)
        normals_matrix = normals_matrix.groupby(level=0).mean()
        normals_matrix = normals_matrix.T

        if logger is not None:
            logger.info(
                f"Normals shape after deduplication: {normals_matrix.shape}"
            )

    if logger is not None:
        logger.info(f"Normals matrix final shape: {normals_matrix.shape}")

    return normals_matrix, feature_id_column


def strip_modality_prefix(feature_name: str) -> str:
    """
    Remove modality prefix from feature name.
    e.g., 'mrna:TP53' -> 'TP53'
          'mrna:TP53|7157' -> 'TP53'
    """
    name = str(feature_name)

    if ":" in name:
        name = name.split(":", 1)[1]

    if "|" in name:
        name = name.split("|", 1)[0]

    return name


def align_normals_with_tumor(
    normals_matrix: pd.DataFrame,
    tumor_matrix: pd.DataFrame,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Align normal samples with tumor features by stripping modality prefixes.

    Returns
    -------
    tuple of (normals_aligned, tumor_aligned, common_features)
    """
    tumor_original_columns = list(tumor_matrix.columns)

    stripped_to_original = {}

    for col in tumor_original_columns:
        stripped = strip_modality_prefix(col)
        if stripped not in stripped_to_original:
            stripped_to_original[stripped] = col

    if logger is not None:
        original_tumor_feature_count = len(tumor_original_columns)
        dedup_tumor_count = len(stripped_to_original)
        if original_tumor_feature_count != dedup_tumor_count:
            logger.info(
                f"Tumor features deduplicated: {original_tumor_feature_count} -> {dedup_tumor_count}"
            )

    normal_features = set(normals_matrix.columns.astype(str))
    tumor_stripped_features = set(stripped_to_original.keys())

    common_stripped = sorted(normal_features & tumor_stripped_features)

    if logger is not None:
        logger.info(f"Normal features count: {len(normal_features)}")
        logger.info(f"Tumor stripped features count: {len(tumor_stripped_features)}")
        logger.info(f"Common features after alignment: {len(common_stripped)}")

    if len(common_stripped) == 0:
        normal_sample = sorted(normal_features)[:10]
        tumor_sample = sorted(tumor_stripped_features)[:10]

        raise ValueError(
            f"No common features found.\n"
            f"Normal features sample: {normal_sample}\n"
            f"Tumor stripped features sample: {tumor_sample}\n"
            f"Check feature naming conventions."
        )

    normals_aligned = normals_matrix[common_stripped].copy()

    tumor_rename_map = {}
    for stripped_name in common_stripped:
        original_col = stripped_to_original[stripped_name]
        tumor_rename_map[original_col] = stripped_name

    tumor_aligned = tumor_matrix[list(tumor_rename_map.keys())].copy()
    tumor_aligned = tumor_aligned.rename(columns=tumor_rename_map)
    tumor_aligned = tumor_aligned[common_stripped]

    if logger is not None:
        logger.info(f"Aligned tumor shape: {tumor_aligned.shape}")
        logger.info(f"Aligned normals shape: {normals_aligned.shape}")

    return normals_aligned, tumor_aligned, common_stripped