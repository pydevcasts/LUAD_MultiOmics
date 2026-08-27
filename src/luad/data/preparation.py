"""Data preparation utilities for aligned patient-level interim datasets.

This module only aligns samples/patients and constructs interim matrices.
It does NOT perform scaling, imputation, variance filtering, mutual information,
feature selection, or model training.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.luad.data.harmonizer import (
    extract_patient_id,
    is_tumor_sample,
    read_header_columns,
)
from src.luad.utils.io import ensure_dir


MISSING_ID_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
    "[not available]",
    "not available",
}


def clean_id(value: Any) -> Optional[str]:
    """
    Clean an identifier-like value.
    """
    if value is None:
        return None

    if pd.isna(value):
        return None

    text_value = str(value).strip()

    if text_value.lower() in MISSING_ID_VALUES:
        return None

    return text_value


class FeatureIdManager:
    """
    Create unique feature IDs within one modality.
    """

    def __init__(self) -> None:
        self.assigned_ids: Set[str] = set()
        self.unnamed_count = 0

    def make_id(
        self,
        original_value: Optional[str],
        fallback_value: Optional[str] = None,
    ) -> str:
        original = clean_id(original_value)
        fallback = clean_id(fallback_value)

        if original is None and fallback is None:
            self.unnamed_count += 1
            base = f"unnamed_feature_{self.unnamed_count}"
        elif original is None:
            base = str(fallback)
        else:
            base = original

        candidate = base

        if candidate in self.assigned_ids:
            if fallback is not None:
                fallback_candidate = f"{base}|{fallback}"

                if fallback_candidate not in self.assigned_ids:
                    candidate = fallback_candidate
                else:
                    duplicate_index = 2

                    while f"{base}|dup{duplicate_index}" in self.assigned_ids:
                        duplicate_index += 1

                    candidate = f"{base}|dup{duplicate_index}"
            else:
                duplicate_index = 2

                while f"{base}|dup{duplicate_index}" in self.assigned_ids:
                    duplicate_index += 1

                candidate = f"{base}|dup{duplicate_index}"

        self.assigned_ids.add(candidate)

        return candidate


def _unique_list(values: List[str]) -> List[str]:
    seen = set()
    unique_values = []

    for value in values:
        if value is None:
            continue

        if value not in seen:
            seen.add(value)
            unique_values.append(value)

    return unique_values


def find_file_config(config: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    """
    Find dataset config by key.
    """
    dataset_files = config.get("datasets_files", [])

    for item in dataset_files:
        if str(item.get("key")) == str(dataset_key):
            return item

    raise ValueError(f"Dataset key '{dataset_key}' not found in datasets_files.")


def load_selected_sample_map(
    sample_map_df: pd.DataFrame,
    dataset_key: str,
    stage_patients: Set[str],
) -> Dict[str, str]:
    """
    Return selected sample_barcode -> patient_id mapping for one modality.
    """
    selected_df = sample_map_df[
        (sample_map_df["modality_key"] == dataset_key)
        & (sample_map_df["selected"] == True)
    ]

    selected_df = selected_df[selected_df["patient_id"].isin(stage_patients)]

    return dict(zip(selected_df["sample_barcode"], selected_df["patient_id"]))


def prepare_clinical_dataset(
    config: Dict[str, Any],
    manifest_df: pd.DataFrame,
    output_dir,
    logger=None,
) -> Dict[str, Any]:
    """
    Prepare clinical stage cohort and clinical feature policy.
    """
    output_dir = ensure_dir(output_dir)

    clinical_file_config = find_file_config(config, "clinical_patient")
    clinical_path = clinical_file_config.get("resolved_path")

    clinical_data_config = config.get("clinical_data", {})
    patient_id_column = clinical_data_config.get("patient_id_column", "PATIENT_ID")

    clinical_df = pd.read_csv(
        clinical_path,
        sep="\t",
        comment="#",
        encoding="utf-8-sig",
        low_memory=False,
    )

    if patient_id_column not in clinical_df.columns:
        raise ValueError(
            f"Patient ID column '{patient_id_column}' not found in clinical file."
        )

    clinical_df["patient_id"] = (
        clinical_df[patient_id_column]
        .astype(str)
        .map(extract_patient_id)
    )

    clinical_df = clinical_df.dropna(subset=["patient_id"])
    clinical_df = clinical_df.drop_duplicates(subset="patient_id", keep="first")

    stage_manifest = manifest_df[["patient_id", "target_raw", "stage_group"]].copy()

    clinical_out = stage_manifest.merge(
        clinical_df,
        on="patient_id",
        how="left",
    )

    output_path = output_dir / "clinical_stage_cohort.csv"
    clinical_out.to_csv(output_path, index=False, encoding="utf-8-sig")

    policy = {
        "patient_id_column": patient_id_column,
        "target_column": config.get("task", {}).get("target"),
        "grouped_target_column": config.get("task", {})
        .get("target_preprocessing", {})
        .get("grouped_target_column", "STAGE_GROUP"),
        "leakage_drop_columns": clinical_data_config.get("leakage_drop_columns", []),
        "constant_or_low_value_drop_columns": clinical_data_config.get(
            "constant_or_low_value_drop_columns",
            [],
        ),
    }

    policy_path = output_dir / "clinical_feature_policy.json"

    with policy_path.open("w", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2)

    if logger is not None:
        logger.info(
            f"Clinical stage cohort saved: rows={clinical_out.shape[0]}, "
            f"columns={clinical_out.shape[1]}"
        )

    return {
        "key": "clinical",
        "status": "ok",
        "output_matrix": str(output_path),
        "feature_metadata": str(policy_path),
        "orientation": "patient_by_feature",
        "n_patients": int(clinical_out.shape[0]),
        "n_features": int(clinical_out.shape[1] - 3),
    }


def prepare_matrix_modality(
    config: Dict[str, Any],
    file_config: Dict[str, Any],
    schema: Dict[str, Any],
    sample_map_df: pd.DataFrame,
    stage_patients: Set[str],
    output_dir,
    chunksize: int = 5000,
    logger=None,
) -> Dict[str, Any]:
    """
    Prepare one features x samples matrix modality.
    """
    output_dir = ensure_dir(output_dir)

    key = str(file_config.get("key"))
    modality = str(file_config.get("modality"))
    resolved_path = str(file_config.get("resolved_path", ""))

    result: Dict[str, Any] = {
        "key": key,
        "modality": modality,
        "status": "not_run",
        "output_matrix": None,
        "feature_metadata": None,
        "orientation": schema.get("output_orientation", "patient_by_feature"),
        "n_patients": 0,
        "n_features": 0,
        "errors": [],
    }

    if not resolved_path:
        result["status"] = "missing_path"
        result["errors"].append("resolved_path is empty.")
        return result

    feature_id_column = schema.get("feature_id_column")

    if feature_id_column is None:
        result["status"] = "missing_feature_id_config"
        result["errors"].append(
            f"feature_id_column is not configured for dataset key '{key}'."
        )
        return result

    sample_to_patient = load_selected_sample_map(
        sample_map_df=sample_map_df,
        dataset_key=key,
        stage_patients=stage_patients,
    )

    if not sample_to_patient:
        result["status"] = "no_selected_samples"
        result["errors"].append("No selected samples found in sample_map.csv.")
        return result

    try:
        header_columns, separator = read_header_columns(resolved_path)
    except Exception as exc:
        result["status"] = "header_read_error"
        result["errors"].append(str(exc))
        return result

    header_set = set(header_columns)

    available_samples = [
        sample for sample in sample_to_patient.keys() if sample in header_set
    ]

    if not available_samples:
        result["status"] = "no_matching_sample_columns"
        result["errors"].append("No selected sample columns found in file header.")
        return result

    patients = sorted({sample_to_patient[sample] for sample in available_samples})

    fallback_column = schema.get("feature_id_fallback_column")
    metadata_columns = schema.get("feature_metadata_columns", [])
    feature_name_column = schema.get("feature_name_column")

    feature_columns = [feature_id_column]

    if fallback_column is not None:
        feature_columns.append(fallback_column)

    if feature_name_column is not None:
        metadata_columns = [feature_name_column] + list(metadata_columns)

    feature_columns.extend(metadata_columns)
    feature_columns = _unique_list(feature_columns)

    usecols = feature_columns + available_samples
    usecols = [column for column in usecols if column in header_set]

    prefix = schema.get("feature_prefix", key)
    orientation = schema.get("output_orientation", "patient_by_feature")

    feature_id_manager = FeatureIdManager()

    if orientation == "feature_by_patient":
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            result["status"] = "missing_pyarrow"
            result["errors"].append(
                "pyarrow is required for feature_by_patient methylation output. "
                "Install it with: pip install pyarrow"
            )
            return result

        output_matrix_path = output_dir / f"{key}.feature_by_patient.parquet"
        feature_metadata_path = output_dir / f"{key}.feature_metadata.csv"

        writer = None
        feature_count = 0
        metadata_header_written = False

        try:
            reader = pd.read_csv(
                resolved_path,
                sep=separator,
                usecols=usecols,
                chunksize=chunksize,
                comment="#",
                on_bad_lines="skip",
                low_memory=False,
                encoding="utf-8-sig",
            )

            for chunk in reader:
                if feature_id_column not in chunk.columns:
                    continue

                feature_ids: List[str] = []
                metadata_rows: List[Dict[str, Any]] = []

                for _, row in chunk.iterrows():
                    original_id = clean_id(row.get(feature_id_column))

                    fallback_id = None

                    if fallback_column is not None:
                        fallback_id = clean_id(row.get(fallback_column))

                    unique_feature_id = feature_id_manager.make_id(
                        original_value=original_id,
                        fallback_value=fallback_id,
                    )

                    prefixed_feature_id = f"{prefix}:{unique_feature_id}"
                    feature_ids.append(prefixed_feature_id)

                    metadata_row = {
                        "feature_id": prefixed_feature_id,
                        "unique_feature_id": unique_feature_id,
                        "original_feature_id": original_id,
                        "fallback_id": fallback_id,
                        "modality_key": key,
                        "modality": modality,
                    }

                    for metadata_column in metadata_columns:
                        if metadata_column in chunk.columns:
                            metadata_row[metadata_column] = row.get(metadata_column)

                    metadata_rows.append(metadata_row)

                data = chunk[available_samples].copy()

                for sample_column in available_samples:
                    data[sample_column] = pd.to_numeric(
                        data[sample_column],
                        errors="coerce",
                    )

                data = data.astype("float32")
                data.insert(0, "feature_id", feature_ids)

                data = data.rename(columns=sample_to_patient)
                data = data[["feature_id"] + patients]

                table = pa.Table.from_pandas(data, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(output_matrix_path, table.schema)

                writer.write_table(table)

                metadata_df = pd.DataFrame(metadata_rows)

                metadata_df.to_csv(
                    feature_metadata_path,
                    mode="a",
                    header=not metadata_header_written,
                    index=False,
                    encoding="utf-8-sig",
                )

                metadata_header_written = True
                feature_count += len(feature_ids)

                if logger is not None:
                    logger.info(
                        f"Processed {feature_count} features for {key}."
                    )

            if writer is not None:
                writer.close()

        except Exception as exc:
            result["status"] = "processing_error"
            result["errors"].append(str(exc))
            return result

        result["status"] = "ok"
        result["output_matrix"] = str(output_matrix_path)
        result["feature_metadata"] = str(feature_metadata_path)
        result["n_patients"] = len(patients)
        result["n_features"] = feature_count

        return result

    # Default orientation: patient_by_feature
    output_matrix_path = output_dir / f"{key}.patient_features.parquet"
    feature_metadata_path = output_dir / f"{key}.feature_metadata.csv"

    parts: List[pd.DataFrame] = []
    feature_metadata_rows: List[Dict[str, Any]] = []
    feature_count = 0

    try:
        reader = pd.read_csv(
            resolved_path,
            sep=separator,
            usecols=usecols,
            chunksize=chunksize,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig",
        )

        for chunk in reader:
            if feature_id_column not in chunk.columns:
                continue

            feature_ids: List[str] = []

            for _, row in chunk.iterrows():
                original_id = clean_id(row.get(feature_id_column))

                fallback_id = None

                if fallback_column is not None:
                    fallback_id = clean_id(row.get(fallback_column))

                unique_feature_id = feature_id_manager.make_id(
                    original_value=original_id,
                    fallback_value=fallback_id,
                )

                prefixed_feature_id = f"{prefix}:{unique_feature_id}"
                feature_ids.append(prefixed_feature_id)

                metadata_row = {
                    "feature_id": prefixed_feature_id,
                    "unique_feature_id": unique_feature_id,
                    "original_feature_id": original_id,
                    "fallback_id": fallback_id,
                    "modality_key": key,
                    "modality": modality,
                }

                for metadata_column in metadata_columns:
                    if metadata_column in chunk.columns:
                        metadata_row[metadata_column] = row.get(metadata_column)

                feature_metadata_rows.append(metadata_row)

            data = chunk[available_samples].copy()

            for sample_column in available_samples:
                data[sample_column] = pd.to_numeric(
                    data[sample_column],
                    errors="coerce",
                )

            data = data.astype("float32")
            data.index = feature_ids

            data = data.rename(columns=sample_to_patient)
            data = data[patients]

            patient_chunk = data.transpose()
            patient_chunk.index.name = "patient_id"

            parts.append(patient_chunk)
            feature_count += len(feature_ids)

            if logger is not None:
                logger.info(f"Processed {feature_count} features for {key}.")

    except Exception as exc:
        result["status"] = "processing_error"
        result["errors"].append(str(exc))
        return result

    if not parts:
        result["status"] = "no_rows_processed"
        result["errors"].append("No data rows were processed.")
        return result

    patient_matrix = pd.concat(parts, axis=1)
    patient_matrix.index.name = "patient_id"

    patient_matrix.to_parquet(output_matrix_path, index=True)

    feature_metadata_df = pd.DataFrame(feature_metadata_rows)
    feature_metadata_df.to_csv(
        feature_metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    result["status"] = "ok"
    result["output_matrix"] = str(output_matrix_path)
    result["feature_metadata"] = str(feature_metadata_path)
    result["n_patients"] = int(patient_matrix.shape[0])
    result["n_features"] = int(patient_matrix.shape[1])

    return result


def prepare_mutation_modality(
    config: Dict[str, Any],
    file_config: Dict[str, Any],
    manifest_df: pd.DataFrame,
    output_dir,
    chunksize: int = 100000,
    logger=None,
) -> Dict[str, Any]:
    """
    Prepare binary mutation matrix: Patient x Gene.
    """
    output_dir = ensure_dir(output_dir)

    key = str(file_config.get("key", "mutations"))
    resolved_path = str(file_config.get("resolved_path", ""))

    result: Dict[str, Any] = {
        "key": key,
        "modality": "mutation",
        "status": "not_run",
        "output_matrix": None,
        "feature_metadata": None,
        "orientation": "patient_by_feature",
        "n_patients": 0,
        "n_features": 0,
        "errors": [],
    }

    if not resolved_path:
        result["status"] = "missing_path"
        result["errors"].append("resolved_path is empty.")
        return result

    mutation_config = config.get("mutation_data", {})

    sample_column = mutation_config.get("sample_column", "Tumor_Sample_Barcode")
    gene_column = mutation_config.get("gene_column", "Hugo_Symbol")
    variant_column = mutation_config.get(
        "variant_classification_column",
        "Variant_Classification",
    )

    excluded_variant_classifications = set(
        mutation_config.get("exclude_variant_classifications", ["Silent"])
    )

    try:
        header_columns, separator = read_header_columns(resolved_path)
    except Exception as exc:
        result["status"] = "header_read_error"
        result["errors"].append(str(exc))
        return result

    header_set = set(header_columns)

    if sample_column not in header_set:
        result["status"] = "missing_sample_column"
        result["errors"].append(f"Sample column '{sample_column}' not found.")
        return result

    if gene_column not in header_set:
        result["status"] = "missing_gene_column"
        result["errors"].append(f"Gene column '{gene_column}' not found.")
        return result

    usecols = [sample_column, gene_column]

    if variant_column in header_set:
        usecols.append(variant_column)

    stage_patients = set(manifest_df["patient_id"].astype(str))

    patient_gene_sets: Dict[str, Set[str]] = {}
    all_genes: Set[str] = set()

    try:
        reader = pd.read_csv(
            resolved_path,
            sep=separator,
            usecols=usecols,
            dtype=str,
            chunksize=chunksize,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig",
        )

        processed_rows = 0

        for chunk in reader:
            if sample_column not in chunk.columns or gene_column not in chunk.columns:
                continue

            if (
                variant_column in chunk.columns
                and excluded_variant_classifications
            ):
                variant_values = chunk[variant_column].astype(str).str.strip()
                chunk = chunk[~variant_values.isin(excluded_variant_classifications)]

            if chunk.empty:
                continue

            patient_series = chunk[sample_column].map(extract_patient_id)
            tumor_mask = chunk[sample_column].map(is_tumor_sample)

            valid_mask = (
                patient_series.notna()
                & tumor_mask
                & patient_series.isin(stage_patients)
                & chunk[gene_column].notna()
            )

            valid_patients = patient_series[valid_mask]
            valid_genes = chunk.loc[valid_mask, gene_column]

            for patient_id, gene_value in zip(valid_patients, valid_genes):
                gene = clean_id(gene_value)

                if gene is None:
                    continue

                patient_gene_sets.setdefault(patient_id, set()).add(gene)
                all_genes.add(gene)

            processed_rows += len(chunk)

            if logger is not None and processed_rows % 200000 < chunksize:
                logger.info(
                    f"Mutation processing: processed_rows={processed_rows}, "
                    f"genes_so_far={len(all_genes)}"
                )

    except Exception as exc:
        result["status"] = "processing_error"
        result["errors"].append(str(exc))
        return result

    if not all_genes:
        result["status"] = "no_mutations_processed"
        result["errors"].append("No mutation genes were extracted.")
        return result

    patients = sorted(stage_patients)
    genes = sorted(all_genes)
    feature_ids = [f"mut:{gene}" for gene in genes]

    mutation_matrix = pd.DataFrame(
        0,
        index=patients,
        columns=feature_ids,
        dtype="uint8",
    )

    mutation_matrix.index.name = "patient_id"

    for patient_id, gene_set in patient_gene_sets.items():
        if patient_id not in mutation_matrix.index:
            continue

        patient_columns = [f"mut:{gene}" for gene in sorted(gene_set)]
        mutation_matrix.loc[patient_id, patient_columns] = 1

    output_matrix_path = output_dir / f"{key}.patient_features.parquet"
    mutation_matrix.to_parquet(output_matrix_path, index=True)

    feature_metadata_rows = []

    for gene in genes:
        feature_metadata_rows.append(
            {
                "feature_id": f"mut:{gene}",
                "unique_feature_id": gene,
                "original_feature_id": gene,
                "fallback_id": None,
                "modality_key": key,
                "modality": "mutation",
            }
        )

    feature_metadata_path = output_dir / f"{key}.feature_metadata.csv"

    pd.DataFrame(feature_metadata_rows).to_csv(
        feature_metadata_path,
        index=False,
        encoding="utf-8-sig",
    )

    result["status"] = "ok"
    result["output_matrix"] = str(output_matrix_path)
    result["feature_metadata"] = str(feature_metadata_path)
    result["n_patients"] = int(mutation_matrix.shape[0])
    result["n_features"] = int(mutation_matrix.shape[1])

    return result


def run_data_preparation(
    config: Dict[str, Any],
    logger=None,
    output_dir: Optional[str] = None,
    prepare_keys: Optional[List[str]] = None,
    chunksize: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run data preparation for selected dataset keys.
    """
    project_root = config.get("runtime", {}).get("project_root", ".")
    project_root_path = ensure_dir(project_root)

    interim_dir = output_dir or config.get("paths", {}).get(
        "interim_dir",
        "data/interim",
    )

    interim_path = ensure_dir(project_root_path / interim_dir)

    cohort_dir = project_root_path / "artifacts" / "cohort"

    manifest_path = cohort_dir / "patient_manifest.csv"
    sample_map_path = cohort_dir / "sample_map.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Patient manifest not found: {manifest_path}. Run build_cohort.py first."
        )

    if not sample_map_path.exists():
        raise FileNotFoundError(
            f"Sample map not found: {sample_map_path}. Run build_cohort.py first."
        )

    manifest_df = pd.read_csv(manifest_path)
    sample_map_df = pd.read_csv(sample_map_path)

    stage_patients = set(manifest_df["patient_id"].astype(str))

    preparation_config = config.get("data_preparation", {})

    if prepare_keys is None:
        prepare_keys = preparation_config.get(
            "prepare_keys",
            [
                "clinical",
                "mrna_rsem",
                "mirna",
                "methylation",
                "cna_raw",
                "rppa_raw",
                "rppa_zscores",
                "mutations",
            ],
        )

    if chunksize is None:
        chunksize = int(preparation_config.get("chunksize", 5000))

    mutation_chunksize = int(preparation_config.get("mutation_chunksize", 100000))

    feature_schemas = preparation_config.get("feature_schemas", {})

    results: List[Dict[str, Any]] = []

    for dataset_key in prepare_keys:
        if logger is not None:
            logger.info(f"Preparing dataset key: {dataset_key}")

        if dataset_key == "clinical":
            result = prepare_clinical_dataset(
                config=config,
                manifest_df=manifest_df,
                output_dir=interim_path,
                logger=logger,
            )

            results.append(result)
            continue

        if dataset_key == "mutations":
            file_config = find_file_config(config, dataset_key)

            result = prepare_mutation_modality(
                config=config,
                file_config=file_config,
                manifest_df=manifest_df,
                output_dir=interim_path,
                chunksize=mutation_chunksize,
                logger=logger,
            )

            results.append(result)
            continue

        file_config = find_file_config(config, dataset_key)
        schema = feature_schemas.get(dataset_key, {})

        result = prepare_matrix_modality(
            config=config,
            file_config=file_config,
            schema=schema,
            sample_map_df=sample_map_df,
            stage_patients=stage_patients,
            output_dir=interim_path,
            chunksize=chunksize,
            logger=logger,
        )

        results.append(result)

        if logger is not None:
            logger.info(
                f"Finished {dataset_key}: status={result['status']}, "
                f"patients={result['n_patients']}, features={result['n_features']}"
            )

    summary_path = interim_path / "preparation_summary.json"

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)

    if logger is not None:
        logger.info(f"Preparation summary saved to: {summary_path}")

    return results