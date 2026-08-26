"""Patient/sample harmonization utilities.

This module builds a patient-level manifest from clinical data and omics files.
It does not train models and does not perform preprocessing.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from src.luad.data.target import map_target_series

PATIENT_EXTRACT_PATTERN = r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})"
PATIENT_REGEX = re.compile(PATIENT_EXTRACT_PATTERN, re.IGNORECASE)

SAMPLE_TYPE_PATTERN = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-([0-9]{2})"
SAMPLE_TYPE_REGEX = re.compile(SAMPLE_TYPE_PATTERN, re.IGNORECASE)

SAMPLE_COLUMN_KEYWORDS = [
    "tumor_sample_barcode",
    "sample_barcode",
    "tumor_sample",
    "sample_id",
    "sample",
    "barcode",
    "case_id",
    "participant",
]

GENE_COLUMN_KEYWORDS = [
    "hugo_symbol",
    "gene_symbol",
    "symbol",
    "gene",
    "hugo",
]

PREFERRED_SAMPLE_COLUMNS = [
    "tumor_sample_barcode",
    "tumor_sample_uuid",
    "sample_barcode",
    "sample_id",
    "barcode",
]

PREFERRED_GENE_COLUMNS = [
    "hugo_symbol",
    "symbol",
    "gene_symbol",
    "gene",
]


def _detect_separator(line: str) -> str:
    if not line:
        return "\t"

    candidates = {
        "\t": line.count("\t"),
        ",": line.count(","),
        ";": line.count(";"),
        "|": line.count("|"),
    }

    separator = max(candidates, key=lambda key: candidates[key])

    if candidates[separator] > 0:
        return separator

    if " " in line.strip():
        return "\s+"

    return "\t"


def read_header_columns(path: str) -> tuple[List[str], str]:
    """
    Read only the first non-comment table line and return columns + separator.
    This is memory-safe for wide files.
    """
    with open(path, "r", encoding="utf-8-sig", errors="ignore") as handle:
        for line in handle:
            stripped_line = line.strip()

            if not stripped_line:
                continue

            if stripped_line.startswith("#"):
                continue

            separator = _detect_separator(stripped_line)

            if separator == "\s+":
                columns = stripped_line.split()
            else:
                columns = stripped_line.split(separator)

            columns = [str(column).strip().strip('"').strip("'") for column in columns]

            return columns, separator

    return [], "\t"


def extract_patient_id(barcode: Any) -> Optional[str]:
    """
    Extract patient-level TCGA ID from a barcode or ID string.

    Example:
    TCGA-05-4244-01 -> TCGA-05-4244
    """
    if barcode is None:
        return None

    match = PATIENT_REGEX.search(str(barcode))

    if match is None:
        return None

    return match.group(1).upper()


def is_tumor_sample(barcode: Any) -> bool:
    """
    Decide whether a TCGA sample barcode is tumor-like.

    TCGA sample codes:
    - 01-09: tumor
    - 10-19: normal
    - others: treated cautiously

    If no sample suffix exists, we assume it may be a patient-level tumor ID.
    """
    if barcode is None:
        return False

    match = SAMPLE_TYPE_REGEX.search(str(barcode))

    if match is None:
        return True

    try:
        sample_code = int(match.group(1))
    except Exception:
        return True

    return sample_code < 10


def choose_best_sample(samples: List[str]) -> str:
    """
    Choose one representative sample for a patient.

    Preference:
    1. sample containing -01 tumor code
    2. first sorted sample
    """
    if not samples:
        raise ValueError("Cannot choose best sample from empty sample list.")

    for sample in sorted(samples):
        if "-01" in sample:
            return sample

    return sorted(samples)[0]


def build_sample_map_from_columns(
    columns: List[str],
    include_tumor_only: bool = True,
) -> Dict[str, List[str]]:
    """
    Build patient_id -> sample_barcodes mapping from matrix column names.
    """
    sample_map: Dict[str, List[str]] = {}

    for column in columns:
        column_str = str(column).strip()

        if not PATIENT_REGEX.search(column_str):
            continue

        if include_tumor_only and not is_tumor_sample(column_str):
            continue

        patient_id = extract_patient_id(column_str)

        if patient_id is None:
            continue

        sample_map.setdefault(patient_id, []).append(column_str)

    return sample_map


def harmonize_matrix_modality(
    file_config: Dict[str, Any],
    include_tumor_only: bool = True,
) -> Dict[str, Any]:
    """
    Harmonize one features x samples matrix file using only its header.
    """
    key = str(file_config.get("key", "unknown"))
    modality = str(file_config.get("modality", "unknown"))
    resolved_path = str(file_config.get("resolved_path", ""))

    result: Dict[str, Any] = {
        "key": key,
        "modality": modality,
        "resolved_path": resolved_path,
        "status": "not_run",
        "separator": None,
        "total_columns": None,
        "tcga_columns": 0,
        "patient_count": 0,
        "multi_sample_patient_count": 0,
        "sample_map": {},
        "selected_samples": {},
        "errors": [],
    }

    if not resolved_path:
        result["status"] = "missing_path"
        result["errors"].append("resolved_path is empty.")
        return result

    try:
        columns, separator = read_header_columns(resolved_path)
    except Exception as exc:
        result["status"] = "read_error"
        result["errors"].append(str(exc))
        return result

    result["separator"] = separator
    result["total_columns"] = len(columns)

    if not columns:
        result["status"] = "no_header"
        result["errors"].append("No header line found.")
        return result

    sample_map = build_sample_map_from_columns(
        columns=columns,
        include_tumor_only=include_tumor_only,
    )

    selected_samples: Dict[str, str] = {}

    for patient_id, samples in sample_map.items():
        selected_samples[patient_id] = choose_best_sample(samples)

    result["status"] = "ok"
    result["tcga_columns"] = sum(len(samples) for samples in sample_map.values())
    result["patient_count"] = len(sample_map)
    result["multi_sample_patient_count"] = sum(
        1 for samples in sample_map.values() if len(samples) > 1
    )
    result["sample_map"] = sample_map
    result["selected_samples"] = selected_samples

    return result


def load_clinical_cohort(
    clinical_path: str,
    patient_id_column: str,
    target_column: Optional[str],
    mapping: Optional[Dict[str, str]] = None,
    uppercase: bool = True,
    strip: bool = True,
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """
    Load clinical data and build a patient-level target table.
    """
    clinical_df = pd.read_csv(
        clinical_path,
        sep="\t",
        comment="#",
        encoding="utf-8-sig",
        low_memory=False,
    )

    if patient_id_column not in clinical_df.columns:
        raise ValueError(
            f"Patient ID column '{patient_id_column}' not found in clinical data."
        )

    clinical_df["patient_id"] = (
        clinical_df[patient_id_column]
        .astype(str)
        .map(extract_patient_id)
    )

    invalid_patient_rows = clinical_df["patient_id"].isna().sum()

    if invalid_patient_rows > 0:
        raise ValueError(
            f"Found {invalid_patient_rows} clinical rows without valid TCGA patient ID."
        )

    duplicated_patient_count = int(clinical_df["patient_id"].duplicated().sum())

    if duplicated_patient_count > 0:
        clinical_df = clinical_df.drop_duplicates(subset="patient_id", keep="first")

    if target_column is None:
        clinical_df["target_raw"] = None
        clinical_df["stage_group"] = None
    else:
        if target_column not in clinical_df.columns:
            raise ValueError(
                f"Target column '{target_column}' not found in clinical data."
            )

        clinical_df["target_raw"] = clinical_df[target_column]

        clinical_df["stage_group"] = map_target_series(
            series=clinical_df[target_column],
            mapping=mapping,
            uppercase=uppercase,
            strip=strip,
        )

    if drop_missing_target:
        clinical_df = clinical_df.dropna(subset=["stage_group"])

    clinical_df = clinical_df[["patient_id", "target_raw", "stage_group"]]

    return clinical_df


def detect_mutation_schema(
    mutation_path: str,
    preview_rows: int = 2000,
    sample_column_override: Optional[str] = None,
    gene_column_override: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inspect mutation file to find gene and sample columns.

    This version scans all columns as strings and does not rely on dtype == object.
    """
    schema: Dict[str, Any] = {
        "path": mutation_path,
        "status": "not_run",
        "separator": None,
        "columns_first_50": [],
        "gene_column": None,
        "sample_column": None,
        "sample_match_counts": {},
        "patient_match_counts": {},
        "preview_rows_loaded": 0,
        "errors": [],
    }

    try:
        columns, separator = read_header_columns(mutation_path)
    except Exception as exc:
        schema["status"] = "read_error"
        schema["errors"].append(str(exc))
        return schema

    schema["separator"] = separator
    schema["columns_first_50"] = columns[:50]

    if not columns:
        schema["status"] = "no_header"
        schema["errors"].append("No header line found.")
        return schema

    lower_header_lookup = {str(column).lower(): str(column) for column in columns}

    if sample_column_override is not None:
        override_lower = str(sample_column_override).lower()

        if override_lower in lower_header_lookup:
            schema["sample_column"] = lower_header_lookup[override_lower]
        else:
            schema["errors"].append(
                f"sample_column_override '{sample_column_override}' not found in header."
            )

    if gene_column_override is not None:
        override_lower = str(gene_column_override).lower()

        if override_lower in lower_header_lookup:
            schema["gene_column"] = lower_header_lookup[override_lower]
        else:
            schema["errors"].append(
                f"gene_column_override '{gene_column_override}' not found in header."
            )

    try:
        preview_df = pd.read_csv(
            mutation_path,
            sep=separator,
            nrows=preview_rows,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        if schema["sample_column"] is not None:
            schema["status"] = "ok_override_but_preview_failed"
            return schema

        schema["status"] = "preview_error"
        schema["errors"].append(str(exc))
        return schema

    schema["preview_rows_loaded"] = int(preview_df.shape[0])

    lower_columns = {str(column): str(column).lower() for column in preview_df.columns}

    gene_candidates = [
        column
        for column, lower_column in lower_columns.items()
        if any(keyword in lower_column for keyword in GENE_COLUMN_KEYWORDS)
    ]

    sample_candidates = [
        column
        for column, lower_column in lower_columns.items()
        if any(keyword in lower_column for keyword in SAMPLE_COLUMN_KEYWORDS)
    ]

    sample_match_counts: Dict[str, int] = {}
    patient_match_counts: Dict[str, int] = {}

    for column in preview_df.columns:
        non_null_values = preview_df[column].dropna()

        if non_null_values.empty:
            continue

        values = non_null_values.astype(str)

        sample_matches = int(
            values.str.contains(
                SAMPLE_TYPE_PATTERN,
                flags=re.IGNORECASE,
                regex=True,
            ).sum()
        )

        patient_matches = int(
            values.str.contains(
                PATIENT_EXTRACT_PATTERN,
                flags=re.IGNORECASE,
                regex=True,
            ).sum()
        )

        if sample_matches > 0:
            sample_match_counts[str(column)] = sample_matches

        if patient_matches > 0:
            patient_match_counts[str(column)] = patient_matches

    schema["sample_match_counts"] = sample_match_counts
    schema["patient_match_counts"] = patient_match_counts

    if schema["sample_column"] is None:
        sample_column: Optional[str] = None

        lower_preview_lookup = {
            str(column).lower(): str(column) for column in preview_df.columns
        }

        for preferred_sample_column in PREFERRED_SAMPLE_COLUMNS:
            actual_column = lower_preview_lookup.get(preferred_sample_column)

            if actual_column is None:
                continue

            if (
                sample_match_counts.get(actual_column, 0) > 0
                or patient_match_counts.get(actual_column, 0) > 0
            ):
                sample_column = actual_column
                break

        if sample_column is None:
            for candidate in sample_candidates:
                if sample_match_counts.get(candidate, 0) > 0:
                    sample_column = candidate
                    break

        if sample_column is None and sample_match_counts:
            sample_column = max(
                sample_match_counts,
                key=lambda key: sample_match_counts[key],
            )

        if sample_column is None:
            for candidate in sample_candidates:
                if patient_match_counts.get(candidate, 0) > 0:
                    sample_column = candidate
                    break

        if sample_column is None and patient_match_counts:
            sample_column = max(
                patient_match_counts,
                key=lambda key: patient_match_counts[key],
            )

        schema["sample_column"] = sample_column

    if schema["gene_column"] is None:
        gene_column: Optional[str] = None

        lower_preview_lookup = {
            str(column).lower(): str(column) for column in preview_df.columns
        }

        for preferred_gene_column in PREFERRED_GENE_COLUMNS:
            actual_column = lower_preview_lookup.get(preferred_gene_column)

            if actual_column is not None:
                gene_column = actual_column
                break

        if gene_column is None and gene_candidates:
            gene_column = gene_candidates[0]

        schema["gene_column"] = gene_column

    schema["status"] = "ok"

    return schema


def scan_mutation_patients(
    mutation_path: str,
    separator: str,
    sample_column: str,
    include_tumor_only: bool = True,
    chunksize: int = 100000,
) -> Set[str]:
    """
    Stream mutation file and collect patient IDs with tumor mutations.
    """
    patient_ids: Set[str] = set()

    reader = pd.read_csv(
        mutation_path,
        sep=separator,
        usecols=[sample_column],
        dtype=str,
        comment="#",
        chunksize=chunksize,
        on_bad_lines="skip",
        low_memory=False,
        encoding="utf-8-sig",
    )

    for chunk in reader:
        if sample_column not in chunk.columns:
            continue

        values = chunk[sample_column].dropna().astype(str)

        if values.empty:
            continue

        patients = values.str.extract(
            PATIENT_EXTRACT_PATTERN,
            expand=False,
            flags=re.IGNORECASE,
        )

        if include_tumor_only:
            sample_types = values.str.extract(
                SAMPLE_TYPE_PATTERN,
                expand=False,
                flags=re.IGNORECASE,
            )

            sample_codes = pd.to_numeric(sample_types, errors="coerce")
            tumor_mask = sample_codes.isna() | (sample_codes < 10)

            patients = patients[tumor_mask]

        patients = patients.dropna().str.upper()

        if not patients.empty:
            patient_ids.update(patients.unique().tolist())

    return patient_ids


def build_cohort(
    config: Dict[str, Any],
    logger: Optional[Any] = None,
    skip_mutation: bool = False,
    mutation_preview_rows: Optional[int] = None,
    mutation_chunksize: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Build patient-level cohort manifest from clinical and omics datasets.
    """
    dataset_files = config.get("datasets_files", [])
    harmonization_config = config.get("harmonization", {})
    clinical_data_config = config.get("clinical_data", {})
    mutation_data_config = config.get("mutation_data", {})
    task_config = config.get("task", {})

    include_tumor_only = bool(harmonization_config.get("include_tumor_only", True))

    if skip_mutation:
        scan_mutation_full = False
    else:
        scan_mutation_full = bool(harmonization_config.get("scan_mutation_full", True))

    if mutation_preview_rows is None:
        mutation_preview_rows = int(harmonization_config.get("mutation_preview_rows", 2000))

    if mutation_chunksize is None:
        mutation_chunksize = int(harmonization_config.get("mutation_chunksize", 100000))

    clinical_file_config = None
    mutation_file_config = None
    matrix_file_configs: List[Dict[str, Any]] = []

    for item in dataset_files:
        modality = str(item.get("modality", ""))

        if modality == "clinical":
            clinical_file_config = item
        elif modality == "mutation":
            mutation_file_config = item
        elif modality in {"mrna", "mirna", "methylation", "cna", "rppa"}:
            if item.get("external_validation_only", False):
                continue

            if modality == "normals_reference":
                continue

            matrix_file_configs.append(item)

    if clinical_file_config is None:
        raise ValueError("No clinical dataset found in config.")

    clinical_path = str(clinical_file_config.get("resolved_path", ""))

    patient_id_column = (
        clinical_data_config.get("patient_id_column")
        or config.get("identifiers", {}).get("patient_id_column")
        or "PATIENT_ID"
    )

    target_column = task_config.get("target")
    target_preprocessing = task_config.get("target_preprocessing", {})
    mapping = target_preprocessing.get("mapping")
    uppercase = bool(target_preprocessing.get("uppercase", True))
    strip = bool(target_preprocessing.get("strip", True))
    drop_missing_target = bool(task_config.get("drop_missing_target", True))

    if logger is not None:
        logger.info(f"Loading clinical cohort from: {clinical_path}")
        logger.info(f"Patient ID column: {patient_id_column}")
        logger.info(f"Target column: {target_column}")

    clinical_manifest = load_clinical_cohort(
        clinical_path=clinical_path,
        patient_id_column=patient_id_column,
        target_column=target_column,
        mapping=mapping,
        uppercase=uppercase,
        strip=strip,
        drop_missing_target=drop_missing_target,
    )

    if logger is not None:
        logger.info(
            f"Clinical cohort after target filtering: {clinical_manifest.shape[0]} patients"
        )

    matrix_results: List[Dict[str, Any]] = []

    for file_config in matrix_file_configs:
        key = file_config.get("key")

        if logger is not None:
            logger.info(f"Harmonizing matrix modality: {key}")

        result = harmonize_matrix_modality(
            file_config=file_config,
            include_tumor_only=include_tumor_only,
        )

        matrix_results.append(result)

        if logger is not None:
            logger.info(
                f"Modality {key}: status={result['status']}, "
                f"patients={result['patient_count']}, "
                f"tcga_columns={result['tcga_columns']}"
            )

    manifest = clinical_manifest.copy()

    sample_map_rows: List[Dict[str, Any]] = []

    for result in matrix_results:
        key = result["key"]
        sample_map: Dict[str, List[str]] = result.get("sample_map", {})
        selected_samples: Dict[str, str] = result.get("selected_samples", {})

        manifest[f"has_{key}"] = manifest["patient_id"].isin(sample_map.keys())

        manifest[f"{key}_sample_count"] = manifest["patient_id"].map(
            lambda patient_id: len(sample_map.get(patient_id, []))
        )

        for patient_id, samples in sample_map.items():
            selected_sample = selected_samples.get(patient_id)

            for sample in samples:
                sample_map_rows.append(
                    {
                        "modality_key": key,
                        "modality": result.get("modality"),
                        "patient_id": patient_id,
                        "sample_barcode": sample,
                        "selected": sample == selected_sample,
                        "tumor": is_tumor_sample(sample),
                    }
                )

    mutation_schema: Dict[str, Any] = {}
    mutation_patient_ids: Set[str] = set()

    if mutation_file_config is not None:
        mutation_path = str(mutation_file_config.get("resolved_path", ""))

        if logger is not None:
            logger.info(f"Inspecting mutation schema: {mutation_path}")

        mutation_schema = detect_mutation_schema(
            mutation_path=mutation_path,
            preview_rows=mutation_preview_rows,
            sample_column_override=mutation_data_config.get("sample_column"),
            gene_column_override=mutation_data_config.get("gene_column"),
        )

        sample_column = mutation_schema.get("sample_column")
        separator = mutation_schema.get("separator", "\t")

        if sample_column is None:
            mutation_schema["status"] = "sample_column_not_found"
            mutation_schema["errors"].append(
                "Could not detect a TCGA sample column in mutation file."
            )

            if logger is not None:
                logger.warning("Mutation sample column not found.")
        else:
            if logger is not None:
                logger.info(f"Mutation sample column: {sample_column}")
                logger.info(f"Mutation gene column: {mutation_schema.get('gene_column')}")

            if scan_mutation_full:
                if logger is not None:
                    logger.info("Scanning full mutation file for patient IDs.")

                mutation_patient_ids = scan_mutation_patients(
                    mutation_path=mutation_path,
                    separator=separator,
                    sample_column=sample_column,
                    include_tumor_only=include_tumor_only,
                    chunksize=mutation_chunksize,
                )

                mutation_schema["mutation_patient_count"] = len(mutation_patient_ids)

                if logger is not None:
                    logger.info(
                        f"Mutation scan finished. Patients with mutations: {len(mutation_patient_ids)}"
                    )
            else:
                mutation_schema["mutation_patient_count"] = None
                mutation_schema["status"] = "sample_column_found_but_scan_skipped"

        manifest["has_mutations"] = manifest["patient_id"].isin(mutation_patient_ids)

    else:
        mutation_schema = {
            "status": "mutation_dataset_not_configured",
            "errors": [],
        }

    summary: Dict[str, Any] = {
        "clinical_rows_with_stage": int(manifest.shape[0]),
        "target_distribution": manifest["stage_group"].value_counts(dropna=True).to_dict(),
        "modality_patient_counts": {
            result["key"]: result["patient_count"] for result in matrix_results
        },
        "modality_status": {
            result["key"]: result["status"] for result in matrix_results
        },
        "multi_sample_modality_counts": {
            result["key"]: result["multi_sample_patient_count"]
            for result in matrix_results
        },
        "mutation_schema_status": mutation_schema.get("status"),
        "mutation_sample_column": mutation_schema.get("sample_column"),
        "mutation_gene_column": mutation_schema.get("gene_column"),
        "mutation_patient_count": mutation_schema.get("mutation_patient_count"),
    }

    return {
        "manifest": manifest,
        "sample_map_rows": sample_map_rows,
        "matrix_results": matrix_results,
        "mutation_schema": mutation_schema,
        "summary": summary,
    }