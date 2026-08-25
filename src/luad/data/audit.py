"""Data audit utilities for inspecting real dataset structure before modeling.

This module performs a safe, memory-aware audit of raw files.
It does NOT train models and does NOT make final assumptions about orientation,
target, or identifiers. It only reports evidence from the real files.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from src.luad.utils.io import ensure_dir, save_json, save_text

DEFAULT_PATIENT_PATTERN = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"
SAMPLE_PATTERN = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2,6}"

ID_KEYWORDS = [
    "patient",
    "sample",
    "barcode",
    "case",
    "donor",
    "specimen",
    "aliquot",
    "participant",
    "gene",
    "symbol",
    "probe",
    "mirna",
    "mir",
    "protein",
    "antibody",
    "hugo",
    "entrez",
    "feature",
    "id",
]

TARGET_KEYWORDS = [
    "stage",
    "pathologic_stage",
    "tumor_stage",
    "ajcc",
    "stage_group",
    "histology",
    "diagnosis",
    "subtype",
    "os_status",
    "dfs_status",
    "recurrence",
    "progression",
    "vital_status",
    "survival",
    "response",
]

LEAKAGE_KEYWORDS = [
    "os_status",
    "os_months",
    "days_to_death",
    "death",
    "vital_status",
    "dfs_status",
    "dfs_months",
    "days_to_recurrence",
    "recurrence",
    "progression",
    "response",
    "treatment",
    "therapy",
    "followup",
    "follow_up",
    "outcome",
]


def _safe_float(value: Any) -> Optional[float]:
    try:
        numeric_value = float(value)
        if pd.isna(numeric_value):
            return None
        return numeric_value
    except Exception:
        return None


def _truncate(value: Optional[str], max_chars: int = 500) -> str:
    if value is None:
        return ""

    value_str = str(value)

    if len(value_str) <= max_chars:
        return value_str

    return value_str[:max_chars] + "... [TRUNCATED]"


def _md_escape(value: Any) -> str:
    if value is None:
        return ""

    return str(value).replace("|", "/").replace("\n", " ")


def _read_head_lines(path: Path, n_lines: int = 30) -> List[str]:
    lines: List[str] = []

    with path.open("r", encoding="utf-8-sig", errors="ignore") as handle:
        for index, line in enumerate(handle):
            if index >= n_lines:
                break

            lines.append(line.rstrip("\n\r"))

    return lines


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


def _split_line(line: Optional[str], separator: str) -> List[str]:
    if line is None:
        return []

    if separator == "\s+":
        return line.split()

    return line.split(separator)


def _choose_header_line(lines: List[str]) -> Tuple[Optional[int], Optional[str]]:
    for index, line in enumerate(lines):
        stripped_line = line.strip()

        if not stripped_line:
            continue

        if stripped_line.startswith("#"):
            continue

        return index, line

    return None, None


def _should_count_rows(count_rows_setting: Any, file_size_mb: float) -> bool:
    if isinstance(count_rows_setting, bool):
        return count_rows_setting

    value = str(count_rows_setting).lower()

    if value == "auto":
        return file_size_mb <= 50.0

    return value in {"true", "1", "yes", "y", "on"}


def _count_rows_binary(path: Path) -> Optional[int]:
    try:
        file_size = path.stat().st_size

        if file_size == 0:
            return 0

        newline_count = 0
        chunk_size = 1024 * 1024

        with path.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)

                if not chunk:
                    break

                newline_count += chunk.count(b"\n")

        with path.open("rb") as handle:
            handle.seek(-1, 2)
            last_char = handle.read(1)

            if last_char != b"\n":
                newline_count += 1

        return newline_count

    except Exception:
        return None


def _detect_ids_from_columns(
    columns: List[str],
    patient_regex: re.Pattern,
    sample_regex: re.Pattern,
    limit: int = 2000,
) -> Tuple[List[str], List[str], int]:
    patient_ids: Set[str] = set()
    sample_ids: Set[str] = set()
    tcga_column_count = 0

    for column in columns:
        column_str = str(column)
        column_has_tcga = False

        for match in patient_regex.finditer(column_str):
            patient_ids.add(match.group(0))
            column_has_tcga = True

        for match in sample_regex.finditer(column_str):
            sample_ids.add(match.group(0))
            column_has_tcga = True

        if column_has_tcga:
            tcga_column_count += 1

        if len(patient_ids) >= limit and len(sample_ids) >= limit:
            break

    return sorted(patient_ids)[:limit], sorted(sample_ids)[:limit], tcga_column_count


def _detect_ids_from_first_fields(
    lines: List[str],
    header_index: Optional[int],
    separator: str,
    patient_regex: re.Pattern,
    sample_regex: re.Pattern,
    max_lines: int = 100,
    limit: int = 2000,
) -> Tuple[List[str], List[str], int]:
    if header_index is None:
        return [], [], 0

    patient_ids: Set[str] = set()
    sample_ids: Set[str] = set()
    match_count = 0

    start_index = header_index + 1
    end_index = min(len(lines), start_index + max_lines)

    for line in lines[start_index:end_index]:
        fields = _split_line(line, separator)

        if not fields:
            continue

        first_value = fields[0].strip().strip('"').strip("'")

        has_match = False

        for match in patient_regex.finditer(first_value):
            patient_ids.add(match.group(0))
            has_match = True

        for match in sample_regex.finditer(first_value):
            sample_ids.add(match.group(0))
            has_match = True

        if has_match:
            match_count += 1

        if len(patient_ids) >= limit and len(sample_ids) >= limit:
            break

    return sorted(patient_ids)[:limit], sorted(sample_ids)[:limit], match_count


def _get_single_column(df: Optional[pd.DataFrame], column_name: str) -> Optional[pd.Series]:
    if df is None:
        return None

    if column_name not in df.columns:
        return None

    if list(df.columns).count(column_name) != 1:
        return None

    return df[column_name]


def _detect_ids_from_dataframe(
    df: Optional[pd.DataFrame],
    patient_regex: re.Pattern,
    sample_regex: re.Pattern,
    max_columns: int = 5,
    max_values_per_column: int = 2000,
    limit: int = 2000,
) -> Tuple[List[str], List[str]]:
    if df is None or df.empty:
        return [], []

    patient_ids: Set[str] = set()
    sample_ids: Set[str] = set()

    scanned_columns = 0

    for column in df.columns:
        if scanned_columns >= max_columns:
            break

        if str(df[column].dtype) != "object":
            continue

        scanned_columns += 1

        values = df[column].dropna().astype(str).head(max_values_per_column)

        for value in values:
            for match in patient_regex.finditer(value):
                patient_ids.add(match.group(0))

            for match in sample_regex.finditer(value):
                sample_ids.add(match.group(0))

            if len(patient_ids) >= limit and len(sample_ids) >= limit:
                break

    return sorted(patient_ids)[:limit], sorted(sample_ids)[:limit]


def _get_id_column_candidates(columns: List[str]) -> List[str]:
    candidates: List[str] = []

    for column in columns:
        column_lower = str(column).lower()

        if any(keyword in column_lower for keyword in ID_KEYWORDS):
            candidates.append(str(column))

    return candidates[:50]


def _load_preview_dataframe(
    path: Path,
    separator: str,
    max_rows: int,
) -> pd.DataFrame:
    engine = "python" if separator == "\s+" else "c"

    return pd.read_csv(
        path,
        sep=separator,
        engine=engine,
        nrows=max_rows,
        comment="#",
        on_bad_lines="skip",
        low_memory=False,
        encoding="utf-8-sig",
    )


def _top_missing_columns(df: pd.DataFrame, top_k: int = 30) -> List[Dict[str, Any]]:
    if df.empty:
        return []

    missing_ratios = df.isna().mean()
    missing_ratios = missing_ratios.sort_values(ascending=False)

    rows: List[Dict[str, Any]] = []

    for column, ratio in missing_ratios.head(top_k).items():
        ratio_value = _safe_float(ratio)

        if ratio_value is None:
            continue

        rows.append(
            {
                "column": str(column),
                "missing_ratio": ratio_value,
                "dtype": str(df[column].dtype),
            }
        )

    return rows


def _clinical_target_and_leakage_rows(
    dataset_key: str,
    columns: List[str],
    df: Optional[pd.DataFrame],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    target_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []

    for column in columns:
        column_lower = str(column).lower()

        if any(keyword in column_lower for keyword in TARGET_KEYWORDS):
            series = _get_single_column(df, column)

            missing_ratio = None
            unique_count = None
            top_values = ""

            if series is not None:
                missing_ratio = _safe_float(series.isna().mean())
                unique_count = int(series.nunique(dropna=True))

                value_counts = series.astype(str).value_counts(dropna=True).head(10)
                top_values = "; ".join(
                    f"{str(index)}: {int(count)}"
                    for index, count in value_counts.items()
                )

            target_rows.append(
                {
                    "dataset": dataset_key,
                    "column": str(column),
                    "role": "target_candidate",
                    "missing_ratio": missing_ratio,
                    "unique_count": unique_count,
                    "top_values": top_values,
                    "decision": "review_required",
                }
            )

        if any(keyword in column_lower for keyword in LEAKAGE_KEYWORDS):
            series = _get_single_column(df, column)

            missing_ratio = None

            if series is not None:
                missing_ratio = _safe_float(series.isna().mean())

            leakage_rows.append(
                {
                    "dataset": dataset_key,
                    "column": str(column),
                    "role": "clinical",
                    "leakage_risk": "high_or_label_related",
                    "missing_ratio": missing_ratio,
                    "decision": "review_required",
                    "reason": "Outcome/survival/recurrence/treatment-related field.",
                }
            )

    return target_rows, leakage_rows


def audit_dataset(
    file_config: Dict[str, Any],
    config: Dict[str, Any],
    logger: Optional[Any] = None,
) -> Tuple[Dict[str, Any], Set[str], Set[str]]:
    audit_config = config.get("audit", {})
    identifiers_config = config.get("identifiers", {})

    patient_pattern = identifiers_config.get(
        "tcga_barcode_pattern",
        DEFAULT_PATIENT_PATTERN,
    )

    try:
        patient_regex = re.compile(patient_pattern)
    except re.error:
        patient_regex = re.compile(DEFAULT_PATIENT_PATTERN)

    sample_regex = re.compile(SAMPLE_PATTERN)

    max_rows_schema = int(audit_config.get("max_rows_for_schema", 1000))
    max_features_profile = int(audit_config.get("max_features_for_profile", 5000))
    max_file_mb_for_preview = float(audit_config.get("max_file_mb_for_preview", 300))
    large_file_mb = float(audit_config.get("large_file_mb", 500))
    external_header_only = bool(audit_config.get("external_header_only", True))
    count_rows_setting = audit_config.get("count_rows", False)

    dataset_key = str(file_config.get("key", "unknown_dataset"))
    modality = str(file_config.get("modality", "unknown_modality"))
    resolved_path = str(file_config.get("resolved_path", ""))

    report: Dict[str, Any] = {
        "key": dataset_key,
        "modality": modality,
        "resolved_path": resolved_path,
        "audit_enabled": bool(file_config.get("audit", True)),
        "exists": False,
        "is_file": False,
        "file_size_mb": None,
        "large_file": False,
        "separator": None,
        "header_line_index": None,
        "n_columns": None,
        "columns_first_50": [],
        "columns_last_20": [],
        "head_lines_truncated": [],
        "row_count": None,
        "profile_status": "not_run",
        "preview_rows_loaded": None,
        "dtypes_first_100": {},
        "numeric_columns_count": None,
        "categorical_columns_count": None,
        "missing_cells_preview": None,
        "missing_percentage_preview": None,
        "top_missing_columns": [],
        "duplicate_columns_count": None,
        "duplicate_column_names": [],
        "first_column_name": None,
        "first_column_duplicate_count_preview": None,
        "id_column_candidates": [],
        "tcga_column_count": None,
        "first_field_tcga_count": None,
        "detected_patient_id_count": None,
        "detected_sample_id_count": None,
        "detected_patient_ids_first_100": [],
        "detected_sample_ids_first_100": [],
        "orientation_hints": [],
        "clinical_target_candidates": [],
        "leakage_candidates": [],
        "errors": [],
    }

    empty_patient_ids: Set[str] = set()
    empty_sample_ids: Set[str] = set()

    if not report["audit_enabled"]:
        report["profile_status"] = "skipped_disabled"
        return report, empty_patient_ids, empty_sample_ids

    if not resolved_path:
        report["errors"].append("resolved_path is empty.")
        report["profile_status"] = "skipped_no_path"
        return report, empty_patient_ids, empty_sample_ids

    path = Path(resolved_path)

    if not path.exists():
        report["errors"].append(f"File not found: {path}")
        report["profile_status"] = "skipped_missing_file"
        return report, empty_patient_ids, empty_sample_ids

    report["exists"] = True

    if path.is_dir():
        report["errors"].append("Path is a directory, not a file.")
        report["profile_status"] = "skipped_directory"
        return report, empty_patient_ids, empty_sample_ids

    report["is_file"] = True

    try:
        file_size_bytes = path.stat().st_size
        file_size_mb = file_size_bytes / (1024 * 1024)
        report["file_size_mb"] = round(file_size_mb, 3)
        report["large_file"] = file_size_mb >= large_file_mb
    except Exception as exc:
        report["errors"].append(f"Could not read file size: {exc}")
        file_size_mb = 0.0

    if logger is not None:
        logger.info(
            f"Auditing dataset key='{dataset_key}' modality='{modality}' "
            f"size_mb={report['file_size_mb']}"
        )

    try:
        head_lines = _read_head_lines(path, n_lines=30)
    except Exception as exc:
        report["errors"].append(f"Could not read head lines: {exc}")
        report["profile_status"] = "skipped_read_error"
        return report, empty_patient_ids, empty_sample_ids

    report["head_lines_truncated"] = [
        _truncate(line, max_chars=500) for line in head_lines[:10]
    ]

    header_index, header_line = _choose_header_line(head_lines)
    report["header_line_index"] = header_index

    if header_line is None:
        report["errors"].append("No header line found among first non-comment lines.")
        report["profile_status"] = "skipped_no_header"
        return report, empty_patient_ids, empty_sample_ids

    separator = _detect_separator(header_line)
    report["separator"] = separator

    columns = _split_line(header_line, separator)
    report["n_columns"] = len(columns)
    report["columns_first_50"] = [str(column) for column in columns[:50]]
    report["columns_last_20"] = [str(column) for column in columns[-20:]]
    report["id_column_candidates"] = _get_id_column_candidates(columns)

    if _should_count_rows(count_rows_setting, file_size_mb):
        report["row_count"] = _count_rows_binary(path)

    patient_ids_from_columns, sample_ids_from_columns, tcga_column_count = (
        _detect_ids_from_columns(
            columns=columns,
            patient_regex=patient_regex,
            sample_regex=sample_regex,
        )
    )

    report["tcga_column_count"] = tcga_column_count

    patient_ids_from_first_fields, sample_ids_from_first_fields, first_field_count = (
        _detect_ids_from_first_fields(
            lines=head_lines,
            header_index=header_index,
            separator=separator,
            patient_regex=patient_regex,
            sample_regex=sample_regex,
        )
    )

    report["first_field_tcga_count"] = first_field_count

    df: Optional[pd.DataFrame] = None

    is_external = bool(file_config.get("external_validation_only", False))

    profile_allowed = (
        report["exists"]
        and report["is_file"]
        and header_line is not None
        and report["n_columns"] is not None
        and report["n_columns"] > 0
        and report["n_columns"] <= max_features_profile
        and file_size_mb <= max_file_mb_for_preview
        and not (is_external and external_header_only)
    )

    if not profile_allowed:
        if is_external and external_header_only:
            report["profile_status"] = "skipped_external_header_only"
        elif file_size_mb > max_file_mb_for_preview:
            report["profile_status"] = "skipped_large_file"
        elif report["n_columns"] is not None and report["n_columns"] > max_features_profile:
            report["profile_status"] = "skipped_too_many_columns"
        else:
            report["profile_status"] = "skipped"
    else:
        try:
            df = _load_preview_dataframe(
                path=path,
                separator=separator,
                max_rows=max_rows_schema,
            )

            report["profile_status"] = "ok"
            report["preview_rows_loaded"] = int(df.shape[0])

            report["dtypes_first_100"] = {
                str(column): str(dtype)
                for column, dtype in list(df.dtypes.items())[:100]
            }

            numeric_df = df.select_dtypes(include=["number"])
            categorical_df = df.select_dtypes(include=["object", "category", "bool"])

            report["numeric_columns_count"] = int(numeric_df.shape[1])
            report["categorical_columns_count"] = int(categorical_df.shape[1])

            missing_cells = int(df.isna().sum().sum())
            total_cells = int(df.size)

            report["missing_cells_preview"] = missing_cells

            if total_cells > 0:
                report["missing_percentage_preview"] = round(
                    100.0 * missing_cells / total_cells,
                    4,
                )

            report["top_missing_columns"] = _top_missing_columns(df, top_k=30)

            duplicated_column_names = df.columns[df.columns.duplicated()].unique().tolist()
            report["duplicate_columns_count"] = len(duplicated_column_names)
            report["duplicate_column_names"] = [
                str(column) for column in duplicated_column_names[:50]
            ]

            if df.shape[1] > 0:
                first_column = df.columns[0]
                report["first_column_name"] = str(first_column)

                try:
                    report["first_column_duplicate_count_preview"] = int(
                        df.iloc[:, 0].duplicated().sum()
                    )
                except Exception:
                    report["first_column_duplicate_count_preview"] = None

        except Exception as exc:
            report["profile_status"] = "error"
            report["errors"].append(f"Preview dataframe failed: {exc}")

    patient_ids_from_preview: List[str] = []
    sample_ids_from_preview: List[str] = []

    if df is not None:
        patient_ids_from_preview, sample_ids_from_preview = _detect_ids_from_dataframe(
            df=df,
            patient_regex=patient_regex,
            sample_regex=sample_regex,
        )

    patient_id_set: Set[str] = set(patient_ids_from_columns)
    patient_id_set.update(patient_ids_from_first_fields)
    patient_id_set.update(patient_ids_from_preview)

    sample_id_set: Set[str] = set(sample_ids_from_columns)
    sample_id_set.update(sample_ids_from_first_fields)
    sample_id_set.update(sample_ids_from_preview)

    report["detected_patient_id_count"] = len(patient_id_set)
    report["detected_sample_id_count"] = len(sample_id_set)
    report["detected_patient_ids_first_100"] = sorted(patient_id_set)[:100]
    report["detected_sample_ids_first_100"] = sorted(sample_id_set)[:100]

    orientation_hints: List[str] = []

    if tcga_column_count >= 5 and tcga_column_count >= first_field_count:
        orientation_hints.append("features_x_samples_candidate")

    if first_field_count >= 5:
        orientation_hints.append("samples_x_features_candidate")

    if not orientation_hints:
        orientation_hints.append("unknown")

    report["orientation_hints"] = orientation_hints

    if modality == "clinical":
        target_rows, leakage_rows = _clinical_target_and_leakage_rows(
            dataset_key=dataset_key,
            columns=columns,
            df=df,
        )

        report["clinical_target_candidates"] = target_rows
        report["leakage_candidates"] = leakage_rows

    if logger is not None:
        logger.info(
            f"Finished audit for key='{dataset_key}' "
            f"profile_status='{report['profile_status']}' "
            f"n_columns={report['n_columns']} "
            f"detected_patient_ids={report['detected_patient_id_count']} "
            f"orientation_hints={','.join(report['orientation_hints'])}"
        )

    return report, patient_id_set, sample_id_set


def _save_csv(rows: List[Dict[str, Any]], path: Path, columns: Optional[List[str]] = None) -> None:
    if rows:
        dataframe = pd.DataFrame(rows)
    else:
        dataframe = pd.DataFrame(columns=columns or [])

    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def _build_overlap_rows(patient_id_sets: Dict[str, Set[str]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = list(patient_id_sets.keys())

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            key_a = keys[i]
            key_b = keys[j]

            set_a = patient_id_sets[key_a]
            set_b = patient_id_sets[key_b]

            common = set_a.intersection(set_b)

            rows.append(
                {
                    "dataset_a": key_a,
                    "dataset_b": key_b,
                    "dataset_a_patient_ids": len(set_a),
                    "dataset_b_patient_ids": len(set_b),
                    "common_patient_ids": len(common),
                }
            )

    return rows


def _build_markdown_report(
    summary_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    leakage_rows: List[Dict[str, Any]],
    overlap_rows: List[Dict[str, Any]],
    common_patient_ids_all: int,
) -> str:
    lines: List[str] = []

    lines.append("# DATA AUDIT REPORT")
    lines.append("")
    lines.append(f"- Generated at: {datetime.now().isoformat()}")
    lines.append("- Important: This is a pre-modeling audit. No target or loader is finalized yet.")
    lines.append("")

    lines.append("## 1. Dataset Summary")
    lines.append("")
    lines.append(
        "| Key | Modality | Exists | Size MB | Columns | Separator | Profile Status | Row Count | Missing % Preview | Detected Patient IDs | Orientation Hints |"
    )
    lines.append(
        "|---|---|---|---:|---:|---|---|---:|---:|---:|---|"
    )

    for row in summary_rows:
        lines.append(
            "| "
            f"{_md_escape(row.get('key'))} | "
            f"{_md_escape(row.get('modality'))} | "
            f"{_md_escape(row.get('exists'))} | "
            f"{_md_escape(row.get('file_size_mb'))} | "
            f"{_md_escape(row.get('n_columns'))} | "
            f"{_md_escape(row.get('separator_display'))} | "
            f"{_md_escape(row.get('profile_status'))} | "
            f"{_md_escape(row.get('row_count'))} | "
            f"{_md_escape(row.get('missing_percentage_preview'))} | "
            f"{_md_escape(row.get('detected_patient_id_count'))} | "
            f"{_md_escape(row.get('orientation_hints'))} |"
        )

    lines.append("")
    lines.append("## 2. Clinical Target Candidates")
    lines.append("")

    if target_rows:
        lines.append(
            "| Dataset | Column | Missing Ratio | Unique Count | Top Values | Decision |"
        )
        lines.append("|---|---|---:|---:|---|---|")

        for row in target_rows:
            lines.append(
                "| "
                f"{_md_escape(row.get('dataset'))} | "
                f"{_md_escape(row.get('column'))} | "
                f"{_md_escape(row.get('missing_ratio'))} | "
                f"{_md_escape(row.get('unique_count'))} | "
                f"{_md_escape(row.get('top_values'))} | "
                f"{_md_escape(row.get('decision'))} |"
            )
    else:
        lines.append("No clinical target candidates detected from column names.")

    lines.append("")
    lines.append("## 3. Clinical Leakage Candidates")
    lines.append("")

    if leakage_rows:
        lines.append(
            "| Dataset | Column | Leakage Risk | Missing Ratio | Decision | Reason |"
        )
        lines.append("|---|---|---|---:|---|---|")

        for row in leakage_rows:
            lines.append(
                "| "
                f"{_md_escape(row.get('dataset'))} | "
                f"{_md_escape(row.get('column'))} | "
                f"{_md_escape(row.get('leakage_risk'))} | "
                f"{_md_escape(row.get('missing_ratio'))} | "
                f"{_md_escape(row.get('decision'))} | "
                f"{_md_escape(row.get('reason'))} |"
            )
    else:
        lines.append("No clinical leakage candidates detected from column names.")

    lines.append("")
    lines.append("## 4. Preliminary Patient ID Overlap")
    lines.append("")
    lines.append(
        "This overlap is based only on patient-like TCGA identifiers detected during audit. "
        "It is preliminary and must be followed by proper harmonization."
    )
    lines.append("")
    lines.append(f"- Common patient IDs across all audited datasets with detected IDs: {common_patient_ids_all}")
    lines.append("")

    if overlap_rows:
        lines.append(
            "| Dataset A | Dataset B | IDs A | IDs B | Common IDs |"
        )
        lines.append("|---|---|---:|---:|---:|")

        for row in overlap_rows:
            lines.append(
                "| "
                f"{_md_escape(row.get('dataset_a'))} | "
                f"{_md_escape(row.get('dataset_b'))} | "
                f"{_md_escape(row.get('dataset_a_patient_ids'))} | "
                f"{_md_escape(row.get('dataset_b_patient_ids'))} | "
                f"{_md_escape(row.get('common_patient_ids'))} |"
            )
    else:
        lines.append("No overlap rows computed.")

    lines.append("")

    return "\n".join(lines)


def run_audit(config: Dict[str, Any], logger: Optional[Any] = None) -> Dict[str, str]:
    audit_config = config.get("audit", {})
    runtime_config = config.get("runtime", {})

    project_root = Path(runtime_config.get("project_root", Path.cwd()))
    output_dir_value = audit_config.get("output_dir", "artifacts/data_audit")
    output_dir_path = Path(output_dir_value)

    if not output_dir_path.is_absolute():
        output_dir_path = project_root / output_dir_path

    output_dir = ensure_dir(output_dir_path)

    dataset_files = config.get("datasets_files", [])

    if not dataset_files:
        message = "No dataset files found in config['datasets_files']."
        if logger is not None:
            logger.error(message)
        raise RuntimeError(message)

    summary_rows: List[Dict[str, Any]] = []
    missingness_rows: List[Dict[str, Any]] = []
    duplicate_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    leakage_rows: List[Dict[str, Any]] = []
    schema_report: Dict[str, Any] = {}
    patient_id_sets: Dict[str, Set[str]] = {}

    for file_config in dataset_files:
        report, patient_id_set, _ = audit_dataset(
            file_config=file_config,
            config=config,
            logger=logger,
        )

        dataset_key = report["key"]
        schema_report[dataset_key] = report

        separator_display = report.get("separator")
        if separator_display == "\t":
            separator_display = "tab"
        elif separator_display == "\s+":
            separator_display = "whitespace"

        summary_rows.append(
            {
                "key": report.get("key"),
                "modality": report.get("modality"),
                "exists": report.get("exists"),
                "file_size_mb": report.get("file_size_mb"),
                "n_columns": report.get("n_columns"),
                "separator_display": separator_display,
                "profile_status": report.get("profile_status"),
                "row_count": report.get("row_count"),
                "missing_percentage_preview": report.get("missing_percentage_preview"),
                "detected_patient_id_count": report.get("detected_patient_id_count"),
                "orientation_hints": ";".join(report.get("orientation_hints", [])),
                "errors": ";".join(report.get("errors", [])),
                "resolved_path": report.get("resolved_path"),
            }
        )

        for missing_row in report.get("top_missing_columns", []):
            missingness_rows.append(
                {
                    "dataset": dataset_key,
                    "modality": report.get("modality"),
                    "column": missing_row.get("column"),
                    "missing_ratio": missing_row.get("missing_ratio"),
                    "dtype": missing_row.get("dtype"),
                }
            )

        duplicate_rows.append(
            {
                "dataset": dataset_key,
                "modality": report.get("modality"),
                "duplicate_columns_count": report.get("duplicate_columns_count"),
                "duplicate_column_names": ";".join(report.get("duplicate_column_names", [])),
                "first_column_name": report.get("first_column_name"),
                "first_column_duplicate_count_preview": report.get(
                    "first_column_duplicate_count_preview"
                ),
            }
        )

        target_rows.extend(report.get("clinical_target_candidates", []))
        leakage_rows.extend(report.get("leakage_candidates", []))

        if patient_id_set:
            patient_id_sets[dataset_key] = patient_id_set

    overlap_rows = _build_overlap_rows(patient_id_sets)

    if patient_id_sets:
        common_all = set.intersection(*patient_id_sets.values())
        common_patient_ids_all = len(common_all)
    else:
        common_patient_ids_all = 0

    dataset_summary_path = output_dir / "dataset_summary.csv"
    schema_report_path = output_dir / "schema_report.json"
    missingness_report_path = output_dir / "missingness_report.csv"
    duplicate_report_path = output_dir / "duplicate_report.csv"
    target_candidates_path = output_dir / "clinical_target_candidates.csv"
    leakage_audit_path = output_dir / "leakage_audit.csv"
    sample_overlap_path = output_dir / "sample_overlap_preliminary.csv"
    markdown_report_path = output_dir / "DATA_AUDIT_REPORT.md"

    _save_csv(
        summary_rows,
        dataset_summary_path,
        columns=[
            "key",
            "modality",
            "exists",
            "file_size_mb",
            "n_columns",
            "separator_display",
            "profile_status",
            "row_count",
            "missing_percentage_preview",
            "detected_patient_id_count",
            "orientation_hints",
            "errors",
            "resolved_path",
        ],
    )

    save_json(schema_report_path, schema_report)

    _save_csv(
        missingness_rows,
        missingness_report_path,
        columns=["dataset", "modality", "column", "missing_ratio", "dtype"],
    )

    _save_csv(
        duplicate_rows,
        duplicate_report_path,
        columns=[
            "dataset",
            "modality",
            "duplicate_columns_count",
            "duplicate_column_names",
            "first_column_name",
            "first_column_duplicate_count_preview",
        ],
    )

    _save_csv(
        target_rows,
        target_candidates_path,
        columns=[
            "dataset",
            "column",
            "role",
            "missing_ratio",
            "unique_count",
            "top_values",
            "decision",
        ],
    )

    _save_csv(
        leakage_rows,
        leakage_audit_path,
        columns=[
            "dataset",
            "column",
            "role",
            "leakage_risk",
            "missing_ratio",
            "decision",
            "reason",
        ],
    )

    _save_csv(
        overlap_rows,
        sample_overlap_path,
        columns=[
            "dataset_a",
            "dataset_b",
            "dataset_a_patient_ids",
            "dataset_b_patient_ids",
            "common_patient_ids",
        ],
    )

    markdown_text = _build_markdown_report(
        summary_rows=summary_rows,
        target_rows=target_rows,
        leakage_rows=leakage_rows,
        overlap_rows=overlap_rows,
        common_patient_ids_all=common_patient_ids_all,
    )

    save_text(markdown_report_path, markdown_text)

    if logger is not None:
        logger.info(f"Data audit outputs saved to: {output_dir}")

    return {
        "output_dir": str(output_dir),
        "dataset_summary": str(dataset_summary_path),
        "schema_report": str(schema_report_path),
        "missingness_report": str(missingness_report_path),
        "duplicate_report": str(duplicate_report_path),
        "clinical_target_candidates": str(target_candidates_path),
        "leakage_audit": str(leakage_audit_path),
        "sample_overlap_preliminary": str(sample_overlap_path),
        "markdown_report": str(markdown_report_path),
    }