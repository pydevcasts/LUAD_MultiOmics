"""Inspect omics matrix files to identify feature identifier columns.

This script does not build modeling matrices yet.
It only inspects header and a few preview rows so that feature IDs are not guessed.
"""

from pathlib import Path
import argparse
import re
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.harmonizer import (
    PATIENT_REGEX,
    is_tumor_sample,
    read_header_columns,
)
from src.luad.utils.io import ensure_dir, save_json


MATRIX_MODALITIES = {
    "mrna",
    "mirna",
    "methylation",
    "cna",
    "rppa",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect omics matrix feature schema."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config YAML file.",
    )

    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of preview rows to read from each matrix file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to artifacts/data_prep.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def truncate_value(value: str, max_chars: int = 80) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "..."


def inspect_matrix_file(
    file_config: dict,
    preview_rows: int = 5,
) -> tuple[dict, list]:
    key = str(file_config.get("key", "unknown"))
    modality = str(file_config.get("modality", "unknown"))
    resolved_path = str(file_config.get("resolved_path", ""))

    summary = {
        "key": key,
        "modality": modality,
        "resolved_path": resolved_path,
        "status": "not_run",
        "separator": None,
        "total_columns": None,
        "sample_columns_count": None,
        "tumor_sample_columns_count": None,
        "normal_sample_columns_count": None,
        "non_sample_columns_count": None,
        "first_sample_column": None,
        "first_sample_column_index": None,
        "columns_before_first_sample": "",
        "feature_id_candidates_first_10": "",
        "errors": "",
    }

    details = []

    if not resolved_path:
        summary["status"] = "missing_path"
        summary["errors"] = "resolved_path is empty."
        return summary, details

    path = Path(resolved_path)

    if not path.exists():
        summary["status"] = "missing_file"
        summary["errors"] = f"File not found: {path}"
        return summary, details

    try:
        columns, separator = read_header_columns(str(path))
    except Exception as exc:
        summary["status"] = "header_read_error"
        summary["errors"] = str(exc)
        return summary, details

    summary["separator"] = separator
    summary["total_columns"] = len(columns)

    sample_columns = []
    tumor_sample_columns = []
    normal_sample_columns = []

    for column in columns:
        column_str = str(column)

        if PATIENT_REGEX.search(column_str):
            sample_columns.append(column_str)

            if is_tumor_sample(column_str):
                tumor_sample_columns.append(column_str)
            else:
                normal_sample_columns.append(column_str)

    summary["sample_columns_count"] = len(sample_columns)
    summary["tumor_sample_columns_count"] = len(tumor_sample_columns)
    summary["normal_sample_columns_count"] = len(normal_sample_columns)

    sample_column_set = set(sample_columns)

    first_sample_column = None
    first_sample_column_index = None

    for index, column in enumerate(columns):
        if column in sample_column_set:
            first_sample_column = column
            first_sample_column_index = index
            break

    summary["first_sample_column"] = first_sample_column
    summary["first_sample_column_index"] = first_sample_column_index

    if first_sample_column_index is not None:
        columns_before_first_sample = columns[:first_sample_column_index]
    else:
        columns_before_first_sample = columns

    summary["non_sample_columns_count"] = len(columns_before_first_sample)

    summary["columns_before_first_sample"] = ";".join(
        columns_before_first_sample[:20]
    )

    summary["feature_id_candidates_first_10"] = ";".join(
        columns_before_first_sample[:10]
    )

    try:
        preview_df = pd.read_csv(
            path,
            sep=separator,
            nrows=preview_rows,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        summary["status"] = "preview_read_error"
        summary["errors"] = str(exc)
        return summary, details

    preview_sample_columns = [
        column for column in preview_df.columns if str(column) in sample_column_set
    ]

    preview_non_sample_columns = [
        column
        for column in preview_df.columns
        if str(column) not in sample_column_set
    ]

    for column in preview_non_sample_columns[:10]:
        series = preview_df[column]

        examples = [
            truncate_value(str(value))
            for value in series.head(3)
        ]

        details.append(
            {
                "dataset_key": key,
                "modality": modality,
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_in_preview": int(series.isna().sum()),
                "unique_in_preview": int(series.nunique(dropna=True)),
                "example_values": "; ".join(examples),
            }
        )

    summary["status"] = "ok"

    return summary, details


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)

    try:
        config = load_config(config_path=config_path)
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "artifacts" / "data_prep"

    output_dir = ensure_dir(output_dir)

    dataset_files = config.get("datasets_files", [])

    summary_rows = []
    all_details = {}

    for file_config in dataset_files:
        modality = str(file_config.get("modality", ""))

        if modality not in MATRIX_MODALITIES:
            continue

        if file_config.get("external_validation_only", False):
            continue

        if modality == "normals_reference":
            continue

        key = str(file_config.get("key", "unknown"))

        print(f"Inspecting {key} ...")

        summary, details = inspect_matrix_file(
            file_config=file_config,
            preview_rows=args.preview_rows,
        )

        summary_rows.append(summary)
        all_details[key] = details

        print(
            f"- status={summary['status']}, "
            f"total_columns={summary['total_columns']}, "
            f"tumor_samples={summary['tumor_sample_columns_count']}, "
            f"feature_candidates={summary['feature_id_candidates_first_10']}"
        )

    summary_df = pd.DataFrame(summary_rows)

    summary_path = output_dir / "feature_schema_inspection.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    details_path = output_dir / "feature_schema_details.json"
    save_json(details_path, all_details)

    print(f"\nSaved feature schema summary to: {summary_path}")
    print(f"Saved feature schema details to: {details_path}")


if __name__ == "__main__":
    main()