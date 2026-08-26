"""Deep inspection of mutation file schema.

This script helps identify the sample/patient identifier column and gene column
in the mutation file. It does not train models and does not modify raw data.
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
from src.luad.data.harmonizer import read_header_columns
from src.luad.utils.io import ensure_dir, save_json


TCGA_PATIENT_PATTERN = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"
TCGA_SAMPLE_PATTERN = r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2,6}"
UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

SAMPLE_NAME_KEYWORDS = [
    "tumor_sample",
    "sample_barcode",
    "sample_id",
    "sample",
    "barcode",
    "case",
    "participant",
    "donor",
    "patient",
]

GENE_NAME_KEYWORDS = [
    "hugo_symbol",
    "gene_symbol",
    "symbol",
    "gene",
    "hugo",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect mutation file schema to detect identifier columns."
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
        default=5000,
        help="Number of rows to inspect from mutation file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to artifacts/mutation_inspection.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def find_mutation_dataset(config: dict) -> dict:
    dataset_files = config.get("datasets_files", [])

    for item in dataset_files:
        if item.get("modality") == "mutation":
            return item

    raise ConfigError("No mutation dataset found in datasets_files.")


def truncate_value(value: str, max_chars: int = 60) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "..."


def inspect_mutation_column(column_name: str, series: pd.Series) -> dict:
    non_null_series = series.dropna()
    non_null_count = len(non_null_series)

    missing_ratio = float(series.isna().mean())
    unique_count = int(series.nunique(dropna=True))

    unique_ratio = unique_count / non_null_count if non_null_count > 0 else 0.0

    example_values = "; ".join(
        truncate_value(str(value)) for value in non_null_series.head(3)
    )

    patient_matches = 0
    sample_matches = 0
    uuid_matches = 0

    if non_null_count > 0:
        values = non_null_series.astype(str)

        patient_matches = int(
            values.str.contains(
                TCGA_PATIENT_PATTERN,
                flags=re.IGNORECASE,
                regex=True,
            ).sum()
        )

        sample_matches = int(
            values.str.contains(
                TCGA_SAMPLE_PATTERN,
                flags=re.IGNORECASE,
                regex=True,
            ).sum()
        )

        uuid_matches = int(
            values.str.contains(
                UUID_PATTERN,
                flags=re.IGNORECASE,
                regex=True,
            ).sum()
        )

    column_lower = str(column_name).lower()

    sample_name_score = sum(
        1 for keyword in SAMPLE_NAME_KEYWORDS if keyword in column_lower
    )

    gene_name_score = sum(
        1 for keyword in GENE_NAME_KEYWORDS if keyword in column_lower
    )

    return {
        "column": str(column_name),
        "dtype": str(series.dtype),
        "missing_ratio": round(missing_ratio, 6),
        "non_null_count": non_null_count,
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 6),
        "tcga_patient_matches": patient_matches,
        "tcga_sample_matches": sample_matches,
        "uuid_matches": uuid_matches,
        "sample_name_score": sample_name_score,
        "gene_name_score": gene_name_score,
        "example_values": example_values,
    }


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
        output_dir = project_root / "artifacts" / "mutation_inspection"

    output_dir = ensure_dir(output_dir)

    mutation_dataset = find_mutation_dataset(config)
    mutation_path = Path(mutation_dataset.get("resolved_path", ""))

    if not mutation_path.exists():
        print(f"[FAILED] Mutation file not found: {mutation_path}")
        sys.exit(1)

    print(f"Inspecting mutation file: {mutation_path}")

    try:
        columns, separator = read_header_columns(str(mutation_path))
    except Exception as exc:
        print(f"[FAILED] Could not read mutation header: {exc}")
        sys.exit(1)

    print(f"Separator: {repr(separator)}")
    print(f"Number of columns in header: {len(columns)}")

    try:
        preview_df = pd.read_csv(
            mutation_path,
            sep=separator,
            nrows=args.preview_rows,
            comment="#",
            on_bad_lines="skip",
            low_memory=False,
            encoding="utf-8-sig",
        )
    except Exception as exc:
        print(f"[FAILED] Could not read mutation preview rows: {exc}")
        sys.exit(1)

    print(f"Preview rows loaded: {preview_df.shape[0]}")
    print(f"Preview columns loaded: {preview_df.shape[1]}")

    inspection_rows = []

    for column in preview_df.columns:
        inspection_rows.append(
            inspect_mutation_column(
                column_name=column,
                series=preview_df[column],
            )
        )

    inspection_df = pd.DataFrame(inspection_rows)

    inspection_df = inspection_df.sort_values(
        by=[
            "tcga_sample_matches",
            "tcga_patient_matches",
            "sample_name_score",
            "uuid_matches",
            "unique_ratio",
        ],
        ascending=False,
    )

    inspection_path = output_dir / "mutation_schema_inspection.csv"
    inspection_df.to_csv(inspection_path, index=False, encoding="utf-8-sig")

    columns_path = output_dir / "mutation_columns.txt"

    with columns_path.open("w", encoding="utf-8") as handle:
        for column in preview_df.columns:
            handle.write(f"{column}\n")

    meta_report = {
        "mutation_path": str(mutation_path),
        "separator": separator,
        "header_columns_count": len(columns),
        "preview_rows_loaded": int(preview_df.shape[0]),
        "preview_columns_loaded": int(preview_df.shape[1]),
        "columns_first_50": [str(column) for column in preview_df.columns[:50]],
    }

    save_json(output_dir / "mutation_preview_meta.json", meta_report)

    print(f"\nSaved inspection report to: {inspection_path}")
    print(f"Saved column list to: {columns_path}")

    print("\nTop identifier candidates:")

    top_candidates = inspection_df[
        (inspection_df["tcga_sample_matches"] > 0)
        | (inspection_df["tcga_patient_matches"] > 0)
        | (inspection_df["uuid_matches"] > 0)
        | (inspection_df["sample_name_score"] > 0)
        | (
            (inspection_df["dtype"] == "object")
            & (inspection_df["unique_ratio"] > 0.5)
        )
    ].head(30)

    if top_candidates.empty:
        print("No obvious sample identifier candidate found.")
        print("Please send mutation_schema_inspection.csv for manual review.")
    else:
        for _, row in top_candidates.iterrows():
            print(
                "- "
                f"column={row['column']} | "
                f"dtype={row['dtype']} | "
                f"sample_matches={row['tcga_sample_matches']} | "
                f"patient_matches={row['tcga_patient_matches']} | "
                f"uuid_matches={row['uuid_matches']} | "
                f"unique={row['unique_count']} | "
                f"unique_ratio={row['unique_ratio']} | "
                f"examples={row['example_values']}"
            )

    print("\nAll mutation columns:")

    for column in preview_df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()