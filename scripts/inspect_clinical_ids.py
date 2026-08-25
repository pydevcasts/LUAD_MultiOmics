"""Inspect clinical columns to detect patient/sample identifier candidates.

This is a diagnostic script. It does not train models and does not modify data.
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
from src.luad.utils.io import ensure_dir


TCGA_PATIENT_REGEX = re.compile(
    r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}",
    re.IGNORECASE,
)

TCGA_SAMPLE_REGEX = re.compile(
    r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-[A-Z0-9]{2,6}",
    re.IGNORECASE,
)

ID_NAME_KEYWORDS = [
    "patient",
    "sample",
    "barcode",
    "case",
    "participant",
    "donor",
    "specimen",
    "aliquot",
    "id",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect clinical columns for patient/sample ID candidates."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config YAML file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to artifacts/target_report.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def find_clinical_dataset(config: dict) -> dict:
    dataset_files = config.get("datasets_files", [])

    for item in dataset_files:
        if item.get("modality") == "clinical":
            return item

    raise ConfigError("No clinical dataset found in datasets_files.")


def read_clinical_dataframe(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            sep="\t",
            comment="#",
            encoding="utf-8-sig",
            low_memory=False,
        )
    except Exception:
        return pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            encoding="utf-8-sig",
            low_memory=False,
        )


def truncate_value(value: str, max_chars: int = 25) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "..."


def inspect_clinical_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        series = df[column]

        non_null_series = series.dropna()
        values = non_null_series.astype(str)

        checked_count = len(values)

        if checked_count > 0:
            patient_matches = int(values.str.contains(TCGA_PATIENT_REGEX).sum())
            sample_matches = int(values.str.contains(TCGA_SAMPLE_REGEX).sum())
        else:
            patient_matches = 0
            sample_matches = 0

        patient_match_ratio = (
            patient_matches / checked_count if checked_count > 0 else 0.0
        )

        sample_match_ratio = (
            sample_matches / checked_count if checked_count > 0 else 0.0
        )

        column_lower = str(column).lower()

        id_name_score = sum(
            1 for keyword in ID_NAME_KEYWORDS if keyword in column_lower
        )

        example_values = "; ".join(
            truncate_value(str(value)) for value in values.head(3)
        )

        rows.append(
            {
                "column": str(column),
                "dtype": str(series.dtype),
                "missing_ratio": round(float(series.isna().mean()), 6),
                "unique_count": int(series.nunique(dropna=True)),
                "checked_values": checked_count,
                "tcga_patient_matches": patient_matches,
                "tcga_patient_match_ratio": round(patient_match_ratio, 6),
                "tcga_sample_matches": sample_matches,
                "tcga_sample_match_ratio": round(sample_match_ratio, 6),
                "id_name_score": id_name_score,
                "example_values": example_values,
            }
        )

    inspection_df = pd.DataFrame(rows)

    inspection_df = inspection_df.sort_values(
        by=[
            "tcga_patient_matches",
            "tcga_sample_matches",
            "id_name_score",
            "unique_count",
        ],
        ascending=False,
    )

    return inspection_df


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
        output_dir = project_root / "artifacts" / "target_report"

    output_dir = ensure_dir(output_dir)

    clinical_dataset = find_clinical_dataset(config)
    clinical_path = Path(clinical_dataset.get("resolved_path", ""))

    if not clinical_path.exists():
        print(f"[FAILED] Clinical file not found: {clinical_path}")
        sys.exit(1)

    print(f"Reading clinical file: {clinical_path}")

    clinical_df = read_clinical_dataframe(clinical_path)

    print(
        f"Clinical file loaded: rows={clinical_df.shape[0]}, "
        f"columns={clinical_df.shape[1]}"
    )

    inspection_df = inspect_clinical_dataframe(clinical_df)

    output_file = output_dir / "clinical_id_inspection.csv"

    inspection_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"\nSaved ID inspection report to: {output_file}")

    print("\nTop identifier candidates:")

    top_candidates = inspection_df[
        (inspection_df["tcga_patient_matches"] > 0)
        | (inspection_df["tcga_sample_matches"] > 0)
        | (inspection_df["id_name_score"] > 0)
    ].head(20)

    if top_candidates.empty:
        print("No obvious TCGA patient/sample ID candidate was found.")
        print("Please send the full clinical_id_inspection.csv for review.")
    else:
        for _, row in top_candidates.iterrows():
            print(
                "- "
                f"column={row['column']} | "
                f"dtype={row['dtype']} | "
                f"patient_matches={row['tcga_patient_matches']} | "
                f"sample_matches={row['tcga_sample_matches']} | "
                f"unique={row['unique_count']} | "
                f"examples={row['example_values']}"
            )

    print("\nAll clinical columns:")

    for column in clinical_df.columns:
        print(f"- {column}")


if __name__ == "__main__":
    main()