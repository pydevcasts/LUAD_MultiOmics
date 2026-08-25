"""Confirm and report the candidate target from clinical data.

This script reads the clinical dataset, profiles columns, evaluates the proposed
target, applies optional stage grouping, and produces a target report.

It does NOT train models.
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
from src.luad.data.target import (
    DEFAULT_STAGE_MAPPING,
    map_target_series,
    summarize_categorical_series,
)
from src.luad.utils.io import ensure_dir, save_json, save_text
from src.luad.utils.logger import get_logger


DEFAULT_TARGET_COLUMN = "AJCC_PATHOLOGIC_TUMOR_STAGE"

DEFAULT_LEAKAGE_COLUMNS = [
    "PATH_T_STAGE",
    "PATH_N_STAGE",
    "PATH_M_STAGE",
    "AJCC_STAGING_EDITION",
    "OS_STATUS",
    "OS_MONTHS",
    "DFS_STATUS",
    "DFS_MONTHS",
    "DAYS_LAST_FOLLOWUP",
    "NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT",
    "RADIATION_THERAPY",
    "CHEMOTHERAPY",
    "TARGETED_MOLECULAR_THERAPY",
    "SYSTEMIC_THERAPY",
    "TREATMENT_OUTCOME",
    "TREATMENT_TYPE",
    "TREATMENT_RESPONSE",
    "SUBTYPE",
    "DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS",
]

TCGA_PATIENT_REGEX = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Confirm candidate target from clinical data."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config YAML file.",
    )

    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Optional experiment overlay YAML file.",
    )

    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Optional target column. Overrides config task.target.",
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


def truncate_value(value: str, max_chars: int = 80) -> str:
    if len(value) <= max_chars:
        return value

    return value[:max_chars] + "..."


def profile_clinical_columns(df: pd.DataFrame) -> list:
    rows = []

    for column in df.columns:
        series = df[column]

        dtype = str(series.dtype)
        missing_ratio = float(series.isna().mean())
        unique_count = int(series.nunique(dropna=True))

        top_values = ""

        if unique_count <= 25:
            value_counts = series.astype(str).value_counts(dropna=True).head(10)
        else:
            value_counts = series.astype(str).value_counts(dropna=True).head(5)

        if len(value_counts) > 0:
            top_values = "; ".join(
                f"{truncate_value(str(index))}: {int(count)}"
                for index, count in value_counts.items()
            )

        rows.append(
            {
                "column": str(column),
                "dtype": dtype,
                "missing_ratio": round(missing_ratio, 6),
                "unique_count": unique_count,
                "top_values": top_values,
            }
        )

    return rows


def detect_patient_id_candidates(df: pd.DataFrame, max_values: int = 5000) -> list:
    candidates = []

    for column in df.columns:
        if str(df[column].dtype) != "object":
            continue

        values = df[column].dropna().astype(str).head(max_values)

        if len(values) == 0:
            continue

        match_count = 0

        for value in values:
            if TCGA_PATIENT_REGEX.search(value):
                match_count += 1

        if match_count > 0:
            candidates.append(
                {
                    "column": str(column),
                    "tcga_patient_matches": int(match_count),
                    "checked_values": int(len(values)),
                    "match_ratio": round(match_count / len(values), 6),
                }
            )

    candidates.sort(key=lambda item: item["tcga_patient_matches"], reverse=True)

    return candidates


def save_csv(rows: list, path: Path, columns: list) -> None:
    if rows:
        dataframe = pd.DataFrame(rows)
    else:
        dataframe = pd.DataFrame(columns=columns)

    dataframe.to_csv(path, index=False, encoding="utf-8-sig")


def build_target_report_markdown(report: dict) -> str:
    lines = []

    lines.append("# Target Confirmation Report")
    lines.append("")
    lines.append(f"- Clinical file: `{report['clinical_file']}`")
    lines.append(f"- Rows: {report['clinical_rows']}")
    lines.append(f"- Columns: {report['clinical_columns']}")
    lines.append(f"- Target column: `{report['target_column']}`")
    lines.append(f"- Target found: {report['target_found']}")
    lines.append("")

    lines.append("## Raw Target Summary")
    lines.append("")

    raw_summary = report.get("raw_target_summary", {})

    lines.append(f"- Missing count: {raw_summary.get('missing_count')}")
    lines.append(f"- Missing ratio: {raw_summary.get('missing_ratio')}")
    lines.append(f"- Unique count: {raw_summary.get('unique_count')}")
    lines.append("")

    lines.append("### Raw target counts")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---:|")

    for class_name, count in raw_summary.get("counts", {}).items():
        lines.append(f"| {class_name} | {count} |")

    lines.append("")
    lines.append("## Mapped/Grouped Target Summary")
    lines.append("")

    mapped_summary = report.get("mapped_target_summary", {})

    lines.append(f"- Missing count: {mapped_summary.get('missing_count')}")
    lines.append(f"- Missing ratio: {mapped_summary.get('missing_ratio')}")
    lines.append(f"- Unique count: {mapped_summary.get('unique_count')}")
    lines.append("")

    lines.append("### Grouped target counts")
    lines.append("")
    lines.append("| Class | Count |")
    lines.append("|---|---:|")

    for class_name, count in mapped_summary.get("counts", {}).items():
        lines.append(f"| {class_name} | {count} |")

    lines.append("")

    if report.get("low_count_classes"):
        lines.append("## Low-count Classes")
        lines.append("")
        lines.append("| Class | Count | Minimum Required |")
        lines.append("|---|---:|---:|")

        minimum_class_count = report.get("minimum_class_count")

        for class_name, count in report["low_count_classes"].items():
            lines.append(f"| {class_name} | {count} | {minimum_class_count} |")

        lines.append("")

    if report.get("outside_allowed_classes"):
        lines.append("## Classes Outside Allowed Classes")
        lines.append("")
        lines.append("| Class | Count |")
        lines.append("|---|---:|")

        for class_name, count in report["outside_allowed_classes"].items():
            lines.append(f"| {class_name} | {count} |")

        lines.append("")

    lines.append("## Patient ID Candidates")
    lines.append("")

    patient_id_candidates = report.get("patient_id_candidates", [])

    if patient_id_candidates:
        lines.append("| Column | TCGA Matches | Checked Values | Match Ratio |")
        lines.append("|---|---:|---:|---:|")

        for candidate in patient_id_candidates:
            lines.append(
                "| "
                f"{candidate['column']} | "
                f"{candidate['tcga_patient_matches']} | "
                f"{candidate['checked_values']} | "
                f"{candidate['match_ratio']} |"
            )
    else:
        lines.append("No TCGA patient ID candidates detected.")

    lines.append("")

    lines.append("## Leakage Columns Present in Clinical File")
    lines.append("")

    leakage_present = report.get("leakage_columns_present", [])

    if leakage_present:
        for column in leakage_present:
            lines.append(f"- {column}")
    else:
        lines.append("No configured leakage columns were found.")

    lines.append("")

    lines.append("## Constant or Near-Constant Columns")
    lines.append("")

    constant_columns = report.get("constant_columns", [])

    if constant_columns:
        for item in constant_columns:
            lines.append(
                f"- {item['column']} | unique_count={item['unique_count']} | top_values={item['top_values']}"
            )
    else:
        lines.append("No constant columns detected.")

    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    lines.append(report.get("recommendation", ""))
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    config_path = resolve_path(args.config)

    experiment_path = None

    if args.experiment is not None:
        experiment_path = resolve_path(args.experiment)

    try:
        config = load_config(
            config_path=config_path,
            experiment_path=experiment_path,
        )
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "artifacts" / "target_report"

    output_dir = ensure_dir(output_dir)

    logger = get_logger(
        name="luad.target",
        log_file=project_root / "artifacts" / "logs" / "target_confirmation.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    logger.info("Starting target confirmation.")

    clinical_dataset = find_clinical_dataset(config)
    clinical_path = Path(clinical_dataset.get("resolved_path", ""))

    if not clinical_path.exists():
        logger.error(f"Clinical file not found: {clinical_path}")
        sys.exit(1)

    logger.info(f"Reading clinical file: {clinical_path}")

    clinical_df = read_clinical_dataframe(clinical_path)

    logger.info(
        f"Clinical data loaded: rows={clinical_df.shape[0]}, columns={clinical_df.shape[1]}"
    )

    task_config = config.get("task", {})
    target_column = args.target or task_config.get("target") or DEFAULT_TARGET_COLUMN

    target_found = target_column in clinical_df.columns

    if not target_found:
        logger.error(f"Target column '{target_column}' not found.")
        logger.error("Available columns:")

        for column in clinical_df.columns:
            logger.error(f"- {column}")

        sys.exit(1)

    logger.info(f"Target column: {target_column}")

    target_preprocessing_config = task_config.get("target_preprocessing", {})

    uppercase = bool(target_preprocessing_config.get("uppercase", True))
    strip = bool(target_preprocessing_config.get("strip", True))
    mapping = target_preprocessing_config.get("mapping", DEFAULT_STAGE_MAPPING)

    raw_target_series = clinical_df[target_column]
    raw_target_summary = summarize_categorical_series(raw_target_series)

    mapped_target_series = map_target_series(
        series=raw_target_series,
        mapping=mapping,
        uppercase=uppercase,
        strip=strip,
    )

    mapped_target_summary = summarize_categorical_series(mapped_target_series)

    minimum_class_count = int(config.get("labels", {}).get("minimum_class_count", 10))

    low_count_classes = {
        class_name: count
        for class_name, count in mapped_target_summary["counts"].items()
        if count < minimum_class_count
    }

    allowed_classes = task_config.get("allowed_classes", None)

    outside_allowed_classes = {}

    if allowed_classes is not None:
        allowed_classes_upper = {str(item).upper() for item in allowed_classes}

        outside_allowed_classes = {
            class_name: count
            for class_name, count in mapped_target_summary["counts"].items()
            if str(class_name).upper() not in allowed_classes_upper
        }

    clinical_column_profile = profile_clinical_columns(clinical_df)
    patient_id_candidates = detect_patient_id_candidates(clinical_df)

    leakage_columns_config = config.get("clinical_data", {}).get(
        "leakage_drop_columns",
        DEFAULT_LEAKAGE_COLUMNS,
    )

    leakage_columns_present = [
        column for column in leakage_columns_config if column in clinical_df.columns
    ]

    leakage_columns_missing = [
        column for column in leakage_columns_config if column not in clinical_df.columns
    ]

    constant_columns = []

    for row in clinical_column_profile:
        if row["unique_count"] <= 1:
            constant_columns.append(row)

    recommendation_lines = []

    if target_column == DEFAULT_TARGET_COLUMN:
        recommendation_lines.append(
            "The recommended primary target is AJCC_PATHOLOGIC_TUMOR_STAGE grouped into STAGE I/II/III/IV."
        )

    if low_count_classes:
        recommendation_lines.append(
            "Some classes have counts below the minimum threshold. "
            "Consider binary early/late staging or class-weighted evaluation."
        )

    if outside_allowed_classes:
        recommendation_lines.append(
            "Some mapped classes are outside allowed_classes. Review mapping or allowed_classes."
        )

    if not low_count_classes and not outside_allowed_classes:
        recommendation_lines.append(
            "Class counts are acceptable for multiclass staging with balanced metrics."
        )

    recommendation_lines.append(
        "For stage prediction, PATH_T_STAGE, PATH_N_STAGE, PATH_M_STAGE, OS/DFS fields, "
        "treatment/follow-up fields, and staging edition should be excluded to avoid leakage."
    )

    recommendation = " ".join(recommendation_lines)

    report = {
        "clinical_file": str(clinical_path),
        "clinical_rows": int(clinical_df.shape[0]),
        "clinical_columns": int(clinical_df.shape[1]),
        "target_column": target_column,
        "target_found": bool(target_found),
        "raw_target_summary": raw_target_summary,
        "mapped_target_summary": mapped_target_summary,
        "minimum_class_count": minimum_class_count,
        "low_count_classes": low_count_classes,
        "allowed_classes": allowed_classes,
        "outside_allowed_classes": outside_allowed_classes,
        "patient_id_candidates": patient_id_candidates,
        "leakage_columns_present": leakage_columns_present,
        "leakage_columns_missing": leakage_columns_missing,
        "constant_columns": constant_columns,
        "recommendation": recommendation,
    }

    save_json(output_dir / "target_report.json", report)

    save_csv(
        clinical_column_profile,
        output_dir / "clinical_column_profile.csv",
        columns=["column", "dtype", "missing_ratio", "unique_count", "top_values"],
    )

    mapped_counts_rows = [
        {"class": class_name, "count": count}
        for class_name, count in mapped_target_summary["counts"].items()
    ]

    save_csv(
        mapped_counts_rows,
        output_dir / "target_class_counts.csv",
        columns=["class", "count"],
    )

    markdown_text = build_target_report_markdown(report)
    save_text(output_dir / "TARGET_REPORT.md", markdown_text)

    logger.info(f"Target report saved to: {output_dir}")

    print("\n=== Target Confirmation Summary ===")
    print(f"Clinical rows: {report['clinical_rows']}")
    print(f"Clinical columns: {report['clinical_columns']}")
    print(f"Target column: {report['target_column']}")
    print(f"Missing target count: {mapped_target_summary['missing_count']}")
    print("Grouped target counts:")

    for class_name, count in mapped_target_summary["counts"].items():
        print(f"- {class_name}: {count}")

    if patient_id_candidates:
        print("\nTop patient ID candidates:")

        for candidate in patient_id_candidates[:5]:
            print(
                f"- {candidate['column']} | matches={candidate['tcga_patient_matches']} | ratio={candidate['match_ratio']}"
            )

    print("\nLeakage columns present:")

    for column in leakage_columns_present:
        print(f"- {column}")

    print("\nRecommendation:")
    print(recommendation)


if __name__ == "__main__":
    main()