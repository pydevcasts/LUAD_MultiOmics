"""Build patient-level cohort manifest after target confirmation.

This script does not train models. It only harmonizes patients and samples.
"""

from pathlib import Path
import argparse
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.harmonizer import build_cohort
from src.luad.utils.io import ensure_dir, save_json, save_text
from src.luad.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build LUAD patient/sample cohort manifest."
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
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Defaults to artifacts/cohort.",
    )

    parser.add_argument(
        "--skip-mutation",
        action="store_true",
        help="Skip full mutation scan.",
    )

    parser.add_argument(
        "--mutation-preview-rows",
        type=int,
        default=None,
        help="Number of mutation rows to inspect for schema detection.",
    )

    parser.add_argument(
        "--mutation-chunksize",
        type=int,
        default=None,
        help="Chunk size for streaming mutation file.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def count_with_all(manifest: pd.DataFrame, modality_keys: list) -> int | None:
    columns = []

    for key in modality_keys:
        column_name = f"has_{key}"

        if column_name not in manifest.columns:
            return None

        columns.append(column_name)

    if not columns:
        return 0

    return int(manifest[columns].all(axis=1).sum())


def build_markdown_report(
    summary: dict,
    intersection_counts: dict,
    mutation_schema: dict,
) -> str:
    lines = []

    lines.append("# Cohort Harmonization Report")
    lines.append("")
    lines.append(f"- Clinical rows with stage: {summary.get('clinical_rows_with_stage')}")
    lines.append("")

    lines.append("## Target Distribution")
    lines.append("")
    lines.append("| Stage | Count |")
    lines.append("|---|---:|")

    target_distribution = summary.get("target_distribution", {})

    for stage, count in target_distribution.items():
        lines.append(f"| {stage} | {count} |")

    lines.append("")

    lines.append("## Modality Patient Counts")
    lines.append("")
    lines.append("| Modality Key | Patients | Status | Multi-sample Patients |")
    lines.append("|---|---:|---|---:|")

    modality_patient_counts = summary.get("modality_patient_counts", {})
    modality_status = summary.get("modality_status", {})
    multi_sample_counts = summary.get("multi_sample_modality_counts", {})

    for key in modality_patient_counts:
        lines.append(
            "| "
            f"{key} | "
            f"{modality_patient_counts[key]} | "
            f"{modality_status.get(key, '')} | "
            f"{multi_sample_counts.get(key, 0)} |"
        )

    lines.append("")

    lines.append("## Intersection Counts")
    lines.append("")
    lines.append("| Cohort Definition | Patients |")
    lines.append("|---|---:|")

    for name, count in intersection_counts.items():
        if count is None:
            display_count = "N/A"
        else:
            display_count = str(count)

        lines.append(f"| {name} | {display_count} |")

    lines.append("")

    lines.append("## Mutation Schema")
    lines.append("")
    lines.append(f"- Status: {mutation_schema.get('status')}")
    lines.append(f"- Sample column: {mutation_schema.get('sample_column')}")
    lines.append(f"- Gene column: {mutation_schema.get('gene_column')}")
    lines.append(f"- Mutation patient count: {mutation_schema.get('mutation_patient_count')}")
    lines.append("")

    if mutation_schema.get("errors"):
        lines.append("### Mutation Errors")
        lines.append("")

        for error in mutation_schema["errors"]:
            lines.append(f"- {error}")

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
        output_dir = project_root / "artifacts" / "cohort"

    output_dir = ensure_dir(output_dir)

    logger = get_logger(
        name="luad.harmonize",
        log_file=project_root / "artifacts" / "logs" / "harmonization.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    logger.info("Starting cohort harmonization.")

    try:
        cohort_output = build_cohort(
            config=config,
            logger=logger,
            skip_mutation=args.skip_mutation,
            mutation_preview_rows=args.mutation_preview_rows,
            mutation_chunksize=args.mutation_chunksize,
        )
    except Exception as exc:
        logger.error(f"Cohort harmonization failed: {exc}")
        raise

    manifest: pd.DataFrame = cohort_output["manifest"]
    sample_map_rows: list = cohort_output["sample_map_rows"]
    mutation_schema: dict = cohort_output["mutation_schema"]
    summary: dict = cohort_output["summary"]

    manifest = manifest.sort_values("patient_id").reset_index(drop=True)

    manifest_path = output_dir / "patient_manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    sample_map_df = pd.DataFrame(sample_map_rows)

    sample_map_path = output_dir / "sample_map.csv"
    sample_map_df.to_csv(sample_map_path, index=False, encoding="utf-8-sig")

    intersection_counts = {
        "stage_only": int(manifest.shape[0]),
        "mrna": count_with_all(manifest, ["mrna_rsem"]),
        "mrna_mirna": count_with_all(manifest, ["mrna_rsem", "mirna"]),
        "mrna_mirna_methylation": count_with_all(
            manifest,
            ["mrna_rsem", "mirna", "methylation"],
        ),
        "mrna_mirna_methylation_cna_raw": count_with_all(
            manifest,
            ["mrna_rsem", "mirna", "methylation", "cna_raw"],
        ),
        "mrna_mirna_methylation_cna_raw_rppa_zscores": count_with_all(
            manifest,
            ["mrna_rsem", "mirna", "methylation", "cna_raw", "rppa_zscores"],
        ),
        "mrna_mirna_methylation_cna_raw_mutations": count_with_all(
            manifest,
            ["mrna_rsem", "mirna", "methylation", "cna_raw", "mutations"],
        ),
    }

    summary["intersection_counts"] = intersection_counts

    save_json(output_dir / "cohort_summary.json", summary)
    save_json(output_dir / "mutation_schema.json", mutation_schema)

    markdown_report = build_markdown_report(
        summary=summary,
        intersection_counts=intersection_counts,
        mutation_schema=mutation_schema,
    )

    save_text(output_dir / "HARMONIZATION_REPORT.md", markdown_report)

    logger.info(f"Cohort outputs saved to: {output_dir}")

    print("\n=== Cohort Harmonization Summary ===")
    print(f"Patients with stage: {summary['clinical_rows_with_stage']}")

    print("\nTarget distribution:")

    for stage, count in summary["target_distribution"].items():
        print(f"- {stage}: {count}")

    print("\nModality patient counts:")

    for key, count in summary["modality_patient_counts"].items():
        print(f"- {key}: {count}")

    print("\nIntersection counts:")

    for name, count in intersection_counts.items():
        print(f"- {name}: {count}")

    print("\nMutation schema:")
    print(f"- status: {mutation_schema.get('status')}")
    print(f"- sample column: {mutation_schema.get('sample_column')}")
    print(f"- gene column: {mutation_schema.get('gene_column')}")
    print(f"- mutation patient count: {mutation_schema.get('mutation_patient_count')}")


if __name__ == "__main__":
    main()