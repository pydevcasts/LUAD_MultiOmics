"""Inspect GSE131907 external validation dataset (header only, no full load)."""

from pathlib import Path
import argparse
import sys
import json
import re

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.harmonizer import read_header_columns
from src.luad.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect GSE131907 header.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-experiment", type=str, default="pso_tumor_normal_v3")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_path = resolve_path(args.experiment) if args.experiment else None

    try:
        config = load_config(config_path=config_path, experiment_path=experiment_path)
    except ConfigError as exc:
        print(f"[FAILED] {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    logger = get_logger(
        name="luad.gse_inspect",
        log_file=project_root / "artifacts" / "logs" / "gse131907_inspect.log",
        level="INFO", console=True,
    )

    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    gse_path = raw_dir / "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt"
    if not gse_path.exists():
        logger.error(f"GSE131907 not found: {gse_path}")
        sys.exit(1)

    file_size_mb = gse_path.stat().st_size / (1024 * 1024)
    logger.info(f"GSE131907 file size: {file_size_mb:.1f} MB")

    # ===== Read Header Only =====
    logger.info("Reading header (first line only)...")
    header_cols, separator = read_header_columns(str(gse_path))
    logger.info(f"Separator: '{separator}'")
    logger.info(f"Total columns: {len(header_cols)}")

    # First column is usually gene ID
    feature_id_col = header_cols[0]
    sample_cols = header_cols[1:]
    logger.info(f"Feature ID column: '{feature_id_col}'")
    logger.info(f"Sample columns: {len(sample_cols)}")
    logger.info(f"First 5 sample columns: {sample_cols[:5]}")
    logger.info(f"Last 5 sample columns: {sample_cols[-5:]}")

    # ===== Check sample barcode patterns =====
    tcga_pattern = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}")
    gsm_pattern = re.compile(r"GSM\d+")
    geo_pattern = re.compile(r"GSE\d+")

    tcga_samples = [c for c in sample_cols if tcga_pattern.search(c)]
    gsm_samples = [c for c in sample_cols if gsm_pattern.search(c)]

    logger.info(f"\nSample ID patterns:")
    logger.info(f"  TCGA barcodes: {len(tcga_samples)}")
    logger.info(f"  GSM IDs: {len(gsm_samples)}")
    logger.info(f"  Other: {len(sample_cols) - len(tcga_samples) - len(gsm_samples)}")

    if tcga_samples:
        logger.info(f"  TCGA samples: {tcga_samples[:5]}")
    if gsm_samples:
        logger.info(f"  GSM samples: {gsm_samples[:5]}")

    # ===== Read first few rows to understand feature IDs =====
    logger.info("\nReading first 10 data rows...")
    preview_df = pd.read_csv(
        gse_path, sep=separator, nrows=10,
        encoding="utf-8-sig", low_memory=False,
    )
    logger.info(f"Preview shape: {preview_df.shape}")
    logger.info(f"Columns: {list(preview_df.columns[:5])}")
    logger.info(f"Feature ID examples: {preview_df.iloc[:5, 0].tolist()}")

    # ===== Load PSO genes and check overlap =====
    pso_genes_path = project_root / "artifacts" / "experiments" / args.pso_experiment / "pso_selected_genes.csv"
    if pso_genes_path.exists():
        pso_genes = set(pd.read_csv(pso_genes_path)["gene"].tolist())
        logger.info(f"\nPSO genes loaded: {len(pso_genes)}")

        # Check overlap with GSE feature IDs
        gse_feature_ids = set(preview_df.iloc[:, 0].astype(str).tolist())
        direct_overlap = pso_genes & gse_feature_ids
        logger.info(f"Direct overlap (first 10 rows): {len(direct_overlap)}")

        # Also check case-insensitive
        gse_upper = {str(x).upper() for x in gse_feature_ids}
        pso_upper = {x.upper() for x in pso_genes}
        upper_overlap = pso_upper & gse_upper
        logger.info(f"Case-insensitive overlap (first 10 rows): {len(upper_overlap)}")
    else:
        logger.warning(f"PSO genes not found: {pso_genes_path}")

    # ===== Check if there's a metadata/label file =====
    possible_label_files = [
        raw_dir / "GSE131907_metadata.txt",
        raw_dir / "GSE131907_series_matrix.txt",
        raw_dir / "GSE131907_labels.csv",
        raw_dir / "GSE131907_sample_info.txt",
    ]

    logger.info("\nSearching for metadata/label files:")
    found_metadata = []
    for lf in possible_label_files:
        exists = lf.exists()
        logger.info(f"  {lf.name}: {'✅ FOUND' if exists else '❌ not found'}")
        if exists:
            found_metadata.append(str(lf))

    # ===== Summary =====
    summary = {
        "file_path": str(gse_path),
        "file_size_mb": round(file_size_mb, 1),
        "total_columns": len(header_cols),
        "feature_id_column": feature_id_col,
        "n_samples": len(sample_cols),
        "separator": separator,
        "tcga_samples": len(tcga_samples),
        "gsm_samples": len(gsm_samples),
        "feature_id_examples": preview_df.iloc[:5, 0].tolist(),
        "metadata_files_found": found_metadata,
    }

    output_dir = project_root / "artifacts" / "external_validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "gse131907_inspection.json").open("w") as f:
        json.dump(summary, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("GSE131907 INSPECTION SUMMARY")
    print("=" * 60)
    print(f"File size:       {file_size_mb:.1f} MB")
    print(f"Total columns:   {len(header_cols)}")
    print(f"Samples:         {len(sample_cols)}")
    print(f"Feature column:  {feature_id_col}")
    print(f"TCGA samples:    {len(tcga_samples)}")
    print(f"GSM samples:     {len(gsm_samples)}")
    print(f"Metadata files:  {len(found_metadata)}")
    print(f"Feature examples: {preview_df.iloc[:3, 0].tolist()}")
    print("=" * 60)
    print(f"\nInspection saved to: {output_dir / 'gse131907_inspection.json'}")


if __name__ == "__main__":
    main()