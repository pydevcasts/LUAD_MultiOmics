"""Prepare tumor vs normal classification dataset."""

from pathlib import Path
import argparse
import sys
import json

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.normal_loader import (
    load_normals_matrix,
    align_normals_with_tumor,
)
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare tumor vs normal classification dataset."
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
        help="Output directory for processed data.",
    )

    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


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

    logger = get_logger(
        name="luad.tumor_normal",
        log_file=project_root / "artifacts" / "logs" / "tumor_normal_prep.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    interim_dir = project_root / config.get("paths", {}).get(
        "interim_dir",
        "data/interim",
    )

    if args.output_dir is not None:
        output_dir = resolve_path(args.output_dir)
    else:
        output_dir = project_root / "data" / "processed" / "tumor_normal"

    output_dir = ensure_dir(output_dir)

    logger.info("Starting tumor vs normal dataset preparation.")

    tumor_mrna_path = interim_dir / "mrna_rsem.patient_features.parquet"

    if not tumor_mrna_path.exists():
        logger.error(f"Tumor mRNA matrix not found: {tumor_mrna_path}")
        logger.error("Run prepare_data.py first.")
        sys.exit(1)

    logger.info(f"Loading tumor mRNA from: {tumor_mrna_path}")
    tumor_matrix = pd.read_parquet(tumor_mrna_path)

    if "patient_id" in tumor_matrix.columns:
        tumor_matrix = tumor_matrix.set_index("patient_id")

    if tumor_matrix.index.name != "sample_id":
        tumor_matrix.index.name = "sample_id"

    tumor_matrix.index = tumor_matrix.index.astype(str)

    logger.info(f"Tumor matrix shape: {tumor_matrix.shape}")
    logger.info(f"Tumor samples: {tumor_matrix.shape[0]}")
    logger.info(f"Tumor features: {tumor_matrix.shape[1]}")
    logger.info(f"Tumor feature names sample: {list(tumor_matrix.columns[:5])}")

    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    normals_path = raw_dir / "data_normals_RNA_Seq_v2_mRNA_median.txt"

    if not normals_path.exists():
        logger.error(f"Normals file not found: {normals_path}")
        sys.exit(1)

    normals_matrix, feature_id_col = load_normals_matrix(
        normals_path=normals_path,
        logger=logger,
    )

    normals_aligned, tumor_aligned, common_features = align_normals_with_tumor(
        normals_matrix=normals_matrix,
        tumor_matrix=tumor_matrix,
        logger=logger,
    )

    normals_aligned["label"] = "NORMAL"
    tumor_aligned["label"] = "TUMOR"

    combined = pd.concat([tumor_aligned, normals_aligned], axis=0)

    tumor_count = int((combined["label"] == "TUMOR").sum())
    normal_count = int((combined["label"] == "NORMAL").sum())

    logger.info(f"Combined dataset shape: {combined.shape}")
    logger.info(f"  Tumor samples: {tumor_count}")
    logger.info(f"  Normal samples: {normal_count}")
    logger.info(f"  Total features: {len(common_features)}")

    combined_path = output_dir / "tumor_normal_combined.parquet"
    combined.to_parquet(combined_path)

    labels_df = combined[["label"]].copy()
    labels_df.index.name = "sample_id"
    labels_path = output_dir / "tumor_normal_labels.csv"
    labels_df.to_csv(labels_path, encoding="utf-8-sig")

    metadata = {
        "tumor_samples": tumor_count,
        "normal_samples": normal_count,
        "total_samples": int(combined.shape[0]),
        "total_features": len(common_features),
        "feature_id_column_in_normals": feature_id_col,
        "combined_path": str(combined_path),
        "labels_path": str(labels_path),
    }

    metadata_path = output_dir / "tumor_normal_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"Combined dataset saved to: {combined_path}")
    logger.info(f"Labels saved to: {labels_path}")
    logger.info(f"Metadata saved to: {metadata_path}")

    print("\n=== Tumor vs Normal Dataset Summary ===")
    print(f"Tumor samples:  {metadata['tumor_samples']}")
    print(f"Normal samples: {metadata['normal_samples']}")
    print(f"Total samples:  {metadata['total_samples']}")
    print(f"Total features: {metadata['total_features']}")
    print(f"Output dir:     {output_dir}")


if __name__ == "__main__":
    main()