"""Prepare 4-modality tumor/normal dataset (mRNA, miRNA, Methylation, CNA)."""

from pathlib import Path
import argparse
import sys
import json
import re

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.harmonizer import read_header_columns
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir

PATIENT_ID_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def extract_patient_id(barcode: str) -> str:
    match = PATIENT_ID_PATTERN.search(str(barcode))
    return match.group(1).upper() if match else str(barcode).strip().upper()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare 4-modality TN dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_raw_normals(raw_path, feature_col_name, logger):
    """Load normal samples from a raw features-x-samples file."""
    if not raw_path.exists():
        logger.error(f"Raw file not found: {raw_path}")
        return None, []

    header_cols, separator = read_header_columns(str(raw_path))
    logger.info(f"  Raw file: {raw_path.name}, cols={len(header_cols)}, sep='{separator}'")

    normal_cols = []
    for col in header_cols:
        col_str = str(col)
        if col_str.startswith("TCGA-"):
            parts = col_str.split("-")
            if len(parts) >= 4:
                try:
                    code_int = int(parts[3][:2])
                    if 10 <= code_int <= 19:
                        normal_cols.append(col)
                except ValueError:
                    pass

    logger.info(f"  Normal columns found: {len(normal_cols)}")
    if len(normal_cols) == 0:
        return None, []

    feature_col = header_cols[0]
    use_cols = [feature_col] + normal_cols

    df_raw = pd.read_csv(
        raw_path, sep=separator, usecols=use_cols,
        encoding="utf-8-sig", low_memory=False,
    )

    normals = df_raw.set_index(feature_col)[normal_cols].T
    normals.index.name = "sample_id"
    normals = normals.apply(pd.to_numeric, errors="coerce")

    # Normalize index to patient-level IDs
    normals.index = [extract_patient_id(idx) for idx in normals.index]
    normals = normals[~normals.index.duplicated(keep="first")]

    logger.info(f"  Normal matrix: {normals.shape}")
    return normals, normal_cols


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
        name="luad.prepare_4mod_tn",
        log_file=project_root / "artifacts" / "logs" / "prepare_4modality_tn.log",
        level="INFO", console=True,
    )

    output_dir = resolve_path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "four_modality_tn"
    output_dir = ensure_dir(output_dir)

    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    interim_dir = project_root / config.get("paths", {}).get("interim_dir", "data/interim")

    # ================================================================
    # STEP 1: Load mRNA tumor+normal (already prepared)
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading mRNA tumor+normal")
    logger.info("=" * 60)

    tn_path = project_root / "data" / "processed" / "tumor_normal" / "tumor_normal_combined.parquet"
    mrna_tn = pd.read_parquet(tn_path)
    if "sample_id" in mrna_tn.columns:
        mrna_tn = mrna_tn.set_index("sample_id")

    label_map = {"NORMAL": 0, "TUMOR": 1}
    mrna_labels = mrna_tn["label"].map(label_map).astype(int)
    mrna_X = mrna_tn.drop(columns=["label"])

    # Build patient-level mapping (prefer normal if patient has both)
    mrna_pid_map = {}
    for idx in mrna_X.index:
        pid = extract_patient_id(idx)
        label = int(mrna_labels[idx])
        if pid not in mrna_pid_map or label == 0:
            mrna_pid_map[pid] = (idx, label)

    logger.info(f"mRNA: {mrna_X.shape}, Patients: {len(mrna_pid_map)}")
    n_tumor_mrna = sum(1 for _, (_, l) in mrna_pid_map.items() if l == 1)
    n_normal_mrna = sum(1 for _, (_, l) in mrna_pid_map.items() if l == 0)
    logger.info(f"  Tumor patients: {n_tumor_mrna}, Normal patients: {n_normal_mrna}")

    # ================================================================
    # STEP 2: Load miRNA tumor+normal (already prepared)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading miRNA tumor+normal")
    logger.info("=" * 60)

    mirna_tn_dir = project_root / "data" / "processed" / "mrna_mirna_tn"
    mirna_labels_path = mirna_tn_dir / "labels.csv"
    mirna_data_path = mirna_tn_dir / "mirna.parquet"

    if mirna_labels_path.exists() and mirna_data_path.exists():
        mirna_labels_df = pd.read_csv(mirna_labels_path).set_index("sample_id")
        mirna_X_all = pd.read_parquet(mirna_data_path)
        if "sample_id" in mirna_X_all.columns:
            mirna_X_all = mirna_X_all.set_index("sample_id")
        if "label" in mirna_X_all.columns:
            mirna_X_all = mirna_X_all.drop(columns=["label"])

        mirna_pid_map = {}
        for pid in mirna_labels_df.index:
            label = int(mirna_labels_df.loc[pid, "label"])
            if pid in mirna_X_all.index:
                if pid not in mirna_pid_map or label == 0:
                    mirna_pid_map[pid] = (pid, label)

        logger.info(f"miRNA loaded: {mirna_X_all.shape}, Patients: {len(mirna_pid_map)}")
    else:
        logger.warning("miRNA TN data not found. Will skip miRNA.")
        mirna_pid_map = {}
        mirna_X_all = None

    # ================================================================
    # STEP 3: Load Methylation tumor+normal
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Loading Methylation tumor+normal")
    logger.info("=" * 60)

    # Tumor methylation from interim
    meth_tumor_path = interim_dir / "methylation.patient_features.parquet"
    if meth_tumor_path.exists():
        meth_tumor = pd.read_parquet(meth_tumor_path)
        if "patient_id" in meth_tumor.columns:
            meth_tumor = meth_tumor.set_index("patient_id")
        meth_tumor.index = [extract_patient_id(idx) for idx in meth_tumor.index]
        meth_tumor = meth_tumor[~meth_tumor.index.duplicated(keep="first")]
        logger.info(f"Methylation tumor: {meth_tumor.shape}")
    else:
        logger.warning("Methylation tumor data not found in interim.")
        meth_tumor = None

    # Normal methylation from raw
    meth_raw_path = raw_dir / "data_methylation_hm27_hm450_merged.txt"
    meth_normal, _ = load_raw_normals(meth_raw_path, "Composite Element REF", logger)

    # Combine tumor + normal methylation
    meth_pid_map = {}
    if meth_tumor is not None:
        for pid in meth_tumor.index:
            meth_pid_map[pid] = ("tumor", 1)
    if meth_normal is not None:
        for pid in meth_normal.index:
            if pid not in meth_pid_map:
                meth_pid_map[pid] = ("normal", 0)

    logger.info(f"Methylation patients: {len(meth_pid_map)}")

    # ================================================================
    # STEP 4: Load CNA tumor+normal
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Loading CNA tumor+normal")
    logger.info("=" * 60)

    cna_tumor_path = interim_dir / "cna_raw.patient_features.parquet"
    if cna_tumor_path.exists():
        cna_tumor = pd.read_parquet(cna_tumor_path)
        if "patient_id" in cna_tumor.columns:
            cna_tumor = cna_tumor.set_index("patient_id")
        cna_tumor.index = [extract_patient_id(idx) for idx in cna_tumor.index]
        cna_tumor = cna_tumor[~cna_tumor.index.duplicated(keep="first")]
        logger.info(f"CNA tumor: {cna_tumor.shape}")
    else:
        logger.warning("CNA tumor data not found in interim.")
        cna_tumor = None

    cna_raw_path = raw_dir / "data_cna.txt"
    cna_normal, _ = load_raw_normals(cna_raw_path, "Sample", logger)

    cna_pid_map = {}
    if cna_tumor is not None:
        for pid in cna_tumor.index:
            cna_pid_map[pid] = ("tumor", 1)
    if cna_normal is not None:
        for pid in cna_normal.index:
            if pid not in cna_pid_map:
                cna_pid_map[pid] = ("normal", 0)

    logger.info(f"CNA patients: {len(cna_pid_map)}")

    # ================================================================
    # STEP 5: Find common patients across all 4 modalities
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Finding common patients")
    logger.info("=" * 60)

    all_pid_sets = [set(mrna_pid_map.keys())]
    if mirna_pid_map:
        all_pid_sets.append(set(mirna_pid_map.keys()))
    if meth_pid_map:
        all_pid_sets.append(set(meth_pid_map.keys()))
    if cna_pid_map:
        all_pid_sets.append(set(cna_pid_map.keys()))

    common_pids = sorted(set.intersection(*all_pid_sets))
    logger.info(f"Common patients across all modalities: {len(common_pids)}")

    # Determine labels from mRNA (ground truth)
    final_labels = {}
    for pid in common_pids:
        _, label = mrna_pid_map[pid]
        final_labels[pid] = label

    n_tumor = sum(1 for v in final_labels.values() if v == 1)
    n_normal = sum(1 for v in final_labels.values() if v == 0)
    logger.info(f"Final: Tumor={n_tumor}, Normal={n_normal}")

    if n_normal == 0:
        logger.error("No normal samples in common set!")
        sys.exit(1)

    # ================================================================
    # STEP 6: Build aligned matrices
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Building aligned matrices")
    logger.info("=" * 60)

    # mRNA
    mrna_rows = []
    for pid in common_pids:
        sample_idx, _ = mrna_pid_map[pid]
        mrna_rows.append(mrna_X.loc[sample_idx])
    mrna_final = pd.DataFrame(mrna_rows, index=common_pids)
    logger.info(f"mRNA final: {mrna_final.shape}")

    # miRNA
    if mirna_X_all is not None and mirna_pid_map:
        mirna_rows = []
        for pid in common_pids:
            if pid in mirna_pid_map:
                sample_idx, _ = mirna_pid_map[pid]
                if sample_idx in mirna_X_all.index:
                    mirna_rows.append(mirna_X_all.loc[sample_idx])
                else:
                    mirna_rows.append(pd.Series(np.nan, index=mirna_X_all.columns))
            else:
                mirna_rows.append(pd.Series(np.nan, index=mirna_X_all.columns))
        mirna_final = pd.DataFrame(mirna_rows, index=common_pids)
        logger.info(f"miRNA final: {mirna_final.shape}")
    else:
        mirna_final = None

    # Methylation
    if meth_tumor is not None and meth_normal is not None:
        meth_rows = []
        meth_features = None
        for pid in common_pids:
            if pid in meth_tumor.index:
                row = meth_tumor.loc[pid]
                if meth_features is None:
                    meth_features = row.index
                meth_rows.append(row)
            elif pid in meth_normal.index:
                row = meth_normal.loc[pid]
                if meth_features is None:
                    meth_features = row.index
                meth_rows.append(row)
            else:
                meth_rows.append(pd.Series(np.nan, index=meth_features))
        meth_final = pd.DataFrame(meth_rows, index=common_pids)
        logger.info(f"Methylation final: {meth_final.shape}")
    else:
        meth_final = None

    # CNA
    if cna_tumor is not None and cna_normal is not None:
        cna_rows = []
        cna_features = None
        for pid in common_pids:
            if pid in cna_tumor.index:
                row = cna_tumor.loc[pid]
                if cna_features is None:
                    cna_features = row.index
                cna_rows.append(row)
            elif pid in cna_normal.index:
                row = cna_normal.loc[pid]
                if cna_features is None:
                    cna_features = row.index
                cna_rows.append(row)
            else:
                cna_rows.append(pd.Series(np.nan, index=cna_features))
        cna_final = pd.DataFrame(cna_rows, index=common_pids)
        logger.info(f"CNA final: {cna_final.shape}")
    else:
        cna_final = None

    # ================================================================
    # STEP 7: Save outputs
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Saving outputs")
    logger.info("=" * 60)

    # Labels
    labels_df = pd.DataFrame({
        "sample_id": common_pids,
        "label": [final_labels[pid] for pid in common_pids],
    })
    labels_df.to_csv(output_dir / "labels.csv", index=False, encoding="utf-8-sig")

    # mRNA
    mrna_out = mrna_final.copy()
    mrna_out.index.name = "sample_id"
    mrna_out.to_parquet(output_dir / "mrna.parquet")

    # miRNA
    if mirna_final is not None:
        mirna_out = mirna_final.copy()
        mirna_out.index.name = "sample_id"
        mirna_out.to_parquet(output_dir / "mirna.parquet")

    # Methylation
    if meth_final is not None:
        meth_out = meth_final.copy()
        meth_out.index.name = "sample_id"
        meth_out.to_parquet(output_dir / "methylation.parquet")

    # CNA
    if cna_final is not None:
        cna_out = cna_final.copy()
        cna_out.index.name = "sample_id"
        cna_out.to_parquet(output_dir / "cna.parquet")

    # Metadata
    metadata = {
        "n_patients": len(common_pids),
        "n_tumor": n_tumor,
        "n_normal": n_normal,
        "modalities": [],
    }
    metadata["modalities"].append({"name": "mrna", "features": mrna_final.shape[1]})
    if mirna_final is not None:
        metadata["modalities"].append({"name": "mirna", "features": mirna_final.shape[1]})
    if meth_final is not None:
        metadata["modalities"].append({"name": "methylation", "features": meth_final.shape[1]})
    if cna_final is not None:
        metadata["modalities"].append({"name": "cna", "features": cna_final.shape[1]})

    with (output_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("4-MODALITY TUMOR/NORMAL DATASET READY")
    print("=" * 60)
    print(f"Patients: {len(common_pids)} (Tumor={n_tumor}, Normal={n_normal})")
    for m in metadata["modalities"]:
        print(f"  {m['name']}: {m['features']} features")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()