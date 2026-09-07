"""Prepare combined mRNA + miRNA tumor/normal dataset (v4 - sample-level, no dedup)."""

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


SAMPLE_TYPE_PATTERN = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})")
PATIENT_ID_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def extract_patient_id(barcode: str) -> str:
    match = PATIENT_ID_PATTERN.search(str(barcode))
    return match.group(1).upper() if match else str(barcode).strip().upper()


def is_normal_sample(barcode: str) -> bool:
    match = SAMPLE_TYPE_PATTERN.search(str(barcode))
    if match:
        code = int(match.group(1))
        return 10 <= code <= 19
    return False


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare mRNA+miRNA TN dataset.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
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
        name="luad.prepare_mrna_mirna_tn",
        log_file=project_root / "artifacts" / "logs" / "prepare_mrna_mirna_tn.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    output_dir = resolve_path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "mrna_mirna_tn"
    output_dir = ensure_dir(output_dir)

    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir
    interim_dir = project_root / config.get("paths", {}).get("interim_dir", "data/interim")

    # ================================================================
    # STEP 1: Load mRNA tumor+normal — KEEP SAMPLE-LEVEL INDEX
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading mRNA tumor+normal (sample-level)")
    logger.info("=" * 60)

    tn_path = project_root / "data" / "processed" / "tumor_normal" / "tumor_normal_combined.parquet"
    if not tn_path.exists():
        logger.error(f"Not found: {tn_path}")
        sys.exit(1)

    mrna_tn = pd.read_parquet(tn_path)
    if "sample_id" in mrna_tn.columns:
        mrna_tn = mrna_tn.set_index("sample_id")

    label_map = {"NORMAL": 0, "TUMOR": 1}
    mrna_labels = mrna_tn["label"].map(label_map).astype(int)
    mrna_X = mrna_tn.drop(columns=["label"])

    # IMPORTANT: Do NOT convert to patient-level. Keep sample barcodes as index.
    # Each row is a unique sample (tumor OR normal), even if same patient.
    n_tumor = int((mrna_labels == 1).sum())
    n_normal = int((mrna_labels == 0).sum())
    logger.info(f"mRNA: {mrna_X.shape}, Tumor={n_tumor}, Normal={n_normal}")
    logger.info(f"Index samples: {list(mrna_X.index[:5])}")

    if n_normal == 0:
        logger.error("No normal samples in mRNA!")
        sys.exit(1)

    # Build patient_id series for later matching (but don't change index)
    mrna_patient_ids = pd.Series([extract_patient_id(idx) for idx in mrna_X.index], index=mrna_X.index)

    # ================================================================
    # STEP 2: Load miRNA tumor
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading miRNA tumor")
    logger.info("=" * 60)

    mirna_tumor_path = interim_dir / "mirna.patient_features.parquet"
    if not mirna_tumor_path.exists():
        logger.error(f"Not found: {mirna_tumor_path}")
        sys.exit(1)

    mirna_tumor = pd.read_parquet(mirna_tumor_path)
    if "patient_id" in mirna_tumor.columns:
        mirna_tumor = mirna_tumor.set_index("patient_id")
    logger.info(f"miRNA tumor: {mirna_tumor.shape}")

    # ================================================================
    # STEP 3: Extract miRNA normal from raw file
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Extracting miRNA normal from raw file")
    logger.info("=" * 60)

    mirna_raw_path = raw_dir / "TCGA-LUAD.mirna.tsv"
    header_cols, separator = read_header_columns(str(mirna_raw_path))

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

    logger.info(f"Normal columns: {len(normal_cols)}")
    if len(normal_cols) == 0:
        logger.error("No normal miRNA samples!")
        sys.exit(1)

    feature_col = header_cols[0]
    mirna_normal_raw = pd.read_csv(
        mirna_raw_path, sep=separator,
        usecols=[feature_col] + normal_cols,
        encoding="utf-8-sig", low_memory=False,
    )
    mirna_normal = mirna_normal_raw.set_index(feature_col)[normal_cols].T
    mirna_normal.index.name = "sample_id"
    mirna_normal = mirna_normal.apply(pd.to_numeric, errors="coerce")
    logger.info(f"miRNA normal: {mirna_normal.shape}")

    # ================================================================
    # STEP 4: Align miRNA features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Aligning miRNA features")
    logger.info("=" * 60)

    tumor_col_map = {}
    for col in mirna_tumor.columns:
        name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        if "|" in name:
            name = name.split("|", 1)[0]
        tumor_col_map[name] = col

    common_features = sorted(set(mirna_normal.columns.astype(str)) & set(tumor_col_map.keys()))
    logger.info(f"Common miRNA features: {len(common_features)}")

    mirna_tumor_aligned = mirna_tumor[[tumor_col_map[f] for f in common_features]].copy()
    mirna_tumor_aligned.columns = common_features
    mirna_tumor_aligned["label"] = 1

    mirna_normal_aligned = mirna_normal[common_features].copy()
    mirna_normal_aligned["label"] = 0

    mirna_combined = pd.concat([mirna_tumor_aligned, mirna_normal_aligned], axis=0)
    logger.info(f"miRNA combined: {mirna_combined.shape}, labels: {mirna_combined['label'].value_counts().to_dict()}")

    # ================================================================
    # STEP 5: Match via Patient ID — keep BOTH tumor and normal per patient
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Matching via Patient ID (keeping both T/N)")
    logger.info("=" * 60)

    # Get patient IDs for miRNA combined
    mirna_pids = pd.Series([extract_patient_id(idx) for idx in mirna_combined.index], index=mirna_combined.index)

    # Find patients that appear in BOTH mRNA and miRNA
    mrna_pid_set = set(mrna_patient_ids.values)
    mirna_pid_set = set(mirna_pids.values)
    common_pids = mrna_pid_set & mirna_pid_set

    logger.info(f"mRNA patients: {len(mrna_pid_set)}, miRNA patients: {len(mirna_pid_set)}, Common: {len(common_pids)}")

    # Filter mRNA to common patients (keep ALL rows — both tumor and normal)
    mrna_mask = mrna_patient_ids.isin(common_pids)
    mrna_filtered = mrna_X[mrna_mask].copy()
    mrna_labels_filtered = mrna_labels[mrna_mask]
    mrna_pids_filtered = mrna_patient_ids[mrna_mask]

    # Filter miRNA to common patients
    mirna_mask = mirna_pids.isin(common_pids)
    mirna_filtered = mirna_combined[mirna_mask].copy()
    mirna_labels_filtered = mirna_filtered["label"]
    mirna_pids_filtered = mirna_pids[mirna_mask]
    mirna_features_filtered = mirna_filtered.drop(columns=["label"])

    n_tumor_final = int((mrna_labels_filtered == 1).sum())
    n_normal_final = int((mrna_labels_filtered == 0).sum())

    logger.info(f"After filtering to common patients:")
    logger.info(f"  mRNA: {mrna_filtered.shape}, Tumor={n_tumor_final}, Normal={n_normal_final}")
    logger.info(f"  miRNA: {mirna_features_filtered.shape}")

    if n_normal_final == 0:
        logger.error("CRITICAL: No normal samples after filtering!")
        sys.exit(1)

    # ================================================================
    # STEP 6: Create unified sample-level index
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Creating unified sample-level datasets")
    logger.info("=" * 60)

    # For mRNA: index is already sample-level barcodes (unique)
    # For miRNA: index may have duplicates (same patient, tumor+normal)
    # Solution: use sample barcode as index for miRNA too
    # Normal miRNA barcodes are like TCGA-XX-XXXX-11A (already unique)
    # Tumor miRNA index is patient-level (TCGA-XX-XXXX), need to make unique

    # Check if miRNA tumor index has patient-level IDs (no -XX suffix)
    mirna_tumor_indices = mirna_tumor_aligned.index.tolist()
    has_suffix = any("-" in str(idx).split("-")[-1] and len(str(idx).split("-")) >= 4 for idx in mirna_tumor_indices[:5])

    if not has_suffix:
        # Tumor miRNA has patient-level IDs, append "-01" to make them sample-level
        new_tumor_index = [f"{idx}-01T" for idx in mirna_tumor_aligned.index]
        mirna_tumor_aligned.index = new_tumor_index
        # Rebuild combined with new index
        mirna_normal_reindexed = mirna_normal_aligned.copy()
        # Normal already has sample barcodes like TCGA-XX-XXXX-11A
        mirna_combined_new = pd.concat([mirna_tumor_aligned, mirna_normal_reindexed], axis=0)

        # Re-filter
        mirna_pids_new = pd.Series([extract_patient_id(idx) for idx in mirna_combined_new.index], index=mirna_combined_new.index)
        mirna_mask_new = mirna_pids_new.isin(common_pids)
        mirna_filtered = mirna_combined_new[mirna_mask_new].copy()
        mirna_labels_filtered = mirna_filtered["label"]
        mirna_features_filtered = mirna_filtered.drop(columns=["label"])

    # Now both mRNA and miRNA have unique sample-level indices
    # But they have DIFFERENT indices (different sample barcodes)
    # We need to align by PATIENT ID and create a shared patient-level dataset
    # KEY INSIGHT: For each patient, pick ONE representative sample per modality
    # For tumor patients: pick tumor sample; for normal patients: pick normal sample

    # Determine which patients are "normal" vs "tumor" based on mRNA labels
    patient_label_map = {}
    for idx in mrna_filtered.index:
        pid = mrna_pids_filtered[idx]
        label = int(mrna_labels_filtered[idx])
        if pid not in patient_label_map or label == 0:
            # Prefer normal label if available (so we don't lose normals)
            patient_label_map[pid] = label

    # Select one mRNA sample per patient (prefer normal if patient has both)
    selected_mrna_rows = []
    selected_mrna_labels = []
    selected_pids = []
    seen_pids = set()

    # First pass: collect normal samples
    for idx in mrna_filtered.index:
        pid = mrna_pids_filtered[idx]
        label = int(mrna_labels_filtered[idx])
        if label == 0 and pid not in seen_pids:
            selected_mrna_rows.append(mrna_filtered.loc[idx])
            selected_mrna_labels.append(0)
            selected_pids.append(pid)
            seen_pids.add(pid)

    # Second pass: collect tumor samples for patients without normal
    for idx in mrna_filtered.index:
        pid = mrna_pids_filtered[idx]
        label = int(mrna_labels_filtered[idx])
        if label == 1 and pid not in seen_pids:
            selected_mrna_rows.append(mrna_filtered.loc[idx])
            selected_mrna_labels.append(1)
            selected_pids.append(pid)
            seen_pids.add(pid)

    mrna_final = pd.DataFrame(selected_mrna_rows, index=selected_pids)
    labels_final = pd.Series(selected_mrna_labels, index=selected_pids, name="label")

    # Select one miRNA sample per patient (matching the same tumor/normal choice)
    selected_mirna_rows = []
    mirna_seen = set()

    for pid in selected_pids:
        target_label = labels_final[pid]
        # Find matching miRNA sample
        candidates = mirna_filtered[mirna_pids_new[mirna_filtered.index] == pid] if 'mirna_pids_new' in dir() else mirna_filtered[[extract_patient_id(idx) == pid for idx in mirna_filtered.index]]

        matched = False
        for idx in candidates.index:
            cand_label = int(mirna_filtered.loc[idx, "label"]) if "label" in mirna_filtered.columns else int(candidates.loc[idx, "label"])
            if cand_label == target_label and idx not in mirna_seen:
                selected_mirna_rows.append(mirna_features_filtered.loc[idx] if idx in mirna_features_filtered.index else candidates.loc[idx].drop("label", errors="ignore"))
                mirna_seen.add(idx)
                matched = True
                break

        if not matched:
            # Fallback: take any available sample for this patient
            for idx in candidates.index:
                if idx not in mirna_seen:
                    row = mirna_features_filtered.loc[idx] if idx in mirna_features_filtered.index else candidates.loc[idx].drop("label", errors="ignore")
                    selected_mirna_rows.append(row)
                    mirna_seen.add(idx)
                    matched = True
                    break

        if not matched:
            # Patient has no miRNA data — fill with NaN
            selected_mirna_rows.append(pd.Series(np.nan, index=common_features))

    mirna_final = pd.DataFrame(selected_mirna_rows, index=selected_pids)

    n_tumor_out = int((labels_final == 1).sum())
    n_normal_out = int((labels_final == 0).sum())

    logger.info(f"Final dataset: {len(selected_pids)} patients")
    logger.info(f"  Tumor: {n_tumor_out}, Normal: {n_normal_out}")
    logger.info(f"  mRNA shape: {mrna_final.shape}")
    logger.info(f"  miRNA shape: {mirna_final.shape}")

    if n_normal_out == 0:
        logger.error("CRITICAL: Still no normal samples!")
        sys.exit(1)

    # ================================================================
    # STEP 7: Save
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Saving outputs")
    logger.info("=" * 60)

    mrna_out = mrna_final.copy()
    mrna_out["label"] = labels_final
    mrna_out.index.name = "sample_id"
    mrna_out.to_parquet(output_dir / "mrna.parquet")

    mirna_out = mirna_final.copy()
    mirna_out.index.name = "sample_id"
    mirna_out.to_parquet(output_dir / "mirna.parquet")

    labels_df = pd.DataFrame({"sample_id": selected_pids, "label": labels_final.values})
    labels_df.to_csv(output_dir / "labels.csv", index=False, encoding="utf-8-sig")

    metadata = {
        "n_patients": len(selected_pids),
        "n_tumor": n_tumor_out,
        "n_normal": n_normal_out,
        "mrna_features": mrna_final.shape[1],
        "mirna_features": mirna_final.shape[1],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("mRNA + miRNA Tumor/Normal Dataset Ready")
    print("=" * 60)
    print(f"Patients: {len(selected_pids)}")
    print(f"  Tumor:  {n_tumor_out}")
    print(f"  Normal: {n_normal_out}")
    print(f"mRNA features:  {metadata['mrna_features']}")
    print(f"miRNA features: {metadata['mirna_features']}")
    print(f"Output dir: {output_dir}")


if __name__ == "__main__":
    main()