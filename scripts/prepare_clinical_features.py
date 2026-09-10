"""Prepare clinical features for Late Fusion with proper leakage prevention and encoding."""

import re
from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


# FIXED-v4: Nominal prefix map for clean column names
NOMINAL_PREFIX_MAP = {
    'American Joint Committee on Cancer Publication Version Type': 'AJCC_Version',
    'Person Neoplasm Cancer Status': 'Tumor_Status',
    'New Neoplasm Event Post Initial Therapy Indicator': 'New_Tumor_Event',
    'Genetic Ancestry Label': 'Ancestry',
    'Sex': 'Sex',
    'Race Category': 'Race',
    'Ethnicity Category': 'Ethnicity',
    'Radiation Therapy': 'Radiation',
    'Prior Diagnosis': 'Prior_Dx',
}

LEAKAGE_COLS = [
    "Last Communication Contact from Initial Pathologic Diagnosis Date",
    "Overall Survival (Months)", "Months of disease-specific survival",
    "Disease Free (Months)", "Progress Free Survival (Months)",
    "Last Alive Less Initial Pathologic Diagnosis Date Calculated Day Value",
    "Disease Free Status", "Progression Free Status",
    "Overall Survival Status", "Disease-specific Survival status",
    "Vital Status", "Follow-up Time", "Days to Last Follow Up", "Days to Death",
    "Days to Last Known Alive",
]

USELESS_COLS = [
    "Subtype", "TCGA PanCanAtlas Cancer Type Acronym", "Other Patient ID",
    "In PanCan Pathway Analysis", "Informed consent verified",
    "Form completion date", "ICD-10 Classification", "ICD-O-3 Histology", "ICD-O-3 Site",
]

NUMERIC_COLS = ["Diagnosis Age", "Birth from Initial Pathologic Diagnosis Date"]

ORDINAL_CONFIG = {
    'Neoplasm Disease Stage American Joint Committee on Cancer Code': {
        "STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
        "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2,
        "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3,
        "STAGE IV": 4, "STAGE IVA": 4, "STAGE IVB": 4,
        "STAGE X": 0, "STAGE TX": 0, "STAGE 0": 0,
    },
    'American Joint Committee on Cancer Tumor Stage Code': {
        "T1": 1, "T1A": 1, "T1B": 1, "T1C": 1,
        "T2": 2, "T2A": 2, "T2B": 2,
        "T3": 3, "T4": 4,
        "TX": 0, "TIS": 0, "T0": 0,
    },
    'Neoplasm Disease Lymph Node Stage American Joint Committee on Cancer Code': {
        "N0": 0, "N0(I+)": 0, "N0(I-)": 0,
        "N1": 1,
        "N2": 2, "N2A": 2, "N2B": 2,
        "N3": 3,
        "NX": 0,
    },
    'American Joint Committee on Cancer Metastasis Stage Code': {
        "M0": 0,
        "M1": 1, "M1A": 1, "M1B": 1,
        "MX": 0,
    },
}

NOMINAL_COLS = [
    'Sex', 'Race Category', 'Ethnicity Category', 'Radiation Therapy',
    'Prior Diagnosis', 'New Neoplasm Event Post Initial Therapy Indicator',
    'Person Neoplasm Cancer Status', 'Genetic Ancestry Label',
    'American Joint Committee on Cancer Publication Version Type',
]


def extract_patient_id(barcode: str) -> str:
    match = re.search(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})", str(barcode))
    return match.group(1).upper() if match else str(barcode).strip().upper()


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare clinical features.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--outlier-method", type=str, default="iqr",
                        choices=["iqr", "zscore", "none"])
    parser.add_argument("--outlier-threshold", type=float, default=3.0)
    parser.add_argument("--low-var-threshold", type=float, default=0.05)
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
        name="luad.prepare_clinical",
        log_file=project_root / "artifacts" / "logs" / "prepare_clinical.log",
        level="INFO", console=True,
    )

    output_dir = resolve_path(args.output_dir) if args.output_dir else project_root / "data" / "processed" / "clinical_tn"
    output_dir = ensure_dir(output_dir)

    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    # ================================================================
    # STEP 1: Loading Clinical Data
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Clinical Data")
    logger.info("=" * 60)

    clinical_path = raw_dir / "data_clinical_patient.txt"
    if not clinical_path.exists():
        logger.error(f"Clinical file not found: {clinical_path}")
        sys.exit(1)

    clin_raw = pd.read_csv(clinical_path, sep="\t", encoding="utf-8-sig")
    logger.info(f"Raw clinical data: {clin_raw.shape}")

    id_col = None
    for candidate in ["#Patient Identifier", "Patient Identifier", "patient_id", "PATIENT_ID"]:
        if candidate in clin_raw.columns:
            id_col = candidate
            break
    if id_col is None:
        id_col = clin_raw.columns[0]
        logger.warning(f"No standard ID column found, using first column: {id_col}")

    clin_raw[id_col] = clin_raw[id_col].apply(extract_patient_id)
    clin_raw = clin_raw.drop_duplicates(subset=[id_col])
    clin_raw = clin_raw.set_index(id_col)
    logger.info(f"Clinical data indexed by Patient ID: {clin_raw.shape}")

    leakage_found = [c for c in LEAKAGE_COLS if c in clin_raw.columns]
    if leakage_found:
        clin_raw = clin_raw.drop(columns=leakage_found)
        logger.info(f"Removed {len(leakage_found)} leakage columns")

    useless_found = [c for c in USELESS_COLS if c in clin_raw.columns]
    if useless_found:
        clin_raw = clin_raw.drop(columns=useless_found)
        logger.info(f"Removed {len(useless_found)} useless columns")

    # ================================================================
    # STEP 2: Aligning with Labels
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Aligning with Labels")
    logger.info("=" * 60)

    labels_path = project_root / "data" / "processed" / "mrna_mirna_tn" / "labels.csv"
    if not labels_path.exists():
        logger.error(f"Labels file not found: {labels_path}")
        sys.exit(1)

    labels_df = pd.read_csv(labels_path).set_index("sample_id")
    labels_df.index = labels_df.index.astype(str)

    common_patients = sorted(set(clin_raw.index) & set(labels_df.index))
    logger.info(f"Common patients: {len(common_patients)}")

    if len(common_patients) == 0:
        logger.error("No common patients found!")
        sys.exit(1)

    clin_aligned = clin_raw.loc[common_patients].copy()
    labels_aligned = labels_df.loc[common_patients]["label"].astype(int)
    logger.info(f"Label distribution: {labels_aligned.value_counts().to_dict()}")

    # ================================================================
    # STEP 3: Processing Numeric Features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Processing Numeric Features")
    logger.info("=" * 60)

    numeric_available = [c for c in NUMERIC_COLS if c in clin_aligned.columns]
    logger.info(f"Numeric features available: {numeric_available}")

    numeric_processed = pd.DataFrame(index=clin_aligned.index)
    final_numeric = []

    for col in numeric_available:
        series = pd.to_numeric(clin_aligned[col], errors="coerce")
        median_val = series.median()
        series = series.fillna(median_val)

        variance = series.var()
        if variance < args.low_var_threshold:
            logger.info(f"  Dropped numeric '{col}': variance={variance:.6f} < {args.low_var_threshold}")
            continue

        # Outlier handling for numeric
        if args.outlier_method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - args.outlier_threshold * IQR
            upper = Q3 + args.outlier_threshold * IQR
            series = series.clip(lower=lower, upper=upper)
        elif args.outlier_method == "zscore":
            mean_val = series.mean()
            std_val = series.std()
            if std_val > 0:
                lower = mean_val - args.outlier_threshold * std_val
                upper = mean_val + args.outlier_threshold * std_val
                series = series.clip(lower=lower, upper=upper)

        scaler = StandardScaler()
        scaled = scaler.fit_transform(series.values.reshape(-1, 1)).flatten()
        numeric_processed[col] = scaled
        final_numeric.append(col)
        logger.info(f"  Kept numeric '{col}': variance={variance:.6f}")

    logger.info(f"Final numeric features: {final_numeric}")

    # ================================================================
    # STEP 4: Processing Ordinal Features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Processing Ordinal Features")
    logger.info("=" * 60)

    ordinal_processed = pd.DataFrame(index=clin_aligned.index)
    final_ordinal = []

    for col, mapping in ORDINAL_CONFIG.items():
        if col not in clin_aligned.columns:
            logger.info(f"  Ordinal '{col}' not in dataset, skipping.")
            continue

        series = clin_aligned[col].astype(str).str.strip().str.upper()
        unique_vals = sorted(series.dropna().unique())
        logger.info(f"  {col} unique values ({len(unique_vals)}): {unique_vals}")

        mapped = series.map(mapping)
        unmapped_mask = mapped.isna() & series.notna()
        n_unmapped = unmapped_mask.sum()

        if n_unmapped > 0:
            median_val = mapped.median()
            mapped = mapped.fillna(median_val)
            logger.info(f"  {n_unmapped} unmapped values filled with median")

        variance = mapped.var()

        # FIXED-v4: Ordinal staging features are ALWAYS kept
        # They are clinically meaningful (AJCC staging system)
        # Low-var filter does NOT apply to ordinal features
        scaler = StandardScaler()
        scaled = scaler.fit_transform(mapped.values.reshape(-1, 1)).flatten()
        ordinal_processed[col] = scaled
        final_ordinal.append(col)
        logger.info(f"  Kept ordinal '{col}' (variance={variance:.4f}, clinically meaningful - no low-var filter)")

    logger.info(f"Final ordinal features: {final_ordinal}")

    # ================================================================
    # STEP 5: Processing Nominal Features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Processing Nominal Features")
    logger.info("=" * 60)

    nominal_available = [c for c in NOMINAL_COLS if c in clin_aligned.columns]
    logger.info(f"Nominal features available: {nominal_available}")

    nominal_processed = pd.DataFrame(index=clin_aligned.index)
    final_nominal = []

    for col in nominal_available:
        series = clin_aligned[col].astype(str).str.strip()
        series = series.replace({"nan": "Unknown", "": "Unknown"})
        series = series.fillna("Unknown")

        value_counts = series.value_counts()
        freq = value_counts / len(series)

        rare_cats = freq[freq < args.low_var_threshold].index.tolist()
        if rare_cats:
            logger.info(f"  {col}: merging {len(rare_cats)} rare categories into 'Other': {rare_cats}")
            series = series.replace(rare_cats, "Other")

        # FIXED-v4: Use clean prefix from NOMINAL_PREFIX_MAP
        prefix = NOMINAL_PREFIX_MAP.get(col, col.replace(" ", "_"))
        dummies = pd.get_dummies(series, prefix=prefix, drop_first=True)

        # Low-var filter applies to One-Hot encoded nominal features
        low_var_cols = [c for c in dummies.columns if dummies[c].var() < args.low_var_threshold]
        if low_var_cols:
            logger.info(f"  Removed low-var one-hot features: {low_var_cols}")
            dummies = dummies.drop(columns=low_var_cols)

        if dummies.shape[1] > 0:
            nominal_processed = pd.concat([nominal_processed, dummies], axis=1)
            final_nominal.extend(dummies.columns.tolist())
            logger.info(f"  Kept {dummies.shape[1]} one-hot features for '{col}'")
        else:
            logger.info(f"  All one-hot features removed for '{col}'")

    logger.info(f"Final nominal features ({len(final_nominal)}): {final_nominal}")

    # ================================================================
    # STEP 6: Combining All Features
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Combining All Features")
    logger.info("=" * 60)

    combined = pd.concat([numeric_processed, ordinal_processed, nominal_processed], axis=1)
    logger.info(f"Combined features shape: {combined.shape}")
    logger.info(f"  Numeric: {len(final_numeric)}")
    logger.info(f"  Ordinal: {len(final_ordinal)}")
    logger.info(f"  Nominal:  {len(final_nominal)}")
    logger.info(f"  Total:    {combined.shape[1]}")

    # ================================================================
    # STEP 7: Saving Outputs
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 7: Saving Outputs")
    logger.info("=" * 60)

    output_df = combined.copy()
    output_df["label"] = labels_aligned.values
    output_df.index.name = "sample_id"

    out_path = output_dir / "clinical_processed.parquet"
    output_df.to_parquet(out_path)
    logger.info(f"Saved: {out_path}")

    metadata = {
        "patients": len(common_patients),
        "tumor": int(labels_aligned.sum()),
        "normal": int(len(labels_aligned) - labels_aligned.sum()),
        "features_total": combined.shape[1],
        "features_numeric": len(final_numeric),
        "features_ordinal": len(final_ordinal),
        "features_nominal": len(final_nominal),
        "feature_names_numeric": final_numeric,
        "feature_names_ordinal": final_ordinal,
        "feature_names_nominal": final_nominal,
        "params": {
            "outlier_method": args.outlier_method,
            "outlier_threshold": args.outlier_threshold,
            "low_var_threshold": args.low_var_threshold,
        },
    }

    meta_path = output_dir / "clinical_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")

    print("\n" + "=" * 60)
    print("CLINICAL FEATURES PREPARATION COMPLETE (v4)")
    print("=" * 60)
    print(f"Patients: {len(common_patients)} (Tumor={metadata['tumor']}, Normal={metadata['normal']})")
    print(f"Total features: {metadata['features_total']}")
    print(f"  Numeric: {metadata['features_numeric']} → {final_numeric}")
    print(f"  Ordinal: {metadata['features_ordinal']} → {final_ordinal}")
    print(f"  Nominal: {metadata['features_nominal']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()