"""External Validation on GSE131907 with memory-efficient chunked loading."""

from pathlib import Path
import argparse
import sys
import json
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.ml.baseline import fit_preprocessor, transform_preprocessor
from src.luad.models.factory import create_model
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="External Validation on GSE131907.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default="artifacts/experiments/pso_tumor_normal_v3")
    parser.add_argument("--experiment-name", type=str, default="external_validation_gse30219")
    parser.add_argument("--chunk-size", type=int, default=5000,
                        help="Number of rows to read per chunk for large files")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_gse_chunked(gse_path: Path, target_genes: list, logger, chunk_size: int = 5000):
    """
    Memory-efficient loading of large GSE matrix files.
    Reads the file in chunks and extracts only the target gene columns.
    """
    logger.info(f"Loading GSE data with chunked reading (chunk_size={chunk_size})...")
    logger.info(f"Target genes to extract: {len(target_genes)}")

    # First pass: read header to find column indices for target genes
    logger.info("Reading header to map gene positions...")
    header_df = pd.read_csv(gse_path, sep="\t", nrows=0, encoding="utf-8-sig")
    header_cols = list(header_df.columns)
    logger.info(f"Total columns in file: {len(header_cols)}")

    # Build mapping from gene name to column index
    # Handle potential prefixes like "ENSG00000..." or direct gene symbols
    col_to_idx = {}
    for i, col in enumerate(header_cols):
        col_clean = str(col).strip()
        # Try exact match first
        if col_clean in target_genes:
            col_to_idx[col_clean] = i
        # Try without version suffix (e.g., ENSG000001.2 -> ENSG000001)
        elif "." in col_clean:
            base_name = col_clean.split(".")[0]
            if base_name in target_genes:
                col_to_idx[base_name] = i

    matched_genes = [g for g in target_genes if g in col_to_idx.values() or 
                     any(g == k for k, v in col_to_idx.items())]
    
    # Rebuild proper mapping
    gene_col_indices = []
    final_matched_genes = []
    for gene in target_genes:
        if gene in col_to_idx:
            gene_col_indices.append(col_to_idx[gene])
            final_matched_genes.append(gene)
        else:
            # Try case-insensitive
            for col_name, idx in col_to_idx.items():
                if col_name.upper() == gene.upper():
                    gene_col_indices.append(idx)
                    final_matched_genes.append(gene)
                    break

    logger.info(f"Matched {len(final_matched_genes)}/{len(target_genes)} genes in GSE file")

    if len(final_matched_genes) < 10:
        logger.error(f"Too few genes matched ({len(final_matched_genes)}). Cannot proceed.")
        sys.exit(1)

    # Second pass: read data in chunks, extracting only needed columns
    # Always include column 0 (gene ID / row identifier)
    usecols = [0] + gene_col_indices
    
    logger.info(f"Reading {len(usecols)} columns in chunks...")
    
    chunks = []
    n_chunks = 0
    for chunk in pd.read_csv(
        gse_path, 
        sep="\t", 
        usecols=usecols,
        chunksize=chunk_size,
        encoding="utf-8-sig",
        low_memory=True
    ):
        n_chunks += 1
        chunks.append(chunk)
        if n_chunks % 10 == 0:
            logger.info(f"  Read {n_chunks} chunks ({len(chunks) * chunk_size} rows)...")

    logger.info(f"Concatenating {n_chunks} chunks...")
    gse_df = pd.concat(chunks, ignore_index=True)
    
    # Set gene names as column names (skip first column which is row IDs)
    first_col = gse_df.columns[0]
    gene_columns = gse_df.columns[1:]
    
    # Rename columns to matched gene names
    rename_dict = dict(zip(gene_columns, final_matched_genes))
    gse_df = gse_df.rename(columns=rename_dict)
    
    # Transpose so samples are rows and genes are columns
    # First column contains sample identifiers
    sample_ids = gse_df[first_col].values
    gene_data = gse_df[final_matched_genes].values.T  # genes x samples -> will transpose
    
    # Create final DataFrame: samples as rows, genes as columns
    result_df = pd.DataFrame(
        gene_data.T,  # samples x genes
        index=sample_ids,
        columns=final_matched_genes
    )
    
    # Convert to numeric
    result_df = result_df.apply(pd.to_numeric, errors="coerce")
    
    logger.info(f"Final GSE matrix shape: {result_df.shape}")
    logger.info(f"Samples: {result_df.shape[0]}, Genes: {result_df.shape[1]}")
    
    return result_df, final_matched_genes


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_path = resolve_path(args.experiment) if args.experiment else None

    try:
        config = load_config(config_path=config_path, experiment_path=experiment_path)
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    logger = get_logger(
        name="luad.external_val",
        log_file=project_root / "artifacts" / "logs" / "external_validation.log",
        level="INFO", console=True,
    )

    output_dir = ensure_dir(project_root / "artifacts" / "experiments" / args.experiment_name)

    # ================================================================
    # STEP 1: Load or Retrain Model
    # ================================================================
    logger.info("=" * 60)
    logger.info("STEP 1: Loading or Retraining Model")
    logger.info("=" * 60)

    model_dir = resolve_path(args.model_dir)
    model_path = model_dir / "xgboost_model.pkl"
    preproc_path = model_dir / "preprocessor.pkl"
    pso_genes_path = model_dir / "pso_selected_genes.csv"

    if not pso_genes_path.exists():
        logger.error(f"PSO genes file not found: {pso_genes_path}")
        sys.exit(1)

    pso_genes = pd.read_csv(pso_genes_path)["gene"].tolist()
    logger.info(f"Loaded {len(pso_genes)} PSO genes")

    if model_path.exists() and preproc_path.exists():
        logger.info(f"Loading saved model from {model_path}")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preproc_path, "rb") as f:
            preprocessor = pickle.load(f)
    else:
        logger.warning("Model files not found. Retraining model with PSO genes...")
        
        # Load TCGA training data
        tn_path = project_root / "data" / "processed" / "tumor_normal" / "tumor_normal_combined.parquet"
        if not tn_path.exists():
            logger.error(f"TCGA data not found: {tn_path}")
            sys.exit(1)

        tn_df = pd.read_parquet(tn_path)
        if "sample_id" in tn_df.columns:
            tn_df = tn_df.set_index("sample_id")

        label_map = {"NORMAL": 0, "TUMOR": 1}
        y_full = tn_df["label"].map(label_map).astype(int)
        X_full = tn_df.drop(columns=["label"])

        # Align to PSO genes
        stripped_map = {}
        for col in X_full.columns:
            name = str(col)
            if ":" in name: name = name.split(":", 1)[1]
            if "|" in name: name = name.split("|", 1)[0]
            stripped_map[name] = col

        matched_cols = [stripped_map[g] for g in pso_genes if g in stripped_map]
        matched_gene_names = [g for g in pso_genes if g in stripped_map]

        X_pso = X_full[matched_cols].copy()
        X_pso.columns = matched_gene_names

        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_pso, y_full, test_size=0.2, random_state=42, stratify=y_full
        )

        preprocessor = fit_preprocessor(
            X_train=X_train, y_train=y_train.values,
            max_missing_ratio=0.2,
            top_variance_features=len(matched_gene_names),
            feature_selection_method="variance",
            log1p=True,
        )
        X_train_p = transform_preprocessor(X_train, preprocessor)

        model = create_model(model_name="xgboost", random_state=42, num_classes=2)
        model.fit(X_train_p, y_train)

        # Internal validation
        X_test_p = transform_preprocessor(X_test, preprocessor)
        y_pred = model.predict(X_test_p)
        y_proba = model.predict_proba(X_test_p)[:, 1]
        from sklearn.metrics import balanced_accuracy_score, roc_auc_score
        ba = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        logger.info(f"Internal Validation: BA={ba:.4f}, AUC={auc:.4f}")

        # Save model and preprocessor
        ensure_dir(model_dir)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with open(preproc_path, "wb") as f:
            pickle.dump(preprocessor, f)
        logger.info(f"Model and preprocessor saved to {model_dir}")

    # ================================================================
    # STEP 2: Load GSE Data with Chunked Reading
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Loading GSE Data (Memory-Efficient)")
    logger.info("=" * 60)

    gse_path = project_root / "datasets" / "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt"
    if not gse_path.exists():
        logger.error(f"GSE file not found: {gse_path}")
        sys.exit(1)

    gse_df, matched_gse_genes = load_gse_chunked(
        gse_path=gse_path,
        target_genes=pso_genes,
        logger=logger,
        chunk_size=args.chunk_size
    )

    # Ensure same gene order as PSO genes used in training
    common_genes = [g for g in pso_genes if g in matched_gse_genes]
    logger.info(f"Common genes between model and GSE: {len(common_genes)}")

    if len(common_genes) < 10:
        logger.error(f"Too few common genes ({len(common_genes)}). Cannot proceed.")
        sys.exit(1)

    # Align both datasets to common genes
    X_gse = gse_df[common_genes].values

    # ================================================================
    # STEP 3: Predict
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Prediction on External Data")
    logger.info("=" * 60)

    # Apply same preprocessing
    X_gse_processed = transform_preprocessor(X_gse, preprocessor)

    y_prob = model.predict_proba(X_gse_processed)[:, 1]
    y_pred = model.predict(X_gse_processed)

    n_samples = len(y_prob)
    n_pred_tumor = int((y_pred == 1).sum())
    n_pred_normal = int((y_pred == 0).sum())
    mean_prob = float(np.mean(y_prob))
    std_prob = float(np.std(y_prob))
    median_prob = float(np.median(y_prob))
    min_prob = float(np.min(y_prob))
    max_prob = float(np.max(y_prob))

    # Confidence distribution
    high_conf = int((y_prob >= 0.9).sum())
    medium_conf = int(((y_prob >= 0.7) & (y_prob < 0.9)).sum())
    low_conf = int((y_prob < 0.7).sum())

    # ================================================================
    # STEP 4: Save Results
    # ================================================================
    results = {
        "dataset": "GSE131907",
        "platform": "Illumina HiSeq (log2TPM)",
        "n_samples": n_samples,
        "pso_genes_total": len(pso_genes),
        "pso_genes_matched": len(common_genes),
        "predictions": {
            "predicted_tumor": n_pred_tumor,
            "predicted_normal": n_pred_normal,
            "pct_tumor": round(n_pred_tumor / n_samples * 100, 1),
        },
        "probability_stats": {
            "mean": round(mean_prob, 4),
            "std": round(std_prob, 4),
            "median": round(median_prob, 4),
            "min": round(min_prob, 4),
            "max": round(max_prob, 4),
        },
        "confidence_distribution": {
            "high_ge_0.9": high_conf,
            "medium_0.7_0.9": medium_conf,
            "low_lt_0.7": low_conf,
        }
    }

    with (output_dir / "external_validation_results.json").open("w") as f:
        json.dump(results, f, indent=2)

    # Save predictions
    pred_df = pd.DataFrame({
        "sample_id": gse_df.index[:n_samples],
        "predicted_label": y_pred,
        "probability_tumor": y_prob,
    })
    pred_df.to_csv(output_dir / "predictions_gse131907.csv", index=False)

    # Print summary
    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION: GSE131907")
    print("=" * 70)
    print(f"Dataset:          GSE131907")
    print(f"Platform:         Illumina HiSeq (log2TPM)")
    print(f"Samples:          {n_samples}")
    print(f"PSO genes used:   {len(common_genes)}/{len(pso_genes)}")
    print("-" * 70)
    print(f"Predicted Tumor:  {n_pred_tumor}/{n_samples} ({n_pred_tumor/n_samples*100:.1f}%)")
    print(f"Predicted Normal: {n_pred_normal}/{n_samples} ({n_pred_normal/n_samples*100:.1f}%)")
    print("-" * 70)
    print(f"Mean Probability: {mean_prob:.4f} ± {std_prob:.4f}")
    print(f"Median Prob:      {median_prob:.4f}")
    print(f"Range:            [{min_prob:.4f}, {max_prob:.4f}]")
    print("-" * 70)
    print(f"High Conf (≥0.9):   {high_conf} ({high_conf/n_samples*100:.1f}%)")
    print(f"Medium (0.7-0.9):   {medium_conf} ({medium_conf/n_samples*100:.1f}%)")
    print(f"Low (<0.7):         {low_conf} ({low_conf/n_samples*100:.1f}%)")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()