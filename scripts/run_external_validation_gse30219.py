"""External validation using GSE30219 with probe-to-gene mapping (v2 - fixed GPL570 parser)."""

from pathlib import Path
import argparse
import sys
import json
import gzip
import re
from io import StringIO

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.ml.baseline import fit_preprocessor, transform_preprocessor
from src.luad.models.factory import create_model
from src.luad.evaluation.metrics import compute_classification_metrics
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="External validation GSE30219.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-experiment", type=str, default="pso_tumor_normal_v3")
    parser.add_argument("--gse-file", type=str, default=None)
    parser.add_argument("--annotation-file", type=str, default=None)
    parser.add_argument("--experiment-name", type=str, default="external_validation_gse30219")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_file(project_root, filename):
    candidates = [
        project_root / "datasets" / "external" / filename,
        project_root / "artifacts" / "external_validation" / filename,
        project_root / "datasets" / filename,
        project_root / "data" / "external" / filename,
    ]
    for c in candidates:
        if c.exists():
            return c
    for f in project_root.rglob(filename):
        return f
    return None


def load_series_matrix(filepath, logger):
    open_func = gzip.open if str(filepath).endswith(".gz") else open
    mode = "rt" if str(filepath).endswith(".gz") else "r"

    metadata = {"characteristics": [], "sample_ids": [], "titles": []}
    data_lines = []
    in_data = False

    logger.info(f"Reading series matrix: {filepath}")

    with open_func(str(filepath), mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if "!series_matrix_table_begin" in stripped:
                in_data = True
                continue
            if "!series_matrix_table_end" in stripped:
                break
            if in_data:
                data_lines.append(stripped)
            elif stripped.startswith("!Sample_geo_accession"):
                metadata["sample_ids"] = [x.strip('"') for x in stripped.split("\t")[1:]]
            elif stripped.startswith("!Sample_title"):
                metadata["titles"] = [x.strip('"') for x in stripped.split("\t")[1:]]
            elif stripped.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"') for x in stripped.split("\t")[1:]]
                metadata["characteristics"].append(vals)

    if not data_lines:
        raise ValueError("No expression data found in series matrix.")

    df = pd.read_csv(StringIO("\n".join(data_lines)), sep="\t", index_col=0)
    df.index = df.index.astype(str).str.strip('"')
    df.columns = [c.strip('"') for c in df.columns]

    logger.info(f"Expression matrix: {df.shape}")
    return df, metadata


def download_gpl570_annotation(output_dir, logger):
    url = "https://ftp.ncbi.nlm.nih.gov/geo/platforms/GPLnnn/GPL570/annot/GPL570.annot.gz"
    output_path = output_dir / "GPL570.annot.gz"

    if output_path.exists():
        logger.info(f"GPL570 annotation already exists: {output_path}")
        return output_path

    logger.info(f"Downloading GPL570 annotation from: {url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(output_path))
        logger.info(f"Downloaded: {output_path} ({output_path.stat().st_size / (1024*1024):.1f} MB)")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return None


def load_probe_to_gene_mapping(annotation_path, logger):
    """
    Load GPL570.annot.gz and build probe → gene symbol mapping.

    File format:
    - Lines starting with ^ or ! are metadata
    - Lines starting with # are column descriptions
    - First line after # lines is the real header (tab-separated)
    - Subsequent lines are data

    Columns of interest:
    - [0] ID: probe ID (e.g., 1007_s_at)
    - [2] Gene symbol: Hugo Symbol (e.g., DDR1)
    """
    logger.info(f"Loading GPL570 annotation: {annotation_path}")

    open_func = gzip.open if str(annotation_path).endswith(".gz") else open
    mode = "rt" if str(annotation_path).endswith(".gz") else "r"

    header_line = None
    header_cols = []
    probe_to_gene = {}
    skipped = 0
    total_data_lines = 0

    with open_func(str(annotation_path), mode, encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()

            if not stripped:
                continue

            # Skip metadata lines (^ and !)
            if stripped.startswith("^") or stripped.startswith("!"):
                continue

            # Skip comment/description lines (#)
            if stripped.startswith("#"):
                continue

            # First non-comment line is the real header
            if header_line is None:
                header_line = stripped
                header_cols = header_line.split("\t")
                logger.info(f"GPL570 header found: {len(header_cols)} columns")
                logger.info(f"Key columns: ID=[0], Gene symbol=[2]")
                continue

            # Data lines
            total_data_lines += 1
            parts = stripped.split("\t")

            if len(parts) < 3:
                skipped += 1
                continue

            probe_id = parts[0].strip().strip('"')
            gene_symbol = parts[2].strip().strip('"')

            # Handle multiple gene symbols separated by ///
            if "///" in gene_symbol:
                gene_symbol = gene_symbol.split("///")[0].strip()

            if gene_symbol and gene_symbol != "---" and gene_symbol != "":
                probe_to_gene[probe_id] = gene_symbol.upper()
            else:
                skipped += 1

    logger.info(f"Total data lines: {total_data_lines}")
    logger.info(f"Probes mapped to gene symbols: {len(probe_to_gene)}")
    logger.info(f"Skipped (no gene symbol): {skipped}")

    # Show some examples
    examples = list(probe_to_gene.items())[:5]
    logger.info(f"Mapping examples: {examples}")

    return probe_to_gene


def extract_labels(metadata, sample_ids, titles, logger):
    labels = {}
    histology = {}

    chars = metadata.get("characteristics", [])

    for char_row in chars:
        for i, val in enumerate(char_row):
            if i >= len(sample_ids):
                break
            sid = sample_ids[i]
            vl = val.lower().strip()

            if vl.startswith("tissue:"):
                tissue = vl.replace("tissue:", "").strip()
                if "tumour" in tissue or "tumor" in tissue:
                    labels[sid] = 1
                elif "normal" in tissue:
                    labels[sid] = 0

            if vl.startswith("histology:"):
                hist = vl.replace("histology:", "").strip()
                histology[sid] = hist

    for i, title in enumerate(titles):
        if i >= len(sample_ids):
            break
        sid = sample_ids[i]
        tl = title.lower()
        if "squamous" in tl or "sqcc" in tl or "scc" in tl:
            histology.setdefault(sid, "SQCC")
        elif "adk" in tl or "adenocarcinoma" in tl or "adc" in tl:
            histology.setdefault(sid, "ADC")

    n_tumor = sum(1 for v in labels.values() if v == 1)
    n_normal = sum(1 for v in labels.values() if v == 0)
    n_adc = sum(1 for v in histology.values() if v.upper() in ["ADC", "ADENOCARCINOMA"])
    n_sqcc = sum(1 for v in histology.values() if v.upper() in ["SQCC", "SQUAMOUS"])

    logger.info(f"Labels: Tumor={n_tumor}, Normal={n_normal}, Unlabeled={len(sample_ids)-len(labels)}")
    logger.info(f"Histology: ADC={n_adc}, SQCC={n_sqcc}, Other={len(histology)-n_adc-n_sqcc}")

    return labels, histology


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
        name="luad.ext_val_gse30219",
        log_file=project_root / "artifacts" / "logs" / "external_validation_gse30219.log",
        level="INFO", console=True,
    )

    output_dir = ensure_dir(project_root / "artifacts" / "experiments" / args.experiment_name)

    # ===== Step 1: Load GSE30219 =====
    logger.info("=" * 60)
    logger.info("STEP 1: Loading GSE30219")
    logger.info("=" * 60)

    gse_file = resolve_path(args.gse_file) if args.gse_file else find_file(project_root, "GSE30219_series_matrix.txt.gz")
    if gse_file is None or not gse_file.exists():
        logger.error("GSE30219 file not found!")
        sys.exit(1)

    expr_df, metadata = load_series_matrix(gse_file, logger)
    sample_ids = metadata.get("sample_ids", list(expr_df.columns))
    titles = metadata.get("titles", [])

    # ===== Step 2: Extract Labels =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Extracting Labels")
    logger.info("=" * 60)

    labels_dict, histology_dict = extract_labels(metadata, sample_ids, titles, logger)

    # ===== Step 3: Filter LUAD Only =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Filtering LUAD (Adenocarcinoma) Only")
    logger.info("=" * 60)

    luad_samples = []
    for sid in sample_ids:
        hist = histology_dict.get(sid, "").upper()
        if hist in ["ADC", "ADENOCARCINOMA", "ADK"]:
            luad_samples.append(sid)
        elif sid in labels_dict:
            luad_samples.append(sid)

    logger.info(f"LUAD samples selected: {len(luad_samples)}")

    expr_luad = expr_df[luad_samples].copy()
    logger.info(f"LUAD expression matrix: {expr_luad.shape}")

    # ===== Step 4: Probe-to-Gene Mapping =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Probe-to-Gene Symbol Mapping (GPL570)")
    logger.info("=" * 60)

    annot_file = resolve_path(args.annotation_file) if args.annotation_file else find_file(project_root, "GPL570.annot.gz")
    if annot_file is None or not annot_file.exists():
        annot_file = download_gpl570_annotation(output_dir, logger)

    if annot_file is None or not annot_file.exists():
        logger.error("GPL570 annotation not available!")
        sys.exit(1)

    probe_to_gene = load_probe_to_gene_mapping(annot_file, logger)

    # Map probes to gene symbols and aggregate
    gene_expr_rows = {}
    unmapped_probes = 0

    for probe_id in expr_luad.index:
        gene = probe_to_gene.get(probe_id, None)
        if gene:
            if gene not in gene_expr_rows:
                gene_expr_rows[gene] = []
            gene_expr_rows[gene].append(expr_luad.loc[probe_id].values)
        else:
            unmapped_probes += 1

    logger.info(f"Mapped probes: {sum(len(v) for v in gene_expr_rows.values())}")
    logger.info(f"Unique genes: {len(gene_expr_rows)}")
    logger.info(f"Unmapped probes: {unmapped_probes}")

    # Aggregate: mean across probes per gene
    gene_data = {}
    for gene, rows in gene_expr_rows.items():
        gene_data[gene] = np.mean(rows, axis=0)

    gene_expr_df = pd.DataFrame(gene_data, index=luad_samples)  # samples x genes
    logger.info(f"Gene-level expression matrix: {gene_expr_df.shape}")

    # ===== Step 5: Match PSO Genes =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: Matching PSO Genes")
    logger.info("=" * 60)

    pso_genes_path = project_root / "artifacts" / "experiments" / args.pso_experiment / "pso_selected_genes.csv"
    if not pso_genes_path.exists():
        logger.error(f"PSO genes not found: {pso_genes_path}")
        sys.exit(1)

    pso_genes = pd.read_csv(pso_genes_path)["gene"].tolist()
    logger.info(f"PSO genes: {len(pso_genes)}")

    gene_upper_map = {g.upper(): g for g in gene_expr_df.columns}
    matched_genes = []
    matched_original = []

    for gene in pso_genes:
        if gene.upper() in gene_upper_map:
            matched_genes.append(gene_upper_map[gene.upper()])
            matched_original.append(gene)

    logger.info(f"PSO genes matched: {len(matched_genes)}/{len(pso_genes)} ({100*len(matched_genes)/len(pso_genes):.1f}%)")

    if len(matched_genes) < 10:
        logger.error(f"Too few PSO genes matched ({len(matched_genes)}). Cannot proceed.")
        sys.exit(1)

    X_ext = gene_expr_df[matched_genes].copy()
    X_ext.columns = matched_original
    X_ext = X_ext.apply(pd.to_numeric, errors="coerce")

    # Assign labels
    y_ext_values = []
    valid_samples = []
    for sid in X_ext.index:
        if sid in labels_dict:
            y_ext_values.append(labels_dict[sid])
            valid_samples.append(sid)

    X_ext = X_ext.loc[valid_samples]
    y_ext = pd.Series(y_ext_values, index=valid_samples, name="label").astype(int)

    n_tumor_ext = int((y_ext == 1).sum())
    n_normal_ext = int((y_ext == 0).sum())
    logger.info(f"Final external dataset: {len(y_ext)} samples (Tumor={n_tumor_ext}, Normal={n_normal_ext})")

    # ===== Step 6: Train on TCGA & Predict =====
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: Training on TCGA & Predicting on External")
    logger.info("=" * 60)

    tn_path = project_root / "data" / "processed" / "tumor_normal" / "tumor_normal_combined.parquet"
    if not tn_path.exists():
        logger.error(f"TCGA data not found: {tn_path}")
        sys.exit(1)

    tcga_df = pd.read_parquet(tn_path)
    if "sample_id" in tcga_df.columns:
        tcga_df = tcga_df.set_index("sample_id")
    label_map = {"NORMAL": 0, "TUMOR": 1}
    y_tcga = tcga_df["label"].map(label_map).astype(int)
    X_tcga = tcga_df.drop(columns=["label"])

    stripped_map = {}
    for col in X_tcga.columns:
        name = str(col)
        if ":" in name: name = name.split(":", 1)[1]
        if "|" in name: name = name.split("|", 1)[0]
        stripped_map[name] = col

    tcga_matched = [g for g in matched_original if g in stripped_map]
    X_tcga_pso = X_tcga[[stripped_map[g] for g in tcga_matched]].copy()
    X_tcga_pso.columns = tcga_matched

    logger.info(f"TCGA PSO features: {X_tcga_pso.shape}")

    common_genes = sorted(set(X_tcga_pso.columns) & set(X_ext.columns))
    logger.info(f"Common genes (TCGA ∩ External): {len(common_genes)}")

    if len(common_genes) < 10:
        logger.error(f"Too few common genes ({len(common_genes)}).")
        sys.exit(1)

    X_tcga_aligned = X_tcga_pso[common_genes]
    X_ext_aligned = X_ext[common_genes]

    preproc = fit_preprocessor(
        X_tcga_aligned, y_tcga.values,
        max_missing_ratio=0.2,
        top_variance_features=len(common_genes),
        feature_selection_method="variance",
        log1p=True,
    )
    X_tcga_processed = transform_preprocessor(X_tcga_aligned, preproc)
    X_ext_processed = transform_preprocessor(X_ext_aligned, preproc)

    model = create_model("xgboost", random_state=42, num_classes=2)
    model.fit(X_tcga_processed, y_tcga)
    logger.info("Model trained on TCGA.")

    y_pred = model.predict(X_ext_processed)
    y_proba = model.predict_proba(X_ext_processed)[:, 1]

    if n_normal_ext > 0:
        metrics = compute_classification_metrics(
            y_true=y_ext.values, y_pred=y_pred, y_proba=y_proba, labels=[0, 1]
        )
    else:
        metrics = {"note": "No normal samples for full classification evaluation"}
        # Still report prediction distribution
        pred_tumor = int((y_pred == 1).sum())
        pred_normal = int((y_pred == 0).sum())
        metrics["predicted_tumor"] = pred_tumor
        metrics["predicted_normal"] = pred_normal
        metrics["mean_probability"] = round(float(np.mean(y_proba)), 4)

    # ===== Save Results =====
    results = {
        "dataset": "GSE30219",
        "platform": "GPL570 (Affymetrix HG-U133 Plus 2.0)",
        "total_samples": len(sample_ids),
        "luad_samples": len(luad_samples),
        "labeled_samples": len(y_ext),
        "n_tumor": n_tumor_ext,
        "n_normal": n_normal_ext,
        "pso_genes_total": len(pso_genes),
        "pso_genes_matched": len(matched_genes),
        "common_genes_used": len(common_genes),
        "metrics": metrics,
    }

    with (output_dir / "external_validation_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 70)
    print("EXTERNAL VALIDATION RESULTS: GSE30219")
    print("=" * 70)
    print(f"Dataset:          GSE30219 (GPL570)")
    print(f"Total samples:    {len(sample_ids)}")
    print(f"LUAD samples:     {len(luad_samples)}")
    print(f"Labeled:          {len(y_ext)} (Tumor={n_tumor_ext}, Normal={n_normal_ext})")
    print(f"PSO genes:        {len(matched_genes)}/{len(pso_genes)} matched")
    print(f"Common genes:     {len(common_genes)}")
    print("-" * 70)

    if n_normal_ext > 0:
        print(f"Accuracy:         {metrics.get('accuracy', 'N/A')}")
        print(f"Balanced Acc:     {metrics.get('balanced_accuracy', 'N/A')}")
        print(f"F1 Macro:         {metrics.get('f1_macro', 'N/A')}")
        print(f"ROC-AUC:          {metrics.get('roc_auc', 'N/A')}")
        print(f"MCC:              {metrics.get('mcc', 'N/A')}")
    else:
        print("⚠️  No normal samples for classification metrics.")
        print(f"   Predicted Tumor:  {metrics.get('predicted_tumor', 'N/A')}")
        print(f"   Predicted Normal: {metrics.get('predicted_normal', 'N/A')}")
        print(f"   Mean Probability: {metrics.get('mean_probability', 'N/A')}")

    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()