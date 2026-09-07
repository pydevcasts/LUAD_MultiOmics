"""SHAP analysis and biomarker discovery for PSO-selected genes.

Uses the XGBoost model trained on PSO-selected mRNA features
for tumor vs normal classification on TCGA-LUAD.
"""

from pathlib import Path
import argparse
import sys
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.ml.baseline import fit_preprocessor, transform_preprocessor
from src.luad.models.factory import create_model
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="SHAP Analysis & Biomarker Discovery.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-experiment", type=str, default="pso_tumor_normal_v3",
                        help="Name of PSO experiment to load selected genes from.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--top-biomarkers", type=int, default=30,
                        help="Number of top biomarkers to report.")
    parser.add_argument("--max-shap-samples", type=int, default=200,
                        help="Max samples for SHAP computation.")
    parser.add_argument("--experiment-name", type=str, default="shap_biomarker")
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def main():
    args = parse_args()
    config_path = resolve_path(args.config)
    experiment_path = None
    if args.experiment is not None:
        experiment_path = resolve_path(args.experiment)

    try:
        config = load_config(config_path=config_path, experiment_path=experiment_path)
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))

    logger = get_logger(
        name="luad.shap",
        log_file=project_root / "artifacts" / "logs" / "shap_analysis.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    # ===== Load PSO Selected Genes =====
    pso_dir = project_root / "artifacts" / "experiments" / args.pso_experiment
    pso_genes_path = pso_dir / "pso_selected_genes.csv"

    if not pso_genes_path.exists():
        logger.error(f"PSO selected genes not found: {pso_genes_path}")
        logger.error(f"Run PSO experiment '{args.pso_experiment}' first.")
        sys.exit(1)

    pso_genes_df = pd.read_csv(pso_genes_path)
    pso_gene_names = pso_genes_df["gene"].tolist()
    logger.info(f"Loaded {len(pso_gene_names)} PSO-selected genes from: {pso_genes_path}")

    # ===== Load Tumor vs Normal Data =====
    if args.data_dir is not None:
        data_dir = resolve_path(args.data_dir)
    else:
        data_dir = project_root / "data" / "processed" / "tumor_normal"

    combined_path = data_dir / "tumor_normal_combined.parquet"
    if not combined_path.exists():
        logger.error(f"Combined dataset not found: {combined_path}")
        sys.exit(1)

    logger.info(f"Loading tumor vs normal data from: {combined_path}")
    combined_df = pd.read_parquet(combined_path)

    if "sample_id" in combined_df.columns:
        combined_df = combined_df.set_index("sample_id")

    label_map = {"NORMAL": 0, "TUMOR": 1}
    y = combined_df["label"].map(label_map).astype(int)
    X = combined_df.drop(columns=["label"])

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    # ===== Align PSO Genes with Data Columns =====
    available_columns = list(X.columns)

    stripped_col_map = {}
    for col in available_columns:
        name = str(col)
        if ":" in name:
            name = name.split(":", 1)[1]
        if "|" in name:
            name = name.split("|", 1)[0]
        stripped_col_map[name] = col

    matched_cols = []
    matched_gene_names = []
    unmatched_genes = []

    for gene in pso_gene_names:
        if gene in stripped_col_map:
            matched_cols.append(stripped_col_map[gene])
            matched_gene_names.append(gene)
        else:
            unmatched_genes.append(gene)

    logger.info(f"PSO genes matched in data: {len(matched_cols)}/{len(pso_gene_names)}")
    if unmatched_genes:
        logger.warning(f"Unmatched genes ({len(unmatched_genes)}): {unmatched_genes[:10]}...")

    if len(matched_cols) == 0:
        logger.error("No PSO genes found in data columns!")
        sys.exit(1)

    X_pso = X[matched_cols].copy()
    X_pso.columns = matched_gene_names

    # ===== Train/Test Split =====
    X_train, X_test, y_train, y_test = train_test_split(
        X_pso, y, test_size=0.20, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # ===== Preprocessing (log1p + ANOVA already done by PSO, just scale) =====
    preproc = fit_preprocessor(
        X_train=X_train,
        y_train=y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=len(matched_cols),
        feature_selection_method="variance",
        log1p=True,
    )
    X_train_processed = transform_preprocessor(X_train, preproc)
    X_test_processed = transform_preprocessor(X_test, preproc)

    # ===== Train Final XGBoost Model =====
    logger.info("Training final XGBoost model for SHAP analysis...")
    model = create_model(model_name="xgboost", random_state=42, num_classes=2)
    model.fit(X_train_processed, y_train)

    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    test_ba = balanced_accuracy_score(y_test, y_pred)
    test_auc = roc_auc_score(y_test, y_proba[:, 1])
    logger.info(f"Final model: Test BA={test_ba:.4f}, Test AUC={test_auc:.4f}")

    # ===== SHAP Analysis =====
    logger.info("Computing SHAP values...")

    try:
        import shap
    except ImportError:
        logger.error("shap is not installed. Install with: pip install shap")
        sys.exit(1)

    n_shap_samples = min(args.max_shap_samples, len(X_test_processed))
    rng = np.random.RandomState(42)
    shap_indices = rng.choice(len(X_test_processed), size=n_shap_samples, replace=False)
    X_shap = X_test_processed[shap_indices]
    y_shap = y_test.iloc[shap_indices].values

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_values_class1 = shap_values[1]
    else:
        shap_values_class1 = shap_values

    logger.info(f"SHAP values computed for {n_shap_samples} samples.")

    # ===== Compute Mean Absolute SHAP per Feature =====
    mean_abs_shap = np.mean(np.abs(shap_values_class1), axis=0)
    shap_importance = pd.DataFrame({
        "gene": matched_gene_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_importance["rank"] = range(1, len(shap_importance) + 1)

    logger.info(f"\nTop {args.top_biomarkers} Biomarkers by SHAP:")
    for _, row in shap_importance.head(args.top_biomarkers).iterrows():
        logger.info(f"  #{int(row['rank']):3d} | {row['gene']:20s} | mean|SHAP|={row['mean_abs_shap']:.6f}")

    # ===== Direction Analysis =====
    mean_shap = np.mean(shap_values_class1, axis=0)
    shap_importance["mean_shap_signed"] = mean_shap
    shap_importance["direction"] = shap_importance["mean_shap_signed"].apply(
        lambda x: "UP_in_TUMOR" if x > 0 else "DOWN_in_TUMOR"
    )

    # ===== Pathway Mapping =====
    try:
        from src.luad.feature_selection.pathway_genes import PATHWAY_GENE_SETS
        gene_to_pathways = {}
        for pathway, genes in PATHWAY_GENE_SETS.items():
            for g in genes:
                gene_to_pathways.setdefault(g, []).append(pathway)

        shap_importance["pathways"] = shap_importance["gene"].apply(
            lambda g: "; ".join(gene_to_pathways.get(g, ["—"]))
        )
    except ImportError:
        shap_importance["pathways"] = "—"

    # ===== Save Outputs =====
    output_dir = project_root / "artifacts" / "experiments" / args.experiment_name
    output_dir = ensure_dir(output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    # Save biomarker table
    biomarker_path = output_dir / "biomarker_ranking.csv"
    shap_importance.to_csv(biomarker_path, index=False, encoding="utf-8-sig")
    logger.info(f"Biomarker ranking saved to: {biomarker_path}")

    # Save top biomarkers summary
    top_biomarkers = shap_importance.head(args.top_biomarkers)
    top_path = output_dir / f"top_{args.top_biomarkers}_biomarkers.csv"
    top_biomarkers.to_csv(top_path, index=False, encoding="utf-8-sig")

    # Save SHAP summary plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 10))
        shap.summary_plot(
            shap_values_class1,
            X_shap,
            feature_names=matched_gene_names,
            show=False,
            max_display=args.top_biomarkers,
            title="SHAP Summary: Tumor vs Normal (PSO-selected Genes)",
        )
        summary_plot_path = figures_dir / "shap_summary.png"
        plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP summary plot saved to: {summary_plot_path}")

        # Bar plot
        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            shap.Explanation(
                values=mean_abs_shap,
                base_values=np.zeros(len(mean_abs_shap)),
                feature_names=matched_gene_names,
            ),
            max_display=args.top_biomarkers,
            show=False,
            title="Mean |SHAP| Feature Importance",
        )
        bar_plot_path = figures_dir / "shap_bar_importance.png"
        plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP bar plot saved to: {bar_plot_path}")

        # Beeswarm plot
        plt.figure(figsize=(12, 10))
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values_class1,
                base_values=np.full(n_shap_samples, explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value),
                data=X_shap,
                feature_names=matched_gene_names,
            ),
            max_display=args.top_biomarkers,
            show=False,
        )
        beeswarm_path = figures_dir / "shap_beeswarm.png"
        plt.savefig(beeswarm_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"SHAP beeswarm plot saved to: {beeswarm_path}")

    except Exception as e:
        logger.warning(f"Could not generate SHAP plots: {e}")

    # Save full results JSON
    results = {
        "dataset": "TCGA-LUAD (mRNA + Normals)",
        "task": "Tumor vs Normal",
        "pso_experiment": args.pso_experiment,
        "n_pso_genes": len(pso_gene_names),
        "n_matched_genes": len(matched_cols),
        "model": "XGBoost",
        "test_balanced_accuracy": round(test_ba, 4),
        "test_roc_auc": round(test_auc, 4),
        "n_shap_samples": n_shap_samples,
        "top_biomarkers": top_biomarkers.to_dict(orient="records"),
    }

    results_path = output_dir / "shap_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    logger.info(f"Full results saved to: {results_path}")

    # ===== Print Summary =====
    print("\n" + "=" * 80)
    print("SHAP BIOMARKER DISCOVERY RESULTS")
    print("=" * 80)
    print(f"Dataset:          TCGA-LUAD (mRNA + Normals)")
    print(f"Task:             Tumor vs Normal")
    print(f"PSO genes:        {len(pso_gene_names)} → {len(matched_cols)} matched")
    print(f"Model:            XGBoost (Test BA={test_ba:.4f}, AUC={test_auc:.4f})")
    print(f"SHAP samples:     {n_shap_samples}")
    print("-" * 80)
    print(f"{'Rank':>4s} | {'Gene':20s} | {'Mean|SHAP|':>12s} | {'Direction':15s} | Pathways")
    print("-" * 80)

    for _, row in shap_importance.head(args.top_biomarkers).iterrows():
        print(
            f"{int(row['rank']):4d} | "
            f"{row['gene']:20s} | "
            f"{row['mean_abs_shap']:12.6f} | "
            f"{row['direction']:15s} | "
            f"{row['pathways']}"
        )

    print("=" * 80)
    print(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()