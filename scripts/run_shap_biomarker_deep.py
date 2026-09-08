"""Deep SHAP analysis on mRNA PSO genes with pathway mapping."""

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
from src.luad.feature_selection.pathway_genes import PATHWAY_GENE_SETS, get_gene_to_pathways
from src.luad.utils.logger import get_logger
from src.luad.utils.io import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Deep SHAP biomarker analysis.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--experiment", type=str, default=None)
    parser.add_argument("--pso-experiment", type=str, default="pso_tumor_normal_v3")
    parser.add_argument("--top-biomarkers", type=int, default=30)
    parser.add_argument("--max-shap-samples", type=int, default=200)
    parser.add_argument("--experiment-name", type=str, default="shap_biomarker_deep")
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
        name="luad.shap_deep",
        log_file=project_root / "artifacts" / "logs" / "shap_biomarker_deep.log",
        level="INFO", console=True,
    )

    output_dir = ensure_dir(project_root / "artifacts" / "experiments" / args.experiment_name)
    figures_dir = ensure_dir(output_dir / "figures")

    # ===== Load Data =====
    logger.info("=" * 60)
    logger.info("Loading tumor/normal data and PSO genes")
    logger.info("=" * 60)

    tn_path = project_root / "data" / "processed" / "tumor_normal" / "tumor_normal_combined.parquet"
    if not tn_path.exists():
        logger.error(f"Tumor/normal data not found: {tn_path}")
        sys.exit(1)

    combined_df = pd.read_parquet(tn_path)
    if "sample_id" in combined_df.columns:
        combined_df = combined_df.set_index("sample_id")

    label_map = {"NORMAL": 0, "TUMOR": 1}
    y_all = combined_df["label"].map(label_map).astype(int)
    X_all = combined_df.drop(columns=["label"])

    logger.info(f"Dataset: {X_all.shape}, Labels: {y_all.value_counts().to_dict()}")

    # Load PSO genes
    pso_genes_path = project_root / "artifacts" / "experiments" / args.pso_experiment / "pso_selected_genes.csv"
    if not pso_genes_path.exists():
        logger.error(f"PSO genes not found: {pso_genes_path}")
        sys.exit(1)

    pso_genes = pd.read_csv(pso_genes_path)["gene"].tolist()
    logger.info(f"PSO genes: {len(pso_genes)}")

    # Align features
    stripped_map = {}
    for col in X_all.columns:
        name = str(col)
        if ":" in name: name = name.split(":", 1)[1]
        if "|" in name: name = name.split("|", 1)[0]
        stripped_map[name] = col

    matched_cols = [stripped_map[g] for g in pso_genes if g in stripped_map]
    matched_names = [g for g in pso_genes if g in stripped_map]

    X_pso = X_all[matched_cols].copy()
    X_pso.columns = matched_names
    logger.info(f"Aligned PSO features: {X_pso.shape}")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_pso, y_all, test_size=0.20, random_state=42, stratify=y_all
    )

    # Preprocess
    preproc = fit_preprocessor(
        X_train=X_train, y_train=y_train.values,
        max_missing_ratio=0.2,
        top_variance_features=len(matched_names),
        feature_selection_method="variance",
        log1p=True,
    )
    X_train_p = transform_preprocessor(X_train, preproc)
    X_test_p = transform_preprocessor(X_test, preproc)

    # Train XGBoost
    logger.info("Training XGBoost model...")
    model = create_model("xgboost", random_state=42, num_classes=2)
    model.fit(X_train_p, y_train)

    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    y_pred = model.predict(X_test_p)
    y_proba = model.predict_proba(X_test_p)[:, 1]
    ba = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    logger.info(f"Model performance: BA={ba:.4f}, AUC={auc:.4f}")

    # ===== SHAP Analysis =====
    logger.info("\n" + "=" * 60)
    logger.info("Computing SHAP values")
    logger.info("=" * 60)

    try:
        import shap
    except ImportError:
        logger.error("shap not installed. Run: pip install shap")
        sys.exit(1)

    n_shap = min(args.max_shap_samples, len(X_test_p))
    rng = np.random.RandomState(42)
    shap_idx = rng.choice(len(X_test_p), size=n_shap, replace=False)
    X_shap = X_test_p[shap_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_shap)

    if isinstance(shap_values, list):
        shap_class1 = shap_values[1]
    else:
        shap_class1 = shap_values

    logger.info(f"SHAP computed for {n_shap} samples")

    # ===== Biomarker Ranking =====
    logger.info("\n" + "=" * 60)
    logger.info("Biomarker Ranking")
    logger.info("=" * 60)

    mean_abs_shap = np.mean(np.abs(shap_class1), axis=0)
    mean_signed_shap = np.mean(shap_class1, axis=0)

    biomarker_df = pd.DataFrame({
        "gene": matched_names,
        "mean_abs_shap": mean_abs_shap,
        "mean_signed_shap": mean_signed_shap,
        "direction": ["UP_in_TUMOR" if s > 0 else "DOWN_in_TUMOR" for s in mean_signed_shap],
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    biomarker_df["rank"] = range(1, len(biomarker_df) + 1)

    # ===== Pathway Mapping =====
    gene_to_pathways = get_gene_to_pathways()

    biomarker_df["pathways"] = biomarker_df["gene"].apply(
        lambda g: "; ".join(gene_to_pathways.get(g, [])) if g in gene_to_pathways else ""
    )

    biomarker_df["in_pathway"] = biomarker_df["pathways"].apply(lambda x: x != "")

    # Print top biomarkers
    logger.info(f"\nTop {args.top_biomarkers} Biomarkers:")
    logger.info(f"{'Rank':>4s} | {'Gene':15s} | {'Mean|SHAP|':>10s} | {'Direction':15s} | Pathways")
    logger.info("-" * 80)

    for _, row in biomarker_df.head(args.top_biomarkers).iterrows():
        pw = row["pathways"] if row["pathways"] else "—"
        logger.info(
            f"{int(row['rank']):4d} | {row['gene']:15s} | "
            f"{row['mean_abs_shap']:10.6f} | {row['direction']:15s} | {pw}"
        )

    # Pathway coverage statistics
    top_n = biomarker_df.head(args.top_biomarkers)
    n_in_pathway = int(top_n["in_pathway"].sum())
    logger.info(f"\nPathway coverage in top-{args.top_biomarkers}: {n_in_pathway}/{args.top_biomarkers} ({100*n_in_pathway/args.top_biomarkers:.1f}%)")

    # Per-pathway breakdown
    pathway_counts = {}
    for _, row in top_n.iterrows():
        if row["pathways"]:
            for pw in row["pathways"].split("; "):
                pathway_counts[pw] = pathway_counts.get(pw, 0) + 1

    if pathway_counts:
        logger.info("\nPathway breakdown in top biomarkers:")
        for pw, count in sorted(pathway_counts.items(), key=lambda x: -x[1]):
            total_in_pw = len(PATHWAY_GENE_SETS.get(pw, []))
            matched_in_pw = sum(1 for g in PATHWAY_GENE_SETS.get(pw, []) if g in set(matched_names))
            logger.info(f"  {pw}: {count} genes in top-{args.top_biomarkers} ({matched_in_pw} total matched)")

    # ===== Generate Plots =====
    logger.info("\nGenerating SHAP plots...")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. Summary plot
        plt.figure(figsize=(14, 10))
        shap.summary_plot(
            shap_class1, X_shap,
            feature_names=matched_names,
            show=False,
            max_display=args.top_biomarkers,
            title="SHAP Summary: Tumor vs Normal (324 PSO Genes)",
        )
        plt.savefig(figures_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: shap_summary.png")

        # 2. Bar importance plot
        plt.figure(figsize=(12, 8))
        shap.plots.bar(
            shap.Explanation(
                values=mean_abs_shap,
                base_values=np.zeros(len(mean_abs_shap)),
                feature_names=matched_names,
            ),
            max_display=args.top_biomarkers,
            show=False,
        )
        plt.title("Mean |SHAP| Feature Importance (Top Biomarkers)")
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_bar_importance.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: shap_bar_importance.png")

        # 3. Beeswarm plot
        plt.figure(figsize=(14, 10))
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)):
            expected_val = expected_val[1]
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_class1,
                base_values=np.full(n_shap, expected_val),
                data=X_shap,
                feature_names=matched_names,
            ),
            max_display=args.top_biomarkers,
            show=False,
        )
        plt.title("SHAP Beeswarm: Feature Impact Distribution")
        plt.tight_layout()
        plt.savefig(figures_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: shap_beeswarm.png")

        # 4. Pathway enrichment bar chart
        if pathway_counts:
            plt.figure(figsize=(10, 6))
            pws = sorted(pathway_counts.keys(), key=lambda x: -pathway_counts[x])
            counts = [pathway_counts[pw] for pw in pws]
            colors = ["#E74C3C" if c >= 3 else "#3498DB" if c >= 2 else "#95A5A6" for c in counts]
            plt.barh(pws[::-1], counts[::-1], color=colors[::-1])
            plt.xlabel("Number of Top Biomarkers")
            plt.title(f"Pathway Enrichment in Top-{args.top_biomarkers} Biomarkers")
            plt.tight_layout()
            plt.savefig(figures_dir / "pathway_enrichment.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"  Saved: pathway_enrichment.png")

    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    # ===== Save Results =====
    # Full biomarker table
    biomarker_df.to_csv(output_dir / "biomarker_ranking_full.csv", index=False, encoding="utf-8-sig")

    # Top biomarkers
    top_df = biomarker_df.head(args.top_biomarkers)
    top_df.to_csv(output_dir / f"top_{args.top_biomarkers}_biomarkers.csv", index=False, encoding="utf-8-sig")

    # JSON results
    results = {
        "task": "Tumor vs Normal",
        "model": "XGBoost",
        "pso_genes_total": len(pso_genes),
        "pso_genes_matched": len(matched_names),
        "model_performance": {"balanced_accuracy": round(ba, 4), "roc_auc": round(auc, 4)},
        "shap_samples": n_shap,
        "top_biomarkers": top_df.to_dict(orient="records"),
        "pathway_coverage": {
            "n_in_pathway": n_in_pathway,
            "total_top": args.top_biomarkers,
            "percentage": round(100 * n_in_pathway / args.top_biomarkers, 1),
        },
        "pathway_breakdown": pathway_counts,
    }

    with (output_dir / "shap_deep_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # ===== Final Summary =====
    print("\n" + "=" * 80)
    print("SHAP BIOMARKER DISCOVERY RESULTS (Deep Analysis)")
    print("=" * 80)
    print(f"Model:            XGBoost (BA={ba:.4f}, AUC={auc:.4f})")
    print(f"PSO genes:        {len(matched_names)}")
    print(f"SHAP samples:     {n_shap}")
    print(f"Pathway coverage: {n_in_pathway}/{args.top_biomarkers} ({100*n_in_pathway/args.top_biomarkers:.1f}%)")
    print("-" * 80)
    print(f"{'Rank':>4s} | {'Gene':15s} | {'Mean|SHAP|':>10s} | {'Direction':15s} | Pathways")
    print("-" * 80)

    for _, row in top_df.iterrows():
        pw = row["pathways"] if row["pathways"] else "—"
        print(
            f"{int(row['rank']):4d} | {row['gene']:15s} | "
            f"{row['mean_abs_shap']:10.6f} | {row['direction']:15s} | {pw}"
        )

    print("=" * 80)

    if pathway_counts:
        print("\nPathway Enrichment:")
        for pw, count in sorted(pathway_counts.items(), key=lambda x: -x[1]):
            print(f"  {pw}: {count} biomarkers")

    print(f"\nOutputs saved to: {output_dir}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == "__main__":
    main()