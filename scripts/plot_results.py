"""Create model-comparison and PSO-convergence plots from experiment artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot model metrics and PSO convergence from an experiment."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=Path("artifacts/experiments/pso_tumor_normal_v3"),
        help="Experiment artifact directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for PNG outputs; defaults to the experiment directory.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_results(experiment_dir: Path) -> tuple[pd.DataFrame, list[dict]]:
    results_path = experiment_dir / "pso_results.json"
    comparison_path = experiment_dir / "model_comparison.csv"

    if results_path.exists():
        with results_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
        model_results = pd.DataFrame(payload.get("model_results", []))
        pso_history = payload.get("pso_history", [])
    elif comparison_path.exists():
        model_results = pd.read_csv(comparison_path)
        pso_history = []
    else:
        raise FileNotFoundError(
            f"No pso_results.json or model_comparison.csv found in {experiment_dir}"
        )

    if model_results.empty:
        raise ValueError("The experiment contains no model results.")
    return model_results, pso_history


def plot_model_comparison(model_results: pd.DataFrame, output_path: Path, dpi: int) -> None:
    required_columns = {"model", "test_balanced_accuracy", "test_roc_auc"}
    missing = required_columns - set(model_results.columns)
    if missing:
        raise ValueError(f"Missing model metric columns: {sorted(missing)}")

    metrics = model_results.set_index("model")[[
        "test_balanced_accuracy",
        "test_roc_auc",
    ]]
    axes = metrics.rename(
        columns={
            "test_balanced_accuracy": "Balanced Accuracy",
            "test_roc_auc": "AUC",
        }
    ).plot(kind="bar", figsize=(10, 6), color=["#277da1", "#f9844a"], width=0.78)

    axes.set_title("Model Performance Comparison", fontsize=15, weight="bold")
    axes.set_xlabel("Model")
    axes.set_ylabel("Score")
    axes.set_ylim(0, 1.08)
    axes.grid(axis="y", linestyle="--", alpha=0.35)
    axes.legend(loc="lower right", frameon=False)
    axes.tick_params(axis="x", rotation=20)

    for container in axes.containers:
        axes.bar_label(container, fmt="%.3f", padding=3, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def plot_pso_convergence(
    pso_history: list[dict], output_path: Path, dpi: int
) -> None:
    if not pso_history:
        raise ValueError("The experiment contains no pso_history entries.")

    history = pd.DataFrame(pso_history).sort_values("iteration")
    required_columns = {"iteration", "best_fitness"}
    missing = required_columns - set(history.columns)
    if missing:
        raise ValueError(f"Missing PSO history columns: {sorted(missing)}")

    figure, axes = plt.subplots(figsize=(10, 6))
    axes.plot(
        history["iteration"],
        history["best_fitness"],
        color="#43aa8b",
        marker="o",
        markersize=3.5,
        linewidth=2,
        label="Best fitness",
    )
    axes.set_title("PSO Convergence", fontsize=15, weight="bold")
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Best fitness")
    axes.grid(True, linestyle="--", alpha=0.35)

    final_point = history.iloc[-1]
    annotation = f"Final fitness: {final_point['best_fitness']:.4f}"
    if "n_selected" in history.columns:
        annotation += f"\nSelected features: {int(final_point['n_selected'])}"
    axes.annotate(
        annotation,
        xy=(final_point["iteration"], final_point["best_fitness"]),
        xytext=(-120, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#555555"},
        bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": "#cccccc"},
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    experiment_dir = resolve_path(args.experiment_dir)
    output_dir = resolve_path(args.output_dir) if args.output_dir else experiment_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_results, pso_history = load_results(experiment_dir)
    model_output = output_dir / "model_performance_comparison.png"
    pso_output = output_dir / "pso_convergence.png"

    plot_model_comparison(model_results, model_output, args.dpi)
    plot_pso_convergence(pso_history, pso_output, args.dpi)

    print(f"Model comparison saved to: {model_output}")
    print(f"PSO convergence saved to: {pso_output}")


if __name__ == "__main__":
    main()