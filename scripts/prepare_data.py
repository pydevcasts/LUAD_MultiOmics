"""Run data preparation for LUAD Multi-Omics pipeline.

This script builds aligned interim datasets but does not perform preprocessing,
feature selection, or modeling.
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.preparation import run_data_preparation
from src.luad.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare aligned interim datasets for LUAD pipeline."
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
        help="Optional interim output directory.",
    )

    parser.add_argument(
        "--chunksize",
        type=int,
        default=None,
        help="Optional chunk size for matrix reading.",
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
        name="luad.prepare",
        log_file=project_root / "artifacts" / "logs" / "data_preparation.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    logger.info("Starting data preparation.")

    try:
        results = run_data_preparation(
            config=config,
            logger=logger,
            output_dir=args.output_dir,
            chunksize=args.chunksize,
        )
    except Exception as exc:
        logger.error(f"Data preparation failed: {exc}")
        raise

    print("\n=== Data Preparation Summary ===")

    for result in results:
        print(
            f"- key={result['key']} | "
            f"status={result['status']} | "
            f"patients={result.get('n_patients')} | "
            f"features={result.get('n_features')} | "
            f"output={result.get('output_matrix')}"
        )

        if result.get("errors"):
            for error in result["errors"]:
                print(f"  error: {error}")


if __name__ == "__main__":
    main()