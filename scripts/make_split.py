"""Create locked patient-level train/test split.

This script does not train models.
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.splitting import make_split
from src.luad.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create patient-level train/test split."
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
        "--cohort",
        type=str,
        default=None,
        help="Cohort name to split. Defaults to cohort.primary in config.",
    )

    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Optional output root for splits. Defaults to artifacts/splits.",
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
        name="luad.split",
        log_file=project_root / "artifacts" / "logs" / "split.log",
        level=str(config.get("logging", {}).get("level", "INFO")),
        console=True,
    )

    logger.info("Starting patient-level split.")

    try:
        summary = make_split(
            config=config,
            cohort_name=args.cohort,
            output_root=args.output_root,
            logger=logger,
        )
    except Exception as exc:
        logger.error(f"Split failed: {exc}")
        raise

    print("\n=== Split Summary ===")
    print(f"Cohort: {summary['cohort_name']}")
    print(f"Cohort size: {summary['cohort_size']}")
    print(f"Train size: {summary['train_size']}")
    print(f"Test size: {summary['test_size']}")
    print(f"Random state: {summary['random_state']}")

    print("\nTrain target distribution:")

    for stage, count in summary["train_target_distribution"].items():
        print(f"- {stage}: {count}")

    print("\nTest target distribution:")

    for stage, count in summary["test_target_distribution"].items():
        print(f"- {stage}: {count}")

    print("\nOutputs:")

    for key, value in summary["outputs"].items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()