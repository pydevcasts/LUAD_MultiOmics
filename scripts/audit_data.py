"""Run data audit on all configured datasets.

This script only inspects data structure and produces audit reports.
It does not train models and does not finalize target or loaders.
"""

from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config
from src.luad.data.audit import run_audit
from src.luad.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LUAD Multi-Omics data audit."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to main config YAML file.",
    )

    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Optional path to datasets YAML file. Defaults to configs/datasets.yaml if exists.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = Path(args.config)

    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    datasets_path = None

    if args.datasets is not None:
        datasets_path = Path(args.datasets)

        if not datasets_path.is_absolute():
            datasets_path = PROJECT_ROOT / datasets_path

    try:
        config = load_config(
            config_path=config_path,
            datasets_path=datasets_path,
        )
    except ConfigError as exc:
        print(f"[FAILED] Config loading failed: {exc}")
        sys.exit(1)

    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    log_file = project_root / "artifacts" / "logs" / "audit.log"
    log_level = str(config.get("logging", {}).get("level", "INFO"))

    logger = get_logger(
        name="luad.audit",
        log_file=log_file,
        level=log_level,
        console=True,
    )

    logger.info("Starting data audit.")
    logger.info(f"Config path: {config_path}")
    logger.info(f"Project root: {project_root}")

    try:
        output_paths = run_audit(config=config, logger=logger)
    except Exception as exc:
        logger.error(f"Data audit failed: {exc}")
        raise

    logger.info("Data audit completed.")

    print("\nAudit outputs:")

    for key, value in output_paths.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()