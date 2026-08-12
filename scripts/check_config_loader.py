"""Quick check script for config loader."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import ConfigError, load_config


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "config.yaml"

    try:
        config = load_config(config_path=config_path)
    except ConfigError as exc:
        print(f"[FAILED] {exc}")
        sys.exit(1)

    print("[OK] Config loaded successfully.")
    print(f"Project root: {config['runtime']['project_root']}")
    print(f"Raw dir config value: {config['paths']['raw_dir']}")

    dataset_files = config.get("datasets_files", [])
    print(f"Dataset entries found: {len(dataset_files)}")

    if not dataset_files:
        print("[WARNING] No dataset entries were found in datasets.yaml.")

    for item in dataset_files:
        resolved_path = item.get("resolved_path", "")
        exists = Path(resolved_path).exists() if resolved_path else False

        print(
            "- "
            f"key={item.get('key')}, "
            f"modality={item.get('modality')}, "
            f"exists={exists}, "
            f"path={resolved_path}"
        )


if __name__ == "__main__":
    main()