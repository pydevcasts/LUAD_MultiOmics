from pathlib import Path
import sys

import yaml


def check_yaml_file(file_path: Path) -> bool:
    try:
        yaml.safe_load(file_path.read_text(encoding="utf-8"))
        print(f"[OK] {file_path}")
        return True
    except Exception as exc:
        print(f"[FAILED] {file_path} -> {exc}")
        return False


def main() -> None:
    files = [
        Path("configs/config.yaml"),
        Path("configs/datasets.yaml"),
        Path("configs/models.yaml"),
    ]

    all_ok = True

    for file_path in files:
        if not file_path.exists():
            print(f"[MISSING] {file_path}")
            all_ok = False
            continue

        if not check_yaml_file(file_path):
            all_ok = False

    if not all_ok:
        sys.exit(1)

    print("All YAML files are valid.")


if __name__ == "__main__":
    main()