"""Inspect Methylation and CNA raw files for normal samples."""

from pathlib import Path
import sys
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.luad.config.loader import load_config
from src.luad.data.harmonizer import read_header_columns


def main():
    config = load_config(config_path=PROJECT_ROOT / "configs" / "config.yaml")
    project_root = Path(config.get("runtime", {}).get("project_root", PROJECT_ROOT))
    raw_dir = Path(config.get("paths", {}).get("raw_dir", "datasets"))
    if not raw_dir.is_absolute():
        raw_dir = project_root / raw_dir

    files_to_check = {
        "Methylation": raw_dir / "data_methylation_hm27_hm450_merged.txt",
        "CNA": raw_dir / "data_cna.txt",
    }

    # TCGA sample type codes:
    # 01-09 = Tumor, 10-19 = Normal, 20-29 = Control, 40-49 = Cell line
    SAMPLE_TYPE_CODES = {
        "01": "Primary Solid Tumor",
        "02": "Recurrent Solid Tumor",
        "03": "Primary Blood Derived Cancer",
        "06": "Metastatic",
        "10": "Blood Derived Normal",
        "11": "Solid Tissue Normal",
        "12": "Buccal Cell Normal",
        "13": "EBV Immortalized Normal",
        "14": "Bone Marrow Normal",
        "20": "Control Analyte",
        "40": "Cell Line",
    }

    for name, filepath in files_to_check.items():
        print(f"\n{'='*70}")
        print(f"INSPECTING: {name}")
        print(f"File: {filepath}")
        print(f"{'='*70}")

        if not filepath.exists():
            print(f"  ❌ FILE NOT FOUND")
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"  Size: {size_mb:.1f} MB")

        header_cols, separator = read_header_columns(str(filepath))
        print(f"  Separator: '{separator}'")
        print(f"  Total columns: {len(header_cols)}")
        print(f"  First column (feature ID): '{header_cols[0]}'")

        # Classify ALL sample columns
        sample_cols = [c for c in header_cols[1:] if str(c).startswith("TCGA-")]
        non_tcga = [c for c in header_cols[1:] if not str(c).startswith("TCGA-")]

        print(f"\n  TCGA sample columns: {len(sample_cols)}")
        if non_tcga:
            print(f"  Non-TCGA columns: {len(non_tcga)} → {non_tcga[:5]}")

        # Parse sample type codes
        type_counts = {}
        normal_samples = []
        tumor_samples = []

        for col in sample_cols:
            col_str = str(col)
            parts = col_str.split("-")
            if len(parts) >= 4:
                sample_code = parts[3][:2]
                type_name = SAMPLE_TYPE_CODES.get(sample_code, f"Unknown({sample_code})")
                type_counts[type_name] = type_counts.get(type_name, 0) + 1

                try:
                    code_int = int(sample_code)
                    if 10 <= code_int <= 19:
                        normal_samples.append(col_str)
                    elif 1 <= code_int <= 9:
                        tumor_samples.append(col_str)
                except ValueError:
                    pass

        print(f"\n  Sample Type Breakdown:")
        for type_name, count in sorted(type_counts.items(), key=lambda x: -x[1]):
            marker = " ← NORMAL" if "Normal" in type_name else ""
            print(f"    {type_name}: {count}{marker}")

        print(f"\n  Tumor samples (code 01-09): {len(tumor_samples)}")
        print(f"  Normal samples (code 10-19): {len(normal_samples)}")

        if normal_samples:
            print(f"\n  ✅ Normal sample examples:")
            for ns in normal_samples[:5]:
                print(f"    {ns}")
        else:
            print(f"\n  ❌ NO NORMAL SAMPLES FOUND")
            print(f"  Showing first 10 sample columns for debugging:")
            for sc in sample_cols[:10]:
                parts = str(sc).split("-")
                code = parts[3][:2] if len(parts) >= 4 else "?"
                print(f"    {sc}  (code={code})")

    print(f"\n{'='*70}")
    print("INSPECTION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()