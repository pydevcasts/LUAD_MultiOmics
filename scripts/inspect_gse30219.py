"""Quick inspection of GSE30219 series matrix."""

from pathlib import Path
import gzip
import sys
import json
import re

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def main():
    # Find the file
    possible_paths = [
        PROJECT_ROOT / "artifacts" / "external_validation" / "GSE30219_series_matrix.txt.gz",
        PROJECT_ROOT / "datasets" / "GSE30219_series_matrix.txt.gz",
        PROJECT_ROOT / "data" / "external" / "GSE30219_series_matrix.txt.gz",
    ]

    filepath = None
    for p in possible_paths:
        if p.exists():
            filepath = p
            break

    if filepath is None:
        # Search recursively
        for f in PROJECT_ROOT.rglob("GSE30219_series_matrix.txt.gz"):
            filepath = f
            break

    if filepath is None:
        print("ERROR: GSE30219_series_matrix.txt.gz not found!")
        print("Searched in:")
        for p in possible_paths:
            print(f"  {p}")
        sys.exit(1)

    print(f"Found: {filepath}")
    print(f"Size: {filepath.stat().st_size / (1024*1024):.1f} MB")

    # Read metadata lines
    metadata = {}
    data_lines = []
    in_data = False
    line_count = 0

    with gzip.open(str(filepath), "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line_count += 1
            stripped = line.strip()

            if "!series_matrix_table_begin" in stripped:
                in_data = True
                continue
            if "!series_matrix_table_end" in stripped:
                break
            if in_data:
                data_lines.append(stripped)
                if len(data_lines) >= 25:  # Read header + 20 data rows
                    break
            elif stripped.startswith("!Sample_geo_accession"):
                parts = [x.strip('"') for x in stripped.split("\t")[1:]]
                metadata["sample_ids"] = parts
                print(f"\nSamples: {len(parts)}")
                print(f"First 5 IDs: {parts[:5]}")
            elif stripped.startswith("!Sample_title"):
                parts = [x.strip('"') for x in stripped.split("\t")[1:]]
                metadata["titles"] = parts
                print(f"First 5 titles: {parts[:5]}")
            elif stripped.startswith("!Sample_characteristics"):
                vals = [x.strip('"') for x in stripped.split("\t")[1:]]
                metadata.setdefault("characteristics", []).append(vals)
                print(f"Characteristic: {vals[0][:80]}...")
            elif stripped.startswith("!Sample_platform_id"):
                metadata["platform"] = stripped.split("\t")[1].strip('"')
            elif stripped.startswith("!Series_title"):
                metadata["title"] = stripped.split("\t")[1].strip('"')

    print(f"\nSeries title: {metadata.get('title', 'N/A')}")
    print(f"Platform: {metadata.get('platform', 'N/A')}")
    print(f"Total lines read: {line_count}")
    print(f"Data rows preview: {len(data_lines)-1}")

    # Parse data preview
    if data_lines:
        from io import StringIO
        preview_text = "\n".join(data_lines[:21])  # header + 20 rows
        try:
            df_preview = pd.read_csv(StringIO(preview_text), sep="\t", index_col=0)
            df_preview.index = df_preview.index.astype(str).str.strip('"')
            df_preview.columns = [c.strip('"') for c in df_preview.columns]

            print(f"\nExpression matrix preview shape: {df_preview.shape}")
            print(f"Feature ID column name: '{df_preview.index.name}'")
            print(f"Feature ID examples: {list(df_preview.index[:10])}")
            print(f"Sample columns: {len(df_preview.columns)}")

            # Check if feature IDs are probe IDs or gene symbols
            first_ids = list(df_preview.index[:20])
            probe_pattern = re.compile(r"^\d+_at$|^\d+_s_at$|^\d+_a_at$|^AFFX-")
            probe_count = sum(1 for fid in first_ids if probe_pattern.match(fid))
            gene_pattern = re.compile(r"^[A-Z][A-Z0-9]+$|^[A-Z][A-Z0-9]+-[A-Z0-9]+$")
            gene_count = sum(1 for fid in first_ids if gene_pattern.match(fid))

            print(f"\nFeature ID type detection:")
            print(f"  Probe-like IDs (e.g., 1234_at): {probe_count}/20")
            print(f"  Gene symbol-like IDs: {gene_count}/20")

            if probe_count > gene_count:
                print("  → Likely PROBE IDs (need mapping to gene symbols)")
            else:
                print("  → Likely GENE SYMBOLS (direct matching possible)")

            # Check overlap with PSO genes
            pso_path = PROJECT_ROOT / "artifacts" / "experiments" / "pso_tumor_normal_v3" / "pso_selected_genes.csv"
            if pso_path.exists():
                pso_genes = set(pd.read_csv(pso_path)["gene"].tolist())
                direct_overlap = set(df_preview.index) & pso_genes
                upper_overlap = {x.upper() for x in df_preview.index} & {x.upper() for x in pso_genes}
                print(f"\nPSO gene overlap (preview 20 rows):")
                print(f"  Direct: {len(direct_overlap)}")
                print(f"  Case-insensitive: {len(upper_overlap)}")
        except Exception as e:
            print(f"Error parsing data preview: {e}")

    # Extract labels from characteristics
    print("\nLabel extraction from metadata:")
    sample_ids = metadata.get("sample_ids", [])
    chars = metadata.get("characteristics", [])
    tumor_count = 0
    normal_count = 0
    for char_row in chars:
        for val in char_row:
            vl = val.lower()
            if "tumor" in vl and "normal" not in vl:
                tumor_count += 1
            elif "normal" in vl:
                normal_count += 1
    print(f"  Tumor mentions: {tumor_count}")
    print(f"  Normal mentions: {normal_count}")

    # Save summary
    summary = {
        "file": str(filepath),
        "size_mb": round(filepath.stat().st_size / (1024*1024), 1),
        "samples": len(sample_ids),
        "platform": metadata.get("platform"),
        "title": metadata.get("title"),
        "characteristics_count": len(chars),
    }

    out_dir = PROJECT_ROOT / "artifacts" / "external_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "gse30219_inspection.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nInspection saved to: {out_dir / 'gse30219_inspection.json'}")


if __name__ == "__main__":
    main()