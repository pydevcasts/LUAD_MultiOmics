"""Inspect GPL570 annotation file structure - find header and data."""

from pathlib import Path
import gzip
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    annot_path = None
    for f in PROJECT_ROOT.rglob("GPL570.annot.gz"):
        annot_path = f
        break

    if annot_path is None:
        print("GPL570.annot.gz not found!")
        sys.exit(1)

    print(f"File: {annot_path}")
    print(f"Size: {annot_path.stat().st_size / (1024*1024):.1f} MB")
    print()

    with gzip.open(str(annot_path), "rt", encoding="utf-8", errors="ignore") as f:
        line_num = 0
        meta_lines = 0
        header_line = None
        data_lines = []

        for line in f:
            line_num += 1
            stripped = line.strip()

            if not stripped:
                continue

            # Metadata lines start with ^ or !
            if stripped.startswith("^") or stripped.startswith("!"):
                meta_lines += 1
                if meta_lines <= 10:
                    print(f"META L{line_num}: {stripped[:150]}")
                continue

            # First non-metadata line is the header
            if header_line is None:
                header_line = stripped
                cols = header_line.split("\t")
                print(f"\n{'='*60}")
                print(f"HEADER found at line {line_num}")
                print(f"Number of columns: {len(cols)}")
                print(f"Column names:")
                for i, c in enumerate(cols):
                    print(f"  [{i:3d}] {c}")
                print(f"{'='*60}\n")
                continue

            # Data lines
            data_lines.append(stripped)
            if len(data_lines) <= 5:
                parts = stripped.split("\t")
                print(f"DATA L{line_num} ({len(parts)} cols): {parts[:5]}")

            if len(data_lines) >= 5:
                break

        print(f"\nTotal metadata lines: {meta_lines}")
        print(f"Header at line: {line_num - len(data_lines) - 1 if header_line else 'NOT FOUND'}")
        print(f"Data lines read: {len(data_lines)}")


if __name__ == "__main__":
    main()