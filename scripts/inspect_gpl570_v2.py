"""Inspect GPL570 annot file - find real header after # comment lines."""

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
    print(f"Size: {annot_path.stat().st_size / (1024*1024):.1f} MB\n")

    with gzip.open(str(annot_path), "rt", encoding="utf-8", errors="ignore") as f:
        line_num = 0
        comment_lines = []
        header_line = None
        data_lines = []

        for line in f:
            line_num += 1
            stripped = line.strip()

            if not stripped:
                continue

            # Skip ^ and ! metadata lines
            if stripped.startswith("^") or stripped.startswith("!"):
                continue

            # Collect # comment lines (these describe columns)
            if stripped.startswith("#"):
                comment_lines.append(stripped)
                continue

            # First non-comment, non-metadata line is the REAL header
            if header_line is None:
                header_line = stripped
                cols = header_line.split("\t")
                print(f"Comment/description lines: {len(comment_lines)}")
                print(f"\nReal HEADER at line {line_num}: {len(cols)} columns")
                print("-" * 60)
                for i, c in enumerate(cols):
                    print(f"  [{i:3d}] {c}")
                print("-" * 60)
                continue

            # Data lines
            data_lines.append(stripped)
            if len(data_lines) <= 3:
                parts = stripped.split("\t")
                print(f"\nDATA row {len(data_lines)} ({len(parts)} cols):")
                for i, p in enumerate(parts[:8]):
                    col_name = cols[i] if i < len(cols) else "?"
                    print(f"  {col_name}: {p[:80]}")

            if len(data_lines) >= 3:
                break

    print(f"\nTotal lines scanned: {line_num}")


if __name__ == "__main__":
    main()