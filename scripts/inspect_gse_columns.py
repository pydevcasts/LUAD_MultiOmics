"""Quick inspection of GSE30219 column names to identify gene ID format."""

from pathlib import Path
import pandas as pd
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    gse_path = PROJECT_ROOT / "datasets" / "GSE131907_Lung_Cancer_normalized_log2TPM_matrix.txt"
    
    if not gse_path.exists():
        print(f"File not found: {gse_path}")
        sys.exit(1)
    
    print(f"File size: {gse_path.stat().st_size / (1024*1024):.1f} MB")
    
    # Read only first 5 rows to inspect column names
    print("\nReading header and first 5 rows...")
    df = pd.read_csv(gse_path, sep="\t", nrows=5)
    
    print(f"\nTotal columns: {len(df.columns)}")
    print(f"First column (index): '{df.columns[0]}'")
    print(f"\nFirst 20 column names:")
    for i, col in enumerate(df.columns[:20]):
        print(f"  [{i:4d}] {col}")
    
    print(f"\nLast 10 column names:")
    for i, col in enumerate(df.columns[-10:]):
        idx = len(df.columns) - 10 + i
        print(f"  [{idx:4d}] {col}")
    
    # Check first few values of first column (gene IDs)
    print(f"\nFirst 10 gene IDs (first column values):")
    for i, val in enumerate(df.iloc[:10, 0]):
        print(f"  {val}")
    
    # Try to match with known PSO genes
    pso_genes_path = PROJECT_ROOT / "artifacts" / "experiments" / "pso_tumor_normal_v3" / "pso_selected_genes.csv"
    if pso_genes_path.exists():
        pso_genes = set(pd.read_csv(pso_genes_path)["gene"].tolist())
        cols_set = set(str(c) for c in df.columns)
        
        direct_match = pso_genes & cols_set
        print(f"\nDirect match with PSO genes: {len(direct_match)}/{len(pso_genes)}")
        
        if len(direct_match) > 0:
            print(f"Matched genes: {list(direct_match)[:10]}")
        
        # Try case-insensitive
        cols_upper = {str(c).upper() for c in df.columns}
        pso_upper = {g.upper() for g in pso_genes}
        upper_match = pso_upper & cols_upper
        print(f"Case-insensitive match: {len(upper_match)}/{len(pso_genes)}")
        
        # Check if columns look like Ensembl IDs
        ensembl_pattern = sum(1 for c in df.columns if str(c).startswith("ENSG"))
        print(f"\nColumns starting with 'ENSG': {ensembl_pattern}")
        
        # Check if columns look like Hugo Symbols
        hugo_pattern = sum(1 for c in df.columns if str(c).isupper() and len(str(c)) < 20 and not str(c).startswith("ENSG"))
        print(f"Columns looking like Hugo Symbols: {hugo_pattern}")

if __name__ == "__main__":
    main()