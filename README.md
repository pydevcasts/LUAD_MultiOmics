# LUAD MultiOmics

This project contains the LUAD (lung adenocarcinoma) data-preparation, feature-selection, model-training, and external-validation pipelines.

## Environment setup

### Option 1: Conda

```bash
conda env create -f environment.yml
conda activate luad_multiomics
```

### Option 2: Pip

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project structure

```bash
.
├── configs/
├── docs/
├── scripts/
├── src/
├── requirements.txt
├── environment.yml
├── README.md
└── .gitignore
```

## Core commands

### Data preparation

```bash
python scripts/prepare_data.py
python scripts/build_cohort.py
python scripts/make_split.py
python scripts/prepare_clinical_features.py
python scripts/prepare_mrna_mirna_tn.py
python scripts/prepare_tumor_normal.py
python scripts/prepare_survival.py
python scripts/prepare_4modality_tn.py
```

### Model training and validation

```bash
python scripts/train_baseline.py
python scripts/train_late_fusion.py
python scripts/train_late_fusion_clinical.py
python scripts/train_late_fusion_pso.py
python scripts/train_pso_feature_selection.py
python scripts/train_pathway_survival.py
python scripts/train_survival.py
python scripts/train_tumor_normal.py
python scripts/run_external_validation_gse30219.py
python scripts/validate_external_gse30219.py
python scripts/run_shap_analysis.py
python scripts/run_shap_biomarker_deep.py
```

### Python path

The project packages are under the `src/` directory. If needed, add the source directory to `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# Windows PowerShell
$env:PYTHONPATH += ";$PWD\src"
```

### Useful checks

```bash
python -m compileall src
python -m pytest
```

## Notes

- The scripts under `scripts/` provide the operational workflow for preparing data and running models.
- Exploratory inspection and diagnostics scripts were removed because they do not contribute to the final pipeline or results.
- Final model execution should use the training and validation scripts listed above.
