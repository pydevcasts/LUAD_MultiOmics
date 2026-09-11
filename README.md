# LUAD MultiOmics

This project contains data preparation, feature selection, model training, and validation workflows for LUAD (lung adenocarcinoma) multi-omics analysis.

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
├── notebook/
├── scripts/
├── src/
├── requirements.txt
├── environment.yml
└── README.md
```

## Common commands

### Run a script

```bash
python scripts/train_baseline.py
python scripts/train_late_fusion.py
python scripts/train_survival.py
python scripts/validate_external_gse30219.py
```

### Prepare datasets

```bash
python scripts/prepare_data.py
python scripts/build_cohort.py
python scripts/make_split.py
python scripts/prepare_clinical_features.py
```

### Data inspection and validation

```bash
python scripts/check_config_loader.py
python scripts/check_yaml.py
python scripts/inspect_clinical_ids.py
python scripts/inspect_gpl570.py
```

### External validation and explainability

```bash
python scripts/run_external_validation_gse30219.py
python scripts/run_shap_analysis.py
python scripts/run_shap_biomarker_deep.py
```

## Python path

The project packages are under the `src/` directory. If needed, add the source directory to `PYTHONPATH`:

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# Windows PowerShell
$env:PYTHONPATH += ";$PWD\src"
```

## Useful checks

```bash
python -m compileall src
python -m pytest
```

## Notes

- Use the scripts under `scripts/` for dataset preparation and training workflows.
- Keep configuration files under `configs/` and update them according to cohort-specific metadata.
- Jupyter notebooks are stored under `notebook/` and are intended for exploratory and reporting work.
