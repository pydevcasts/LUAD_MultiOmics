# Cohort Harmonization Report

- Clinical rows with stage: 512

## Target Distribution

| Stage | Count |
|---|---:|
| STAGE I | 278 |
| STAGE II | 124 |
| STAGE III | 83 |
| STAGE IV | 27 |

## Modality Patient Counts

| Modality Key | Patients | Status | Multi-sample Patients |
|---|---:|---|---:|
| mrna_rsem | 510 | ok | 0 |
| mirna | 513 | ok | 5 |
| methylation | 562 | ok | 0 |
| cna_raw | 511 | ok | 0 |
| cna_log2 | 511 | ok | 0 |
| rppa_raw | 360 | ok | 0 |
| rppa_zscores | 360 | ok | 0 |

## Intersection Counts

| Cohort Definition | Patients |
|---|---:|
| stage_only | 512 |
| mrna | 508 |
| mrna_mirna | 503 |
| mrna_mirna_methylation | 501 |
| mrna_mirna_methylation_cna_raw | 498 |
| mrna_mirna_methylation_cna_raw_rppa_zscores | 354 |
| mrna_mirna_methylation_cna_raw_mutations | 494 |

## Mutation Schema

- Status: ok
- Sample column: Tumor_Sample_Barcode
- Gene column: Hugo_Symbol
- Mutation patient count: 562
