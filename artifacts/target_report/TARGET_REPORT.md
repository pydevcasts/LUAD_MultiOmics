# Target Confirmation Report

- Clinical file: `D:\LUAD_MultiOmics\LUAD_MultiOmics\datasets\data_clinical_patient.txt`
- Rows: 566
- Columns: 38
- Target column: `AJCC_PATHOLOGIC_TUMOR_STAGE`
- Target found: True

## Raw Target Summary

- Missing count: 54
- Missing ratio: 0.09540636042402827
- Unique count: 9

### Raw target counts

| Class | Count |
|---|---:|
| STAGE IB | 139 |
| STAGE IA | 134 |
| STAGE IIIA | 72 |
| STAGE IIB | 71 |
| STAGE IIA | 52 |
| STAGE IV | 27 |
| STAGE IIIB | 11 |
| STAGE I | 5 |
| STAGE II | 1 |

## Mapped/Grouped Target Summary

- Missing count: 54
- Missing ratio: 0.09540636042402827
- Unique count: 4

### Grouped target counts

| Class | Count |
|---|---:|
| STAGE I | 278 |
| STAGE II | 124 |
| STAGE III | 83 |
| STAGE IV | 27 |

## Patient ID Candidates

No TCGA patient ID candidates detected.

## Leakage Columns Present in Clinical File

- PATH_T_STAGE
- PATH_N_STAGE
- PATH_M_STAGE
- AJCC_STAGING_EDITION
- OS_STATUS
- OS_MONTHS
- DFS_STATUS
- DFS_MONTHS
- DAYS_LAST_FOLLOWUP
- NEW_TUMOR_EVENT_AFTER_INITIAL_TREATMENT
- RADIATION_THERAPY
- SUBTYPE
- DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS

## Constant or Near-Constant Columns

- SUBTYPE | unique_count=1 | top_values=LUAD: 502
- CANCER_TYPE_ACRONYM | unique_count=1 | top_values=LUAD: 566
- DAYS_TO_INITIAL_PATHOLOGIC_DIAGNOSIS | unique_count=1 | top_values=0.0: 495
- INFORMED_CONSENT_VERIFIED | unique_count=1 | top_values=Yes: 514
- PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT | unique_count=0 | top_values=
- WEIGHT | unique_count=0 | top_values=

## Recommendation

The recommended primary target is AJCC_PATHOLOGIC_TUMOR_STAGE grouped into STAGE I/II/III/IV. Class counts are acceptable for multiclass staging with balanced metrics. For stage prediction, PATH_T_STAGE, PATH_N_STAGE, PATH_M_STAGE, OS/DFS fields, treatment/follow-up fields, and staging edition should be excluded to avoid leakage.
