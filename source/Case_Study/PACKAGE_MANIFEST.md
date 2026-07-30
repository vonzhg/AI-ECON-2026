# Case Study Package Manifest

This folder is organized as a self-contained teaching and provenance package for the Deep Ramsey AI case study.

## Source-Copy Policy

External source repositories are copied into `01_source_code/` with only main source/configuration files using `.py` and `.json` extensions. The package intentionally excludes README files, licenses, C++ files, text data, figures, models, output folders, virtual environments, caches, `.git` metadata, and other generated or non-source artifacts from those external repositories.

The case-study markdown, TeX progression files, and audit notes are retained separately because they are supporting documents, not copied external source artifacts.

## Folder Layout

| Folder | Contents |
|---|---|
| `deep_ramsey_ai_case_study.md` | Main completed case-study narrative. |
| `01_source_code/RA_Ramsey_NN_original/` | Bundled `.py`/`.json` source/config files from the original RA repository. |
| `01_source_code/RA_Ramsey_NN_original/alpha-fold/` | Bundled `.py`/`.json` files from the RA alpha-fold refinement snapshot. |
| `01_source_code/RA_Ramsey_NN_original/working_version_Jan_2026/` | Bundled `.py`/`.json` files from the RA January working version. |
| `01_source_code/RA_Ramsey_NN_original/files_calude_fixed_jan_11_2026/` | Bundled `.py`/`.json` files from the corrected RA snapshot. |
| `01_source_code/RA_local_refined_v2/` | Local refined RA code/config files aligned with the refined RA TeX document. |
| `01_source_code/RA_local_earlier_versions/` | Earlier local RA Python versions retained for provenance. |
| `01_source_code/HA_Ramsey_HA_NN/v1/` | First HA implementation, `.py`/`.json` only. |
| `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/` | Full HA implementation, `.py`/`.json` only. |
| `01_source_code/HA_1_local_snapshot/` | Local HA teaching/support source snapshot, `.py`/`.json` only. |
| `02_tex_model_algorithm_growth/ra_progression/` | RA TeX files tracing the recursive formulation and algorithm refinements. |
| `02_tex_model_algorithm_growth/ha_progression/` | HA TeX files tracing the model, complementarity, and boundary-learning enrichment. |
| `03_notes_and_audits/` | Markdown audits, dialogue notes, and supporting writeups. |
| `99_archive_metadata/` | Archive manifest from the earlier cleanup step. |

## Quick Entry Points

- Main case study: `deep_ramsey_ai_case_study.md`
- RA original source: `01_source_code/RA_Ramsey_NN_original/dashboard.py`
- HA first version: `01_source_code/HA_Ramsey_HA_NN/v1/dashboard.py`
- HA full model: `01_source_code/HA_Ramsey_HA_NN/v2_fullmodel/dashboard.py`
- RA model-growth TeX: `02_tex_model_algorithm_growth/ra_progression/`
- HA model-growth TeX: `02_tex_model_algorithm_growth/ha_progression/`

## Verification

As of the latest organization pass, every file under `01_source_code/` has extension `.py` or `.json`.
