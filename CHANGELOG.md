# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-09-01
### Added
- Initial public release of **AttnDEC-PPR: Robust Anomaly Detection in 5G Networks** repository.
- Experiment **exp01_attnDEC_PPR**:
  - Semi-supervised AttnDEC-HDBSCAN-PPR pipeline.
  - Configs for **5G-NIDD**, **InSDN**, and **CICIDS2017** datasets.
- Experiment **exp02_mlp_attention_5gnidd**:
  - Baselines (MLP-only, Attention-only) on 5G-NIDD.
- Experiment **exp03_graph_appnp_ids_5gnidd**:
  - CPU-only Graph-APPNP baseline on 5G-NIDD (no torch-geometric needed).
- Documentation:
  - `README.md` with quick start and usage instructions.
  - `datasets.md` with Kaggle and official dataset links.
  - `CITATION.cff` with full paper title and authors.
- Licensing:
  - MIT License file.
  - `.gitignore` template for Python projects.

### Notes
- Dataset CSV files must be downloaded manually from Kaggle or official pages (see `datasets.md`).
- Config paths should be updated to match your local environment.
