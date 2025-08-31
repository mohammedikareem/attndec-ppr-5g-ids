# Experiment: AttnDEC‑HDBSCAN‑PPR (5G‑NIDD first pass)

This folder contains a faithful, modularized implementation of the **AttnDEC‑HDBSCAN‑PPR** pipeline on **5G‑NIDD** (no baselines), extracted from the user's Colab script, with config-driven hyperparameters and reproducible outputs.

> Datasets for CICIDS2017‑PortScan and InSDN can be added next using the same `src/pipeline.py` entrypoint with different configs.

## Quick start
```bash
pip install -r env/requirements.txt
python -m src.pipeline --config configs/5g_nidd.yaml --mode train_eval
```
Outputs:
- `results/metrics.json`
- `results/figures/roc.png`, `pr.png`, `cm.png`


## InSDN run example
```bash
cd experiments/exp01_attnDEC_PPR
pip install -r env/requirements.txt
python -m src.pipeline --config configs/insdn.yaml --mode train_eval
```

## CICIDS2017 run example
```bash
cd experiments/exp01_attnDEC_PPR
pip install -r env/requirements.txt
python -m src.pipeline --config configs/cicids_portscan.yaml --mode train_eval
```
