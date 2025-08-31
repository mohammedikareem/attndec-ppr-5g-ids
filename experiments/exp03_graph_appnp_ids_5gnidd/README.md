# Experiment: Graph-APPNP-IDS (CPU-Only) — 5G-NIDD

A **fast CPU-only** APPNP (PPR-like) baseline for 5G-NIDD using:
- Approximate kNN graph via **pynndescent**
- Full-batch propagation on a **sparse normalized adjacency** with plain PyTorch
- **No** torch-geometric required

> This is a faithful modularization of the user's script. You can run on stratified samples (fast) or full data (slower).

## Quick start
```bash
pip install -r env/requirements.txt
python -m src.appnp_baseline --config configs/5g_nidd_cpu.yaml
```

Outputs:
- `results/metrics.json`  (VAL/TEST/TRAIN: Acc / ROC-AUC / PR-AUC / F1)
- Printed logs per epoch with early stopping
