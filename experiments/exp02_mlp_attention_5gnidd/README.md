# Experiment: MLP & Attention Baselines on 5G-NIDD

Two supervised baselines implemented from the user's Colab:
- **Attention-only** classifier
- **MLP-only** classifier

The pipeline performs: load → clean → one-hot → scale (MinMax) → Chi² feature selection → train/evaluate, and saves figures + metrics.

## Quick start
```bash
pip install -r env/requirements.txt
python -m src.train_eval --config configs/5g_nidd.yaml --model attention
python -m src.train_eval --config configs/5g_nidd.yaml --model mlp
```

Outputs:
- `results/metrics_attention.json` / `results/metrics_mlp.json`
- `results/figures/{model}_cm.png`
- (optional) `results/figures/{model}_roc.png`, `{model}_pr.png`
