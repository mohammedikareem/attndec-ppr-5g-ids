# AttnDEC-PPR: Robust Anomaly Detection in 5G Networks

Official code repository for the paper:

**AttnDEC-PPR: An Attention-Driven Deep Embedded Clustering Model with Personalized PageRank Propagation for Robust Anomaly Detection in 5G Networks**  
*Mohammed Ibrahim Kareem, Ali Z.K. Matloob, Karrar Imran Ogaili, Ali Kadhum M. Al-Qurabat*  
(Corresponding author: mohamed.ibrahim@uobabylon.edu.iq)

---

> Official code for *"AttnDEC-PPR: Robust Anomaly Detection in 5G Networks"*.  
Includes semi-supervised AttnDEC-HDBSCAN-PPR on 5G-NIDD, InSDN, and CICIDS2017,  
plus supervised baselines (MLP, Attention) and a CPU-only Graph-APPNP baseline.

**GitHub Topics:**  
`5g-networks`, `intrusion-detection`, `anomaly-detection`, `deep-learning`,  
`attention-mechanism`, `graph-neural-networks`, `clustering`, `sdn`, `ids`,  
`cicids2017`, `insdn`, `nidd`, `python`, `machine-learning`

## 📂 Repository Structure
- `experiments/exp01_attnDEC_PPR/` — Semi-supervised AttnDEC-HDBSCAN-PPR on **5G-NIDD**, **InSDN**, **CICIDS2017**.  
- `experiments/exp02_mlp_attention_5gnidd/` — Baselines (MLP-only, Attention-only) on 5G-NIDD.  
- `experiments/exp03_graph_appnp_ids_5gnidd/` — CPU-only Graph-APPNP baseline on 5G-NIDD.

## 🚀 Quick Start
```bash
git clone https://github.com/mohammedikareem/SDN-IDS-Experiments.git
cd SDN-IDS-Experiments/experiments/exp01_attnDEC_PPR
pip install -r env/requirements.txt
python -m src.pipeline --config configs/5g_nidd.yaml --mode train_eval
```

To run on **InSDN**:
```bash
python -m src.pipeline --config configs/insdn.yaml --mode train_eval
```

To run on **CICIDS2017** (PortScan subset or folder/CSV):
```bash
python -m src.pipeline --config configs/cicids_portscan.yaml --mode train_eval
```

For **Graph-APPNP baseline**:
```bash
cd experiments/exp03_graph_appnp_ids_5gnidd
pip install -r env/requirements.txt
python -m src.appnp_baseline --config configs/5g_nidd_cpu.yaml
```

## 📊 Datasets
See **[datasets.md](./datasets.md)** for official download links and how to set file paths in configs.

## 📜 License
MIT License © 2025 Mohammed Ibrahim Kareem

## 📖 Citation
If you use this repository, please cite:

```
@article{Kareem2025AttnDEC-PPR,
  title   = {AttnDEC-PPR: An Attention-Driven Deep Embedded Clustering Model with Personalized PageRank Propagation for Robust Anomaly Detection in 5G Networks},
  author  = {Mohammed Ibrahim Kareem and Ali Z.K. Matloob and Karrar Imran Ogaili and Ali Kadhum M. Al-Qurabat},
  journal = {TBD},
  year    = {2025},
  doi     = {TBD}
}
```
