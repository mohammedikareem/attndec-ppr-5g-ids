import argparse, yaml, json, os
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix
from .data import load_data
from .model import TinyNet

def eval_model(cfg, ckpt):
    (_, _), (_, _), (Xt, yt) = load_data(cfg)
    model = TinyNet(in_dim=Xt.shape[1],
                    latent_dim=cfg["model"]["latent_dim"],
                    dropout=cfg["model"]["dropout"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(Xt).float())
        prob = torch.softmax(logits, dim=1)[:,1].numpy()
        pred = (prob > 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(yt, pred)),
        "f1": float(f1_score(yt, pred)),
        "roc_auc": float(roc_auc_score(yt, prob)),
        "confusion_matrix": confusion_matrix(yt, pred).tolist()
    }
    precision, recall, _ = precision_recall_curve(yt, prob)
    pr_auc = auc(recall, precision)
    metrics["pr_auc"] = float(pr_auc)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    with open("results/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("[eval] metrics:", metrics)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    eval_model(cfg, args.ckpt)

if __name__ == "__main__":
    main()
