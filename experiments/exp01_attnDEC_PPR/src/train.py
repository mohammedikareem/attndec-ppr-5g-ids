import argparse, yaml, json, os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from .data import load_data
from .model import TinyNet

def train(cfg):
    (Xtr, ytr), (Xval, yval), _ = load_data(cfg)
    model = TinyNet(in_dim=Xtr.shape[1],
                    latent_dim=cfg["model"]["latent_dim"],
                    dropout=cfg["model"]["dropout"])
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["train"]["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()

    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True)

    model.train()
    for epoch in range(cfg["train"]["epochs"]):
        for xb, yb in dl:
            optimizer.zero_grad()
            logits = model(xb.float())
            loss = loss_fn(logits, yb.long())
            loss.backward()
            optimizer.step()
        print(f"[train] epoch {epoch+1}/{cfg['train']['epochs']} done.")
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/best.pt")
    print("[train] saved checkpoints/best.pt")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    train(cfg)

if __name__ == "__main__":
    main()
