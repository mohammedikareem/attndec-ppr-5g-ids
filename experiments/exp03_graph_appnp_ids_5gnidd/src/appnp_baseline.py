import os, argparse, yaml, json, time, numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, average_precision_score, f1_score)
from imblearn.over_sampling import SMOTE
from pynndescent import NNDescent

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data_5gnidd_simple import load_5g_binary

def stratified_sample(X, y, n, seed=42):
    if n is None or n >= len(X):
        return X, y
    Xs, _, ys, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=seed)
    return Xs, ys

def variance_topk(X, k):
    if k is None or k >= X.shape[1]:
        return X, list(X.columns)
    var = X.var().sort_values(ascending=False)
    feats = list(var.index[:k])
    return X[feats].copy(), feats

def build_knn_graph(X_all, k=10, metric='cosine', seed=42):
    n_all = len(X_all)
    index = NNDescent(X_all.values.astype(np.float32), n_neighbors=min(k+1, n_all),
                      metric=metric, random_state=seed, n_jobs=-1)
    idx, _ = index.query(X_all.values.astype(np.float32), k=min(k+1, n_all))
    src = np.repeat(np.arange(n_all), idx.shape[1]-1)
    dst = idx[:,1:].reshape(-1)
    edges = np.vstack([src, dst])
    edges_rev = np.vstack([dst, src])
    edges = np.hstack([edges, edges_rev])
    edges = np.unique(edges, axis=1)
    return edges

def sparse_norm_adjacency(edges, n_all):
    self_edges = np.vstack([np.arange(n_all), np.arange(n_all)])
    edges = np.hstack([edges, self_edges])
    deg = np.bincount(edges[0], minlength=n_all)
    vals = 1.0 / np.sqrt(deg[edges[0]] * deg[edges[1]] + 1e-12)
    i_idx = torch.tensor(edges, dtype=torch.long)
    v_idx = torch.tensor(vals,  dtype=torch.float32)
    return torch.sparse_coo_tensor(i_idx, v_idx, size=(n_all, n_all)).coalesce()

class MLP_APPNP(nn.Module):
    def __init__(self, in_dim, hidden=64, out_dim=2, dropout=0.2, K=10, alpha=0.1, S=None):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, out_dim)
        self.dp   = nn.Dropout(dropout)
        self.K = K
        self.alpha = alpha
        self.S = S

    def appnp_propagate(self, Z):
        H = Z
        for _ in range(self.K):
            H = self.alpha * Z + (1.0 - self.alpha) * torch.sparse.mm(self.S, H)
        return H

    def forward(self, x):
        h = F.relu(self.lin1(x))
        h = self.dp(h)
        Z = self.lin2(h)
        H = self.appnp_propagate(Z)
        return H

def evaluate(model, X_t, y_t, split_idx, name):
    model.eval()
    with torch.no_grad():
        logits = model(X_t)
        prob = F.softmax(logits, dim=1)[:,1].cpu().numpy()
        pred = (prob > 0.5).astype(int)
        ytrue = y_t[split_idx].cpu().numpy()
        p = prob[split_idx]; yhat = pred[split_idx]
        acc = accuracy_score(ytrue, yhat)
        roc = roc_auc_score(ytrue, p)
        pr  = average_precision_score(ytrue, p)
        f1  = f1_score(ytrue, yhat)
    print(f"{name}: Acc {acc:.4f} | ROC-AUC {roc:.4f} | PR-AUC {pr:.4f} | F1 {f1:.4f}")
    return {"accuracy":float(acc),"roc_auc":float(roc),"pr_auc":float(pr),"f1":float(f1)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # --- Load & prepare ---
    X, y = load_5g_binary(
        cfg["data"]["csv_path"],
        cfg["data"].get("label_col","Label"),
        cfg["data"].get("positive_when_not_benign", True),
        cfg["data"].get("drop_cols", [])
    )
    X, y = stratified_sample(X, y, cfg["data"].get("sample_size"), cfg["data"]["random_state"])
    X, feats = variance_topk(X, cfg["data"].get("top_k_features"))

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=cfg["split"]["test_size"],
                                                stratify=y if cfg["split"]["stratify"] else None,
                                                random_state=cfg["data"]["random_state"])
    X_va, X_te, y_va, y_te   = train_test_split(X_tmp, y_tmp, test_size=cfg["split"]["val_size_from_tmp"],
                                                stratify=y_tmp if cfg["split"]["stratify"] else None,
                                                random_state=cfg["data"]["random_state"])

    scaler = MinMaxScaler()
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_va = pd.DataFrame(scaler.transform(X_va), columns=X_va.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

    # Optional SMOTE if heavily imbalanced
    if cfg["balance"]["smote"]:
        vc = y_tr.value_counts()
        if (vc.min() / vc.max()) < 0.7:
            X_tr, y_tr = SMOTE(random_state=cfg["data"]["random_state"]).fit_resample(X_tr, y_tr)

    # Build graph on all splits (like script)
    X_all = pd.concat([X_tr, X_va, X_te], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_va, y_te], axis=0).reset_index(drop=True)
    n_all = len(X_all)

    edges = build_knn_graph(X_all, k=cfg["graph"]["k_neighbors"], metric=cfg["graph"]["metric"],
                            seed=cfg["data"]["random_state"])
    S = sparse_norm_adjacency(edges, n_all)

    # Indices
    n_tr, n_va = len(X_tr), len(X_va)
    idx_train = np.arange(0, n_tr)
    idx_val   = np.arange(n_tr, n_tr+n_va)
    idx_test  = np.arange(n_tr+n_va, n_all)

    # Tensors
    X_t = torch.tensor(X_all.values, dtype=torch.float32)
    y_t = torch.tensor(y_all.values, dtype=torch.long)

    # Model & training prep
    pos_ratio = y_t[idx_train].float().mean().item()
    w0 = 1.0 / (1.0 - pos_ratio + 1e-9)
    w1 = 1.0 / (pos_ratio + 1e-9)
    class_weights = torch.tensor([w0, w1], dtype=torch.float32)

    model = MLP_APPNP(
        in_dim=X_t.shape[1], hidden=cfg["model"]["hidden"], out_dim=2,
        dropout=cfg["model"]["dropout"], K=cfg["model"]["K_prop"],
        alpha=cfg["model"]["alpha"], S=S
    )
    opt  = torch.optim.AdamW(model.parameters(), lr=cfg["model"]["lr"], weight_decay=cfg["model"]["weight_decay"])
    crit = nn.CrossEntropyLoss(weight=class_weights)

    best_score, best_state, wait = -1, None, 0

    for ep in range(cfg["model"]["epochs"]):
        model.train()
        opt.zero_grad()
        logits = model(X_t)  # full-batch
        loss = crit(logits[idx_train], y_t[idx_train])
        loss.backward()
        opt.step()

        # validation
        val_metrics = evaluate(model, X_t, y_t, idx_val, "VAL")
        score = val_metrics["roc_auc"] + val_metrics["pr_auc"] + val_metrics["f1"]
        if score > best_score:
            best_score, best_state, wait = score, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= cfg["model"]["patience"]:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    print("\\n=== FINAL EVALUATION ===")
    test_metrics  = evaluate(model, X_t, y_t, idx_test, "TEST")
    train_metrics = evaluate(model, X_t, y_t, idx_train, "TRAIN")

    out_dir = cfg["paths"]["results_dir"]
    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "features": feats,
        "val": val_metrics,
        "test": test_metrics,
        "train": train_metrics,
        "config": cfg
    }
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(payload, f, indent=2)

if __name__ == "__main__":
    main()
