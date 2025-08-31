import os, argparse, yaml, json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2

import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, average_precision_score,
                             matthews_corrcoef, balanced_accuracy_score,
                             roc_curve, precision_recall_curve, auc, ConfusionMatrixDisplay)

import hdbscan
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from hdbscan import prediction as hdb_pred

from .data_5gnidd import load_5gnidd
from .data_insdn import load_insdn
from .data_cicids import load_cicids2017
from .attn_autoencoder import build_attn_autoencoder, fit_autoencoder

def calc_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    auc_ = roc_auc_score(y_true, y_proba)
    ap   = average_precision_score(y_true, y_proba)
    mcc  = matthews_corrcoef(y_true, y_pred)
    bal  = balanced_accuracy_score(y_true, y_pred)
    spec = tn / (tn + fp) if (tn+fp) > 0 else 0.0
    gmean = np.sqrt(max(rec,0.0)*max(spec,0.0))
    return acc, prec, rec, f1, auc_, mcc, bal, spec, gmean, ap, (tn, fp, fn, tp)

def ppr_predict_with_proba(
    z_sample, Z_train, y_train, clusterer, cluster_labels, cluster_classes, cluster_mal_ratio,
    k=12, alpha=0.85, sim_edge=0.80
):
    pred_label, _ = hdb_pred.approximate_predict(clusterer, z_sample.reshape(1, -1))
    sample_cluster = int(pred_label[0])

    if sample_cluster == -1 or sample_cluster not in cluster_classes:
        centers = np.array([Z_train[cluster_labels == c].mean(axis=0) for c in cluster_classes.keys()])
        dists = np.linalg.norm(centers - z_sample, axis=1)
        sample_cluster = list(cluster_classes.keys())[int(np.argmin(dists))]

    idx = np.where(cluster_labels == sample_cluster)[0]
    if len(idx) == 0:
        p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        return (1 if p >= 0.5 else 0), p

    Z_cluster = Z_train[idx]
    sims = cosine_similarity(z_sample.reshape(1, -1), Z_cluster)[0]
    k_eff = int(min(k, len(idx)))
    top_k_idx = sims.argsort()[-k_eff:]
    neighbor_idx = idx[top_k_idx]
    neighbor_sims = sims[top_k_idx]

    G = nx.Graph()
    G.add_node("sample")
    for i, ni in enumerate(neighbor_idx):
        G.add_node(int(ni))
        G.add_edge("sample", int(ni), weight=float(neighbor_sims[i]))
    for i in range(len(neighbor_idx)):
        zi = Z_train[neighbor_idx[i]].reshape(1, -1)
        for j in range(i+1, len(neighbor_idx)):
            zj = Z_train[neighbor_idx[j]].reshape(1, -1)
            sim_ij = float(cosine_similarity(zi, zj)[0, 0])
            if sim_ij >= sim_edge:
                G.add_edge(int(neighbor_idx[i]), int(neighbor_idx[j]), weight=sim_ij)

    personalization = {"sample": 0.0}
    for ni in neighbor_idx:
        lbl = int(y_train[ni])
        personalization[int(ni)] = 1.0 if lbl == 1 else 0.25

    pr = nx.pagerank(G, personalization=personalization, alpha=alpha, weight='weight')

    pr_neighbors = np.array([pr[int(ni)] for ni in neighbor_idx], dtype=float)
    y_neighbors  = np.array([int(y_train[ni]) for ni in neighbor_idx], dtype=int)

    if pr_neighbors.size == 0:
        p = float(cluster_mal_ratio.get(sample_cluster, 0.0))
        return (1 if p >= 0.5 else 0), p

    sum_total = float(pr_neighbors.sum())
    sum_mal   = float(pr_neighbors[y_neighbors == 1].sum())
    p = float(sum_mal / sum_total) if sum_total > 0 else float(cluster_mal_ratio.get(sample_cluster, 0.0))

    return (1 if p >= 0.5 else 0), p

def run(cfg):
    # --- Load data ---
    ds = cfg.get("data", {}).get("dataset", "5g-nidd").lower()
    if ds in ["insdn", "in-sdn"]:
        X, y = load_insdn(cfg["data"]["base_dir"], cfg["data"]["files"])
    elif ds in ["cicids","cicids2017","cicids-2017"]:
        X, y = load_cicids2017(cfg["data"]["path"], cfg["data"]["binary"], cfg["data"]["drop_id_cols"])
    else:
        X, y = load_5gnidd(cfg["data"]["csv_path"], cfg["data"]["binary"], cfg["data"]["drop_id_cols"])

    # --- splits ---
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=cfg["data"]["random_state"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=cfg["data"]["val_size_from_tmp"], stratify=y_tmp, random_state=cfg["data"]["random_state"]
    )

    # --- scaling + chi2 ---
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled   = scaler.transform(X_val)
    X_test_scaled  = scaler.transform(X_test)

    k = cfg["preprocess"]["select_k_best"]["k"]
    selector = SelectKBest(chi2, k=k)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel   = selector.transform(X_val_scaled)
    X_test_sel  = selector.transform(X_test_scaled)

    selected_cols = X.columns[selector.get_support()]

    # --- Optional supervised baselines on CICIDS (for quick comparison) ---
    if cfg.get("baselines", {}).get("enabled", False) and ds.startswith("cicids"):
        from .baselines_supervised import run_supervised_baselines
        out_fig = os.path.join(cfg["paths"]["results_dir"], "figures")
        os.makedirs(out_fig, exist_ok=True)
        _ = run_supervised_baselines(X_train_sel, (y_train.values if hasattr(y_train,'values') else y_train),
                                     X_test_sel, (y_test.values if hasattr(y_test,'values') else y_test),
                                     out_fig, sample_n=cfg["baselines"].get("sample_n"))

    print(f"Selected features (k={k}): {list(selected_cols)}")

    # --- AE ---
    input_dim = X_train_sel.shape[1]
    autoencoder, encoder = build_attn_autoencoder(
        input_dim=input_dim,
        latent_dim=cfg["model"]["latent_dim"],
        heads=cfg["model"]["attn_heads"],
        key_dim=cfg["model"]["attn_key_dim"],
        dropout=cfg["model"]["dropout"]
    )
    fit_autoencoder(
        autoencoder,
        X_train_sel, X_val_sel,
        epochs=cfg["model"]["epochs"],
        batch_size=cfg["model"]["batch_size"],
        patience=cfg["model"]["patience"],
        verbose=2
    )

    Z_train = encoder.predict(X_train_sel, verbose=0)
    Z_val   = encoder.predict(X_val_sel,   verbose=0)
    Z_test  = encoder.predict(X_test_sel,  verbose=0)

    # --- HDBSCAN ---
    clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg["cluster"]["min_cluster_size"], prediction_data=True)
    cluster_labels = clusterer.fit_predict(Z_train)

    valid_clusters = [c for c in np.unique(cluster_labels) if c != -1]
    cluster_classes = {}
    cluster_mal_ratio = {}
    y_train_arr = y_train.values if hasattr(y_train, 'values') else y_train
    for c in valid_clusters:
        idx = (cluster_labels == c)
        lbls = y_train_arr[idx].astype(int)
        majority = 0 if len(lbls) == 0 else np.bincount(lbls).argmax()
        cluster_classes[c] = int(majority)
        cluster_mal_ratio[c] = float(lbls.mean()) if len(lbls) else 0.0

    print("\\n=== Cluster Stats ===")
    print(f"{'Cluster':<8} {'Samples':<8} {'Percent':<8} {'Majority':<10} {'Mal.Ratio':<10}")
    N_tr = len(cluster_labels)
    for c in valid_clusters:
        idx = (cluster_labels == c)
        cnt = int(idx.sum())
        perc = 100.0 * cnt / max(1, N_tr)
        print(f"{c:<8} {cnt:<8} {perc:7.2f}% {cluster_classes[c]:<10} {cluster_mal_ratio[c]:<10.3f}")

    # --- PPR Predictions on test ---
    y_test_np = (y_test.values if hasattr(y_test, "values") else y_test).astype(int)
    y_pred, y_proba = [], []
    for z in Z_test:
        lbl, p = ppr_predict_with_proba(
            z, Z_train, y_train_arr, clusterer, cluster_labels, cluster_classes, cluster_mal_ratio,
            k=cfg["ppr"]["k"], alpha=cfg["ppr"]["alpha"], sim_edge=cfg["ppr"]["sim_edge"]
        )
        y_pred.append(lbl); y_proba.append(p)
    y_pred = np.array(y_pred, dtype=int)
    y_proba = np.array(y_proba, dtype=float)

    # --- Metrics ---
    acc, prec, rec, f1, roc_, mcc, bal, spec, gmean, ap, cm = calc_metrics(y_test_np, y_pred, y_proba)
    metrics = {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec), "f1": float(f1),
        "roc_auc": float(roc_), "mcc": float(mcc), "balanced_acc": float(bal),
        "specificity": float(spec), "gmean": float(gmean), "pr_auc": float(ap),
        "confusion_matrix": {"tn": int(cm[0]), "fp": int(cm[1]), "fn": int(cm[2]), "tp": int(cm[3])},
        "selected_features": list(map(str, selected_cols))
    }

    # --- Save ---
    results_dir = cfg["paths"]["results_dir"]
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "tables"), exist_ok=True)

    with open(os.path.join(results_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("[eval] metrics:", json.dumps(metrics, indent=2))

    # --- Plots ---
    fpr, tpr, _ = roc_curve(y_test_np, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.3f}')
    plt.plot([0,1], [0,1], '--')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC — AttnDEC‑HDBSCAN‑PPR (5G‑NIDD)'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "figures", "roc.png"), dpi=150)
    plt.close()

    pre, rec_, _ = precision_recall_curve(y_test_np, y_proba)
    ap_ = average_precision_score(y_test_np, y_proba)
    plt.figure(figsize=(6,5))
    plt.plot(rec_, pre, label=f'AP={ap_:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision')
    plt.title('PR — AttnDEC‑HDBSCAN‑PPR (5G‑NIDD)'); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "figures", "pr.png"), dpi=150)
    plt.close()

    ConfusionMatrixDisplay.from_predictions(y_test_np, y_pred)
    plt.title('Confusion Matrix — AttnDEC‑HDBSCAN‑PPR (5G‑NIDD)')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "figures", "cm.png"), dpi=150)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["train_eval"], default="train_eval")
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    os.makedirs(cfg["paths"]["checkpoints_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    run(cfg)

if __name__ == "__main__":
    main()
