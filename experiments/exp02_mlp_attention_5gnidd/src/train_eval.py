import os, argparse, yaml, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, precision_recall_curve,
                             average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score)

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from .data_5gnidd import load_data_5g
from .models_attn_mlp import create_attention_model, create_mlp_model

def dynamic_feature_selection(X_train, y_train, min_k=10, max_k=50):
    print("\\nPerforming dynamic feature selection...")
    scaler = MinMaxScaler()
    Xs = scaler.fit_transform(X_train)
    chi_scores, _ = chi2(Xs, y_train)
    n_features = X_train.shape[1]
    min_k = max(10, int(n_features * 0.1), min_k)
    max_k = min(50, int(n_features * 0.5), max_k)
    sorted_scores = np.sort(chi_scores)[::-1]
    diffs = np.diff(sorted_scores)
    elbow_point = np.argmax(diffs < 0) + 1
    k = max(min_k, min(max_k, elbow_point))
    print(f"Automatically selected k={k} features")
    top_k_indices = np.argsort(chi_scores)[-k:][::-1]
    feature_names = X_train.columns[top_k_indices]
    return list(feature_names)

def evaluate_model(model, X, y, set_name, out_dir, tag):
    y_pred_prob = model.predict(X, verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_pred_prob)
    pr_auc = average_precision_score(y, y_pred_prob)
    print(f"\\n=== {set_name} Set ({tag}) ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print("\\nClassification Report:\\n", classification_report(y, y_pred, target_names=['Normal','Attack']))

    # Save confusion matrix plot
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'{set_name} Confusion Matrix — {tag}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    os.makedirs(out_dir, exist_ok=True)
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{tag}_cm.png"), dpi=150); plt.close()

    # Save ROC
    fpr, tpr, _ = roc_curve(y, y_pred_prob)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f'AUC={roc:.3f}')
    plt.plot([0,1], [0,1], '--')
    plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title(f'ROC — {tag}')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{tag}_roc.png"), dpi=150); plt.close()

    # Save PR
    pre, rec, _ = precision_recall_curve(y, y_pred_prob)
    ap = average_precision_score(y, y_pred_prob)
    plt.figure(figsize=(5,4))
    plt.plot(rec, pre, label=f'AP={ap:.3f}')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title(f'PR — {tag}')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{tag}_pr.png"), dpi=150); plt.close()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "cm": cm.tolist()
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", choices=["attention","mlp"], required=True)
    args = ap.parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Load data
    X, y = load_data_5g(cfg["data"]["csv_path"])

    # Train/val/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y if cfg["split"]["stratify"] else None,
        random_state=cfg["split"]["random_state"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=cfg["split"]["val_size_from_tmp"], stratify=y_test if cfg["split"]["stratify"] else None,
        random_state=cfg["split"]["random_state"]
    )

    # Feature selection
    if cfg["preprocess"]["chi2_dynamic"]:
        top_features = dynamic_feature_selection(
            X_train, y_train, cfg["preprocess"]["chi2_min_k"], cfg["preprocess"]["chi2_max_k"]
        )
    else:
        top_features = list(X_train.columns)

    # Scale
    scaler = MinMaxScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns).astype("float32")
    X_val_s   = pd.DataFrame(scaler.transform(X_val),  columns=X_val.columns).astype("float32")
    X_test_s  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns).astype("float32")

    X_train_top = X_train_s[top_features].astype("float32")
    X_val_top   = X_val_s[top_features].astype("float32")
    X_test_top  = X_test_s[top_features].astype("float32")

    # Build model
    if args.model == "attention":
        model = create_attention_model(X_train_top.shape[1])
        tag = "attention"
    else:
        model = create_mlp_model(X_train_top.shape[1])
        tag = "mlp"

    # Callbacks
    es = EarlyStopping(monitor=cfg["train"]["earlystop_monitor"], patience=cfg["train"]["earlystop_patience"],
                       mode='max', restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor=cfg["train"]["earlystop_monitor"], factor=cfg["train"]["reduce_lr_factor"],
                              patience=cfg["train"]["reduce_lr_patience"], mode='max')

    # Train
    model.fit(
        X_train_top, y_train,
        validation_data=(X_val_top, y_val),
        epochs=cfg["train"]["epochs"],
        batch_size=cfg["train"]["batch_size"],
        callbacks=[es, rlrop],
        verbose=1
    )

    # Evaluate
    results_dir = os.path.join(cfg["paths"]["results_dir"], "figures")
    os.makedirs(results_dir, exist_ok=True)
    metrics = evaluate_model(model, X_test_top, y_test, "Test", results_dir, tag)

    # Save metrics
    os.makedirs(cfg["paths"]["results_dir"], exist_ok=True)
    with open(os.path.join(cfg["paths"]["results_dir"], f"metrics_{tag}.json"), "w") as f:
        import json; json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    main()
