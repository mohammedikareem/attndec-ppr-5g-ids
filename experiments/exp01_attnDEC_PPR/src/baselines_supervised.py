import numpy as np, os, json
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix,
                             roc_curve, precision_recall_curve, ConfusionMatrixDisplay)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def _calc_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)
    ap  = average_precision_score(y_true, y_proba)
    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1,"roc_auc":auc,"pr_auc":ap}

def run_supervised_baselines(X_train_sel, y_train, X_test_sel, y_test, out_dir, sample_n=4000):
    os.makedirs(out_dir, exist_ok=True)
    # Optional sample for speed
    if sample_n is not None and sample_n < len(X_test_sel):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_test_sel), sample_n, replace=False)
        X_te = X_test_sel[idx]; y_te = y_test[idx]
    else:
        X_te, y_te = X_test_sel, y_test

    # SVM (RBF, balanced)
    svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_sel, y_train)
    y_svm = svm.predict(X_te); y_svm_p = svm.predict_proba(X_te)[:,1]
    m_svm = _calc_metrics(y_te, y_svm, y_svm_p)

    # MLP (sklearn)
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=30, random_state=42)
    mlp.fit(X_train_sel, y_train)
    y_mlp = mlp.predict(X_te); y_mlp_p = mlp.predict_proba(X_te)[:,1]
    m_mlp = _calc_metrics(y_te, y_mlp, y_mlp_p)

    # Save basic plots
    fpr_svm, tpr_svm, _ = roc_curve(y_te, y_svm_p)
    fpr_mlp, tpr_mlp, _ = roc_curve(y_te, y_mlp_p)
    plt.figure(figsize=(6,5)); plt.plot(fpr_svm,tpr_svm,label='SVM'); plt.plot(fpr_mlp,tpr_mlp,label='MLP')
    plt.plot([0,1],[0,1],'--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC — Supervised Baselines')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,"baselines_roc.png"), dpi=150); plt.close()

    pre_svm, rec_svm, _ = precision_recall_curve(y_te, y_svm_p)
    pre_mlp, rec_mlp, _ = precision_recall_curve(y_te, y_mlp_p)
    plt.figure(figsize=(6,5)); plt.plot(rec_svm,pre_svm,label='SVM'); plt.plot(rec_mlp,pre_mlp,label='MLP')
    plt.xlabel('Recall'); plt.ylabel('Precision'); plt.title('PR — Supervised Baselines')
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir,"baselines_pr.png"), dpi=150); plt.close()

    # Confusion Matrices
    for name, yhat in [('SVM', y_svm), ('MLP', y_mlp)]:
        disp = ConfusionMatrixDisplay.from_predictions(y_te, yhat)
        disp.ax_.set_title(f'Confusion Matrix — {name}'); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"cm_{name.lower()}.png"), dpi=150); plt.close()

    # Save metrics
    with open(os.path.join(out_dir, "metrics_supervised.json"), "w") as f:
        json.dump({"svm":m_svm,"mlp":m_mlp}, f, indent=2)

    return {"svm":m_svm,"mlp":m_mlp}
