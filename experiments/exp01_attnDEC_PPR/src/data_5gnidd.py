import os
import numpy as np
import pandas as pd

def load_5gnidd(path_csv, binary=True, drop_id_cols=True):
    if not (os.path.isfile(path_csv) and path_csv.lower().endswith(".csv")):
        raise FileNotFoundError("يرجى تمرير ملف CSV صالح (Combined.csv).")

    df = pd.read_csv(path_csv, sep=",", low_memory=False)
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    candidates = [c for c in df.columns if c.lower().strip() in {"label", "class"}]
    if not candidates:
        print("Columns head (first 15):", list(df.columns[:15]))
        raise KeyError("لم يتم العثور على عمود Label.")
    label_col = candidates[0]

    df = df.replace([np.inf, -np.inf], np.nan)

    if binary:
        y = df[label_col].astype(str).str.strip().map({"Benign": 0, "Malicious": 1})
    else:
        y = df[label_col].astype(str).str.strip()

    mask_valid = y.notna()
    df = df.loc[mask_valid].copy()
    y = y.loc[mask_valid].astype(int if binary else "category")

    X = df.drop(columns=[label_col], errors="ignore")
    if drop_id_cols:
        id_like = [
            "Flow ID","Flow_ID","Src IP","Destination IP","Dst IP",
            "Source IP","Timestamp","SimillarHTTP","Fwd Header Length.1"
        ]
        X = X[[c for c in X.columns if c not in id_like]]

    leakage_patterns = ['attack', 'tool', 'type', 'category', 'family', 'label']
    cols_lower = {c: c.lower() for c in X.columns}
    drop_leak = [c for c in X.columns if any(k in cols_lower[c] for k in leakage_patterns)]
    X = X.drop(columns=list(set(drop_leak)), errors='ignore')
    print(f"[filter] dropped {len(set(drop_leak))} potential leakage columns")

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    if binary:
        y = y.astype(np.float32)

    print(f"[load] shape={X.shape}, positives={int((y==1).sum())}, negatives={int((y==0).sum())}")
    return X, y
