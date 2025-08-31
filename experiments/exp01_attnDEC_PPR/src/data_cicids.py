import os, glob
import numpy as np
import pandas as pd

def load_cicids2017(path, binary=True, drop_id_cols=True):
    """
    path: folder with CSVs or a single CSV file
    binary=True: 0=Benign, 1=Attack
    """
    if os.path.isdir(path):
        csv_files = sorted(glob.glob(os.path.join(path, "*.csv")))
        if len(csv_files) == 0:
            raise FileNotFoundError("No CSV files found in the folder.")
        dfs = []
        for f in csv_files:
            df_tmp = pd.read_csv(f, sep=",", low_memory=False)
            dfs.append(df_tmp)
        df = pd.concat(dfs, ignore_index=True)
    elif os.path.isfile(path) and path.lower().endswith(".csv"):
        df = pd.read_csv(path, sep=",", low_memory=False)
    else:
        raise FileNotFoundError("Path is neither a CSV folder nor a CSV file.")

    # Clean columns
    df.columns = (
        df.columns.astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.strip()
    )

    # Label column
    candidates = [c for c in df.columns if c.lower().strip() in {"label", "class"}]
    if not candidates:
        raise KeyError("Label column not found in CICIDS2017.")
    label_col = candidates[0]

    df = df.replace([np.inf, -np.inf], np.nan)

    if binary:
        y = df[label_col].astype(str).str.upper().str.strip().map(lambda v: 0 if "BENIGN" in v else 1)
    else:
        y = df[label_col].astype(str).str.strip()

    mask_valid = y.notna()
    df = df.loc[mask_valid].copy()
    y = y.loc[mask_valid].astype(int if binary else "category")

    X = df.drop(columns=[label_col], errors="ignore")

    if drop_id_cols:
        id_like = [
            "Flow ID", "Flow_ID", "Src IP", "Destination IP", "Dst IP",
            "Source IP", "Timestamp", "SimillarHTTP", "Fwd Header Length.1"
        ]
        X = X[[c for c in X.columns if c not in id_like]]

    # One-hot
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, dummy_na=True)

    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    if binary:
        y = y.astype(np.float32)

    print(f"[CICIDS] shape={X.shape}, positives={(y==1).sum()}, negatives={(y==0).sum()}")
    return X, y
