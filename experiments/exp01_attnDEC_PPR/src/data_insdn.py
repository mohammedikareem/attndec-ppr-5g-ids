import os
import numpy as np
import pandas as pd

def load_insdn(base_dir, files_map):
    dfs = []
    for fname, label in files_map.items():
        path = os.path.join(base_dir, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        # drop ID/leaky columns
        drop_cols = ['Flow ID','Src IP','Src Port','Dst IP','Dst Port','Timestamp']
        df = df.drop(columns=drop_cols, errors='ignore')
        df['label'] = int(label)
        dfs.append(df)

    data = pd.concat(dfs, ignore_index=True)
    numeric = data.select_dtypes(include=['number']).fillna(0.0).astype('float32')
    if 'label' not in numeric.columns:
        raise KeyError("Column 'label' not found after numeric selection.")
    X = numeric.drop(columns=['label'])
    y = numeric['label'].astype('float32')

    print(f"[load InSDN] shape={X.shape}, positives={(y==1).sum()}, negatives={(y==0).sum()}")
    return X, y
