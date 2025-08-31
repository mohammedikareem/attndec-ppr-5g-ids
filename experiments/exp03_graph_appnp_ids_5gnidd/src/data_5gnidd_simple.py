import pandas as pd
import numpy as np

def load_5g_binary(csv_path, label_col="Label", positive_when_not_benign=True, drop_cols=None):
    assert isinstance(csv_path, str) and csv_path, "CSV path must be provided"
    df = pd.read_csv(csv_path)
    assert label_col in df.columns, f"{label_col} not found in CSV"
    df = df[df[label_col].notna()].copy()
    if positive_when_not_benign:
        df['label'] = (df[label_col] != 'Benign').astype(int)
    else:
        df['label'] = (df[label_col] == 'Malicious').astype(int)
    drop_cols = drop_cols or []
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    num = df.select_dtypes(include=['number']).fillna(0.0)
    assert 'label' in num.columns, "label column missing after numeric filter"
    X = num.drop(columns=['label'])
    y = num['label'].astype(int)
    return X, y
