import pandas as pd
import numpy as np

def load_data_5g(path):
    print(f"Loading 5G-NIDD from: {path}")
    df = pd.read_csv(path)
    drop_cols = [col for col in df.columns if
                 'attack' in col.lower() or 'tool' in col.lower() or
                 col.lower().startswith('unnamed') or df[col].isnull().all()]
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df[df['Label'].notna()]
    df['Label'] = df['Label'].map({'Benign': 0, 'Malicious': 1}).astype(int)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    cat_cols = [c for c in cat_cols if c != 'Label']
    if cat_cols:
        print(f"Categorical columns: {cat_cols}")
        df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    y = df['Label'].astype(np.float32)
    X = df.drop('Label', axis=1).apply(pd.to_numeric, errors='coerce').fillna(0).astype(np.float32)
    print(f"Final shape: {X.shape}, Classes: {{0: {(y==0).sum()}, 1: {(y==1).sum()}}}")
    return X, y
