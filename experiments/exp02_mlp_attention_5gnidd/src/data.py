import os, json

def load_data(cfg):
    # Placeholder loader — replace with real dataset code.
    ds = cfg.get("dataset", "unknown")
    root = cfg.get("data_root", ".")
    print(f"[data] Loading {ds} from {root}")
    # Return tiny fake tensors/arrays to keep smoke tests fast.
    import numpy as np
    X = np.random.randn(100, 16).astype("float32")
    y = (np.random.rand(100) > 0.5).astype("int64")
    return (X, y), (X, y), (X, y)
