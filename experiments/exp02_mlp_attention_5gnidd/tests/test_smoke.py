def test_imports():
    import importlib
    for m in ["src.data_5gnidd", "src.models_attn_mlp", "src.train_eval"]:
        importlib.import_module(m)

def test_config_load():
    import yaml, os
    with open(os.path.join("configs","5g_nidd.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg and "split" in cfg and "preprocess" in cfg and "train" in cfg
