def test_imports():
    import importlib
    for m in ["src.data_5gnidd_simple", "src.appnp_baseline"]:
        importlib.import_module(m)

def test_config_load():
    import yaml, os
    with open(os.path.join("configs","5g_nidd_cpu.yaml")) as f:
        cfg = yaml.safe_load(f)
    assert "data" in cfg and "graph" in cfg and "model" in cfg
