from scripts.config.benchmark_config import BenchmarkConfig

"""

trad_ml = BenchmarkConfig(
    name="my_opensmile_benchmark",
    models=["knn", "rf", "mlp"],
    model_params={
        "knn": {"n_neighbors": 5},
        "rf": {"n_estimators": 100, "random_state": 42},
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500},
    },
    output_dir="/Users/erikverdiermontero/Downloads/TFG_benchmarking/results",
    seed=42,
)

"""

trad_ml = BenchmarkConfig(
    name="my_opensmile_benchmark",
    models=["mlp"],
    model_params={
        "mlp": {"hidden_layer_sizes": (12,), "max_iter": 500},
    },
    output_dir="/Users/erikverdiermontero/Downloads/TFG_benchmarking/results",
    seed=42,
)


