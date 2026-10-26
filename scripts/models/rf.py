from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.data.features import FeaturesData
from scripts.evaluation.metrics import (
    compute_metrics,
    save_metrics_csv,
)
from scripts.utils import ensure_dir, save_pickle

MODEL_METADATA = {
    "name": "rf",
    "supported_data_types": ["features"],
    "description": "Random Forest classifier",
    "default_params": {"n_estimators": 100, "random_state": 42},
}


def train(data: FeaturesData, params: Dict[str, Any]) -> Pipeline:
    rf_params = params.copy()
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(**rf_params)),
    ])
    pipeline.fit(data.X_train, data.y_train)
    return pipeline


def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    from scripts.data.features import load_features_data

    output_dir = ensure_dir(output_dir)
    start_time = time.perf_counter()
    data = load_features_data(dataset_config)
    feature_load_time = time.perf_counter() - start_time

    params = bench_config.get_model_params("rf")
    default_params = MODEL_METADATA["default_params"].copy()
    default_params.update(params)

    from scripts.evaluation.metrics_utils import ResourceTracker
    resource_tracker = ResourceTracker()
    resource_tracker.start()

    fit_start = time.perf_counter()
    model = train(data, default_params)
    fit_time = time.perf_counter() - fit_start

    infer_start = time.perf_counter()
    y_train_pred = model.predict(data.X_train)
    y_val_pred = model.predict(data.X_val)

    try:
        y_train_prob = model.predict_proba(data.X_train)
        y_val_prob = model.predict_proba(data.X_val)
    except Exception:
        y_train_prob = None
        y_val_prob = None
    infer_time = time.perf_counter() - infer_start

    resource_stats = resource_tracker.stop()

    train_metrics = compute_metrics(data.y_train, y_train_pred, y_train_prob, data.class_names)
    val_metrics = compute_metrics(data.y_val, y_val_pred, y_val_prob, data.class_names)

    from scripts.evaluation.latency import measure_sklearn_model_latency
    latency_extra = measure_sklearn_model_latency(
        getattr(bench_config, "measure_latency", False),
        model,
        data.X_val,
        dataset_config=dataset_config,
    )

    metrics_path = Path(dataset_config.metrics_path or output_dir / "metrics.csv")
    elapsed = time.perf_counter() - start_time
    n_features = data.X_train.shape[1]
    save_metrics_csv(
        metrics_path,
        "rf",
        dataset_config.name,
        train_metrics,
        val_metrics,
        extra={
            "Epochs": "-",
            "Batch Size": "-",
            "Feature Loading Time (s)": round(feature_load_time, 4),
            "Classifier Fit Time (s)": round(fit_time, 4),
            "Inference Time (s)": round(infer_time, 4),
            "Train+Eval Time (s)": round(elapsed, 4),
            "Total Time (s)": round(elapsed, 4),
            "Key Parameters": f"{default_params['n_estimators']} trees",
            "Input Features": n_features,
            "Input Feature Type": dataset_config.feature_type or dataset_config.name,
            **resource_stats,
            **latency_extra,
        },
    )

    model_path = output_dir / "model.pkl"
    save_pickle(model, model_path)

    return {
        "model_name": "rf",
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "model_path": str(model_path),
    }
