"""Inference-latency measurement utilities.

Separates encoder and classifier cost for the fine-tuned models (CLAP /
Whisper), and measures classifier/inference cost for the traditional models.
All timings are wall clock on the active device with explicit CUDA
synchronisation, a warmup phase, and a fixed number of iterations.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np


def _sync(device: Optional[str]) -> None:
    try:
        import torch

        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def time_callable(
    fn: Callable[[], object],
    warmups: int = 10,
    iters: int = 50,
    device: Optional[str] = None,
) -> Dict[str, float]:
    """Times ``fn`` over ``iters`` calls. Returns mean ms and calls/second."""
    for _ in range(max(0, warmups)):
        fn()
    _sync(device)

    start = time.perf_counter()
    for _ in range(max(1, iters)):
        fn()
    _sync(device)
    elapsed = time.perf_counter() - start

    return {
        "mean_ms": (elapsed * 1000.0) / max(1, iters),
        "calls_per_s": (max(1, iters) / elapsed) if elapsed > 0 else 0.0,
    }


def to_latency_columns(
    component_stats: Dict[str, Dict[str, float]],
    batch_size: int,
) -> Dict[str, float]:
    """Flattens component stats into per-window CSV columns."""
    out: Dict[str, float] = {}
    for name, stats in component_stats.items():
        label = name.replace("_", " ").title()
        out[f"{label} Latency (ms/window)"] = round(
            float(stats["mean_ms"]) / batch_size, 4
        )
        out[f"{label} Throughput (windows/s)"] = round(
            float(stats["calls_per_s"]) * batch_size, 3
        )
    return out


def measure_clap_components(
    model,
    input_features,
    is_longer,
    device: Optional[str] = None,
    warmups: int = 10,
    iters: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Times CLAP audio encoder, linear head, and end-to-end forward."""
    import torch

    model.eval()
    with torch.no_grad():
        encoder_out = model.clap.get_audio_features(
            input_features=input_features, is_longer=is_longer
        )

        def encode():
            return model.clap.get_audio_features(
                input_features=input_features, is_longer=is_longer
            )

        def classify():
            return model.classifier(encoder_out)

        def end_to_end():
            return model(input_features, is_longer)

        return {
            "encoder": time_callable(encode, warmups, iters, device),
            "classifier": time_callable(classify, warmups, iters, device),
            "end_to_end": time_callable(end_to_end, warmups, iters, device),
        }


def measure_whisper_components(
    model,
    input_features,
    device: Optional[str] = None,
    warmups: int = 10,
    iters: int = 50,
) -> Dict[str, Dict[str, float]]:
    """Times Whisper encoder, linear head, and end-to-end forward."""
    import torch

    model.eval()
    with torch.no_grad():
        encoder_out = model.whisper.encoder(input_features)
        pooled = encoder_out.last_hidden_state.mean(dim=1)

        def encode():
            return model.whisper.encoder(input_features)

        def classify():
            return model.classifier(pooled)

        def end_to_end():
            return model(input_features)

        return {
            "encoder": time_callable(encode, warmups, iters, device),
            "classifier": time_callable(classify, warmups, iters, device),
            "end_to_end": time_callable(end_to_end, warmups, iters, device),
        }


def measure_sklearn_inference(
    model,
    X: np.ndarray,
    iters: int = 20,
) -> Dict[str, Dict[str, float]]:
    """Times an sklearn model's ``predict`` over the full ``X`` matrix.

    Reports per-window latency and windows/second. Note this measures the
    classifier only; the upstream encoder/feature extraction cost is not
    included (it is reported separately from the extraction run logs).
    """
    n = len(X)
    if n == 0:
        return {}

    def predict():
        return model.predict(X)

    stats = time_callable(predict, warmups=2, iters=iters, device=None)
    return {
        "inference": {
            "mean_ms": stats["mean_ms"],
            "calls_per_s": stats["calls_per_s"],
            "n_samples": n,
        }
    }


def measure_sklearn_classifier_latency_bs1(
    model,
    X: np.ndarray,
    n_sample: int = 100,
    iters: int = 5,
    warmups: int = 2,
) -> Dict[str, float]:
    """Times the classifier alone at batch size 1.

    The fine-tuning blocks publish their latency at batch size 1, so the
    traditional models are measured the same way to make the two families
    comparable. The batched ``Inference Latency`` column is kept as well.
    """
    rows = np.asarray(X)
    n = len(rows)
    if n == 0:
        return {}

    k = min(n_sample, n)
    indices = np.linspace(0, n - 1, num=k).astype(int)

    for _ in range(warmups):
        model.predict(rows[indices[0] : indices[0] + 1])

    per_window_ms = []
    for idx in indices:
        row = rows[idx : idx + 1]
        start = time.perf_counter()
        for _ in range(iters):
            model.predict(row)
        elapsed = time.perf_counter() - start
        per_window_ms.append((elapsed * 1000.0) / max(1, iters))

    mean_ms = float(np.mean(per_window_ms))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(per_window_ms)),
        "calls_per_s": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "n_samples": int(k),
    }


def sklearn_latency_columns(stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Flattens sklearn inference stats into per-window CSV columns."""
    if not stats:
        return {}
    inference = stats["inference"]
    n = max(1, int(inference.get("n_samples", 1)))
    return {
        "Inference Latency (ms/window)": round(float(inference["mean_ms"]) / n, 4),
        "Inference Throughput (windows/s)": round(
            float(inference["calls_per_s"]) * n, 3
        ),
    }


def measure_sklearn_model_latency(
    enabled: bool,
    model,
    X,
    dataset_config=None,
) -> Dict[str, float]:
    """Convenience wrapper: returns latency columns (or ``{}``) for the
    traditional models, guarded so a measurement failure never breaks a run.

    Besides the batched classifier latency this adds the batch-size-1
    classifier latency and the end-to-end latency (representation extraction
    + classifier), which is the number that is directly comparable with the
    end-to-end latency of the fine-tuning blocks.
    """
    if not enabled or X is None or len(X) == 0:
        return {}

    columns: Dict[str, float] = {}

    try:
        columns.update(sklearn_latency_columns(measure_sklearn_inference(model, X)))
    except Exception as e:
        print(f"[WARN] sklearn latency measurement failed: {e}")

    try:
        classifier = measure_sklearn_classifier_latency_bs1(model, X)
        columns["Classifier Latency (ms/window) (bs=1)"] = round(
            float(classifier["mean_ms"]), 4
        )
        columns["Classifier Throughput (windows/s) (bs=1)"] = round(
            float(classifier["calls_per_s"]), 3
        )
    except Exception as e:
        print(f"[WARN] sklearn bs=1 latency measurement failed: {e}")

    try:
        from scripts.evaluation.representation_latency import (
            representation_latency_columns,
        )

        columns.update(
            representation_latency_columns(
                getattr(dataset_config, "feature_type", None)
                if dataset_config is not None
                else None,
                dataset_config,
            )
        )
    except Exception as e:
        print(f"[WARN] representation latency measurement failed: {e}")

    representation_ms = columns.get("Feature Extraction Latency (ms/window) (bs=1)")
    classifier_ms = columns.get("Classifier Latency (ms/window) (bs=1)")
    if representation_ms is not None and classifier_ms is not None:
        end_to_end = float(representation_ms) + float(classifier_ms)
        columns["End To End Latency (ms/window) (bs=1)"] = round(end_to_end, 4)
        columns["End To End Throughput (windows/s) (bs=1)"] = round(
            1000.0 / end_to_end if end_to_end > 0 else 0.0, 3
        )

    return columns
