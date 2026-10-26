from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from scripts.config.partition_config import PartitionConfig
from scripts.evaluation.metrics import (
    _get_report_f1,
    log_environment_snapshot,
    save_metrics_csv,
)
from scripts.registry import get_model, list_models
from scripts.utils import ensure_dir, set_seed, save_json
from scripts.validate_config import validate


# ---------------------------------------------------------------------------
# Config loaders
# ---------------------------------------------------------------------------

def load_config(path: str | Path, cls: type) -> Any:
    path = Path(path)
    if path.suffix == ".py":
        import importlib.util
        spec = importlib.util.spec_from_file_location("config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for attr_name in dir(mod):
            attr = getattr(mod, attr_name)
            if isinstance(attr, cls):
                return attr
        raise ValueError(f"No {cls.__name__} instance found in {path}")
    elif path.suffix in (".yml", ".yaml"):
        return cls.load_yaml(path)
    elif path.suffix == ".json":
        return cls.load_json(path)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")


def load_dataset_config(path: str | Path) -> DatasetConfig:
    return load_config(path, DatasetConfig)


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return load_config(path, BenchmarkConfig)


def load_partition_config(path: str | Path) -> PartitionConfig:
    return load_config(path, PartitionConfig)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    return load_config(path, ExperimentConfig)


# ---------------------------------------------------------------------------
# Resolution: combine dataset + partition + filters into a full DatasetConfig
# ---------------------------------------------------------------------------

def resolve_dataset(
    dataset: DatasetConfig,
    partition: PartitionConfig,
    filters: ExperimentFilters,
    frame_duration: Optional[float] = None,
    overlap: Optional[float] = None,
) -> DatasetConfig:
    merged = {
        "name": dataset.name,
        "data_type": dataset.data_type,
        "data_dir": dataset.data_dir,
        "feature_type": dataset.feature_type,
        "train_data_path": dataset.train_data_path,
        "val_data_path": dataset.val_data_path,
        "train_data_paths": dataset.train_data_paths,
        "val_data_paths": dataset.val_data_paths,
        "feature_column": dataset.feature_column,
        "label_column": dataset.label_column,
        "data_column_label": dataset.data_column_label,
        "parquet_file": dataset.parquet_file,
        "audio_dir": dataset.audio_dir,
        "sample_rate": dataset.sample_rate,
        "frame_duration": dataset.frame_duration,
        "overlap": dataset.overlap,
        "image_dir": dataset.image_dir,
        "image_size": dataset.image_size,
        "level_names": dataset.level_names,
        "label_level": dataset.label_level,
        "metrics_path": dataset.metrics_path,
        "pad_sequences": dataset.pad_sequences,
        "add_channel_dim": dataset.add_channel_dim,
        "tail_window_mode": dataset.tail_window_mode,
        "short_audio_handling": dataset.short_audio_handling,
        "enable_silence_trimming": dataset.enable_silence_trimming,
        "silence_threshold_db": dataset.silence_threshold_db,
        "silence_min_duration_ms": dataset.silence_min_duration_ms,
        "normalization_mode": dataset.normalization_mode,
    }
    _override_if_set(merged, "train_data_ids", partition.train_data_ids)
    _override_if_set(merged, "test_data_ids", partition.test_data_ids)
    _override_if_set(merged, "data_column_label", partition.data_column_label)
    _override_if_set(merged, "gender_split", filters.gender)
    _override_if_set(merged, "include_labels", filters.include_labels)
    _override_if_set(merged, "exclude_labels", filters.exclude_labels)
    _override_if_set(merged, "label_map", filters.label_map)
    _override_if_set(merged, "pad_sequences", filters.pad_sequences)
    _override_if_set(merged, "add_channel_dim", filters.add_channel_dim)
    _override_if_set(merged, "frame_duration", frame_duration)
    _override_if_set(merged, "overlap", overlap)
    return DatasetConfig(**merged)


def _override_if_set(target: dict, key: str, value, skip_value=None):
    if value is not None and value != skip_value:
        target[key] = value


# ---------------------------------------------------------------------------
# Single dataset + single benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    dataset_config: DatasetConfig | str | Path,
    benchmark_config: BenchmarkConfig | str | Path,
) -> Dict[str, Any]:
    if not isinstance(dataset_config, DatasetConfig):
        dataset_config = load_dataset_config(dataset_config)
    if not isinstance(benchmark_config, BenchmarkConfig):
        benchmark_config = load_benchmark_config(benchmark_config)

    print(f"Dataset:  {dataset_config.name} (type: {dataset_config.data_type})")
    print(f"Benchmark: {benchmark_config.name}")
    print(f"Models:   {', '.join(benchmark_config.models)}")
    print()

    set_seed(benchmark_config.seed)

    warnings = validate(dataset_config, benchmark_config)
    has_errors = False
    for level, msg in warnings:
        print(f"[{level.upper()}] {msg}")
        if level == "error":
            has_errors = True
    if has_errors:
        print("Configuration errors found. Aborting.")
        sys.exit(1)

    output_dir = ensure_dir(benchmark_config.output_dir)

    results = {}
    for model_name in benchmark_config.models:
        print(f"\n{'=' * 60}")
        print(f"Running model: {model_name}")
        print(f"{'=' * 60}")

        model_output_dir = ensure_dir(
            output_dir / f"{benchmark_config.name}_{model_name}"
        )

        model_info = get_model(model_name)
        result = model_info["run"](dataset_config, benchmark_config, str(model_output_dir))
        results[model_name] = result

        print(f"Completed: {model_name}")

    summary_path = output_dir / f"{benchmark_config.name}_summary.json"
    summary = {
        "dataset": dataset_config.name,
        "benchmark": benchmark_config.name,
        "models": list(results.keys()),
        "results": {
            name: {
                "val_balanced_accuracy": res.get("val_metrics", {}).get("balanced_accuracy"),
                "val_accuracy": res.get("val_metrics", {}).get("accuracy"),
            }
            for name, res in results.items()
        },
    }
    save_json(summary, summary_path)
    print(f"\nSummary saved to {summary_path}")

    return results


# ---------------------------------------------------------------------------
# Repeated-run aggregation
# ---------------------------------------------------------------------------

def _mean_std_metrics(metric_dicts: List[Dict]) -> Tuple[Dict, Dict]:
    """Average a list of compute_metrics()-style dicts into (mean, std).

    Only the metric fields that change across runs are averaged (accuracy,
    balanced_accuracy, F1, top-2, top-3, per-class accuracy). n_samples is
    constant per run so it is read from the first sample.
    """
    numeric_keys = ("accuracy", "balanced_accuracy", "top2_accuracy", "top3_accuracy")
    mean: Dict[str, Any] = {}
    std: Dict[str, Any] = {}

    for key in numeric_keys:
        values = [m.get(key) for m in metric_dicts if m.get(key) is not None]
        if values:
            mean[key] = float(np.mean(values))
            std[key] = float(np.std(values)) if len(values) > 1 else 0.0

    f1s = []
    for m in metric_dicts:
        f1 = _get_report_f1(m.get("classification_report", {}))
        if isinstance(f1, float):
            f1s.append(f1)
    if f1s:
        mean_f1 = float(np.mean(f1s))
        mean["classification_report"] = {"macro avg": {"f1-score": mean_f1}}
        std["f1"] = float(np.std(f1s)) if len(f1s) > 1 else 0.0

    per_class = {}
    for m in metric_dicts:
        for cls, acc in (m.get("per_class_accuracy") or {}).items():
            per_class.setdefault(cls, []).append(acc)
    if per_class:
        mean["per_class_accuracy"] = {}
        std["per_class_accuracy"] = {}
        mean["class_names"] = metric_dicts[0].get("class_names")
        for cls, accs in per_class.items():
            mean["per_class_accuracy"][cls] = float(np.mean(accs))
            std["per_class_accuracy"][cls] = float(np.std(accs)) if len(accs) > 1 else 0.0

    if metric_dicts:
        mean["n_samples"] = metric_dicts[0].get("n_samples")

    return mean, std


_METRIC_COLUMNS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced Accuracy",
    "f1": "F1",
    "top2_accuracy": "Top-2",
    "top3_accuracy": "Top-3",
}


def _std_extra(split: str, std_metrics: Dict) -> Dict[str, Any]:
    """Maps a std dict (keyed by metric) to extra CSV columns like
    'Val Accuracy Std' or 'Train Top-2 Std'."""
    extra = {}
    for key, col in _METRIC_COLUMNS.items():
        if key in std_metrics:
            extra[f"{split} {col} Std"] = round(float(std_metrics[key]), 4)
    for cls, acc in (std_metrics.get("per_class_accuracy") or {}).items():
        extra[f"{split} {cls} Accuracy Std"] = round(float(acc), 4)
    return extra


def _aggregate_repeated_run(
    run_results: List[Dict],
    dataset_config: DatasetConfig,
    block_metrics_path: Path,
    input_data: str,
) -> Dict[str, Any]:
    """Combines N repeated runs into a single mean (with ±std columns) CSV row."""
    n_runs = len(run_results)
    mean_train, std_train = _mean_std_metrics(
        [r["train_metrics"] for r in run_results]
    )
    mean_val, std_val = _mean_std_metrics([r["val_metrics"] for r in run_results])

    last = run_results[-1]
    extra = {
        "Epochs": round(float(np.mean([r["actual_epochs"] for r in run_results])), 2),
        "Max Epochs": last.get("max_epochs", ""),
        "Batch Size": last.get("batch_size", ""),
        "Train+Eval Time (s)": round(
            float(np.sum([r.get("train_time_s", 0.0) for r in run_results])), 4
        ),
        "Total Time (s)": round(
            float(np.sum([r.get("train_time_s", 0.0) for r in run_results])), 4
        ),
        "frame_duration": dataset_config.frame_duration,
        "overlap": dataset_config.overlap,
        "n_runs": n_runs,
        **last.get("resource_stats", {}),
    }
    for src_key, col in (
        ("feature_extraction_time_s", "Feature Extraction Time (s)"),
        ("training_time_s", "Encoder Training Time (s)"),
        ("eval_time_s", "Eval Inference Time (s)"),
    ):
        values = [r.get(src_key) for r in run_results if r.get(src_key) is not None]
        if values:
            extra[col] = round(float(np.sum(values)), 4)

    latency_keys = set()
    for r in run_results:
        latency_keys.update((r.get("latency") or {}).keys())
    for key in sorted(latency_keys):
        values = [
            r["latency"][key]
            for r in run_results
            if key in (r.get("latency") or {})
        ]
        if values:
            extra[key] = round(float(np.mean(values)), 4)

    extra.update(_std_extra("Train", std_train))
    extra.update(_std_extra("Val", std_val))

    save_metrics_csv(
        block_metrics_path,
        last["model_name"],
        input_data,
        mean_train,
        mean_val,
        extra=extra,
        complexity=last.get("complexity"),
    )

    return {
        "model_name": last["model_name"],
        "train_metrics": mean_train,
        "val_metrics": mean_val,
        "history": None,
        "n_runs": n_runs,
        "run_results": run_results,
    }


# ---------------------------------------------------------------------------
# Consolidated experiment outline
# ---------------------------------------------------------------------------

def _round_metric_value(value, decimals: int = 3):
    """Rounds a single metric value to `decimals` places, leaving
    non-numeric strings (e.g. '-', model names) untouched."""
    if isinstance(value, (int, float, np.integer, np.floating)):
        if isinstance(value, float) and np.isnan(value):
            return value
        return round(float(value), decimals)
    if isinstance(value, str):
        try:
            return round(float(value), decimals)
        except ValueError:
            return value
    return value


def _consolidate_metrics_table(
    metrics_paths: List[Tuple[str | Path, str]],
    output_path: str | Path,
    decimals: int = 3,
) -> Optional[pd.DataFrame]:
    """Merges all per-block metrics CSVs into a single rounded table.

    Each block (trad ML, frozen finetune) writes its own metrics.csv; the
    repeated full-finetune blocks add their mean row (plus std columns) to
    their shared metrics.csv. This collects every row across those files so
    the final outline has the deterministic single runs AND the N-run
    averaged full-finetune rows, all rounded to `decimals` places.

    Every entry is a (metrics_path, block_label) pair; the block's
    ``model_type`` is added as a "Block" column so rows from different
    blocks (e.g. arousal vs valence) stay distinguishable in the outline.
    """
    import pandas as pd

    frames = []
    seen = set()
    for p, block_label in metrics_paths:
        p = Path(p)
        if not p.exists() or p.resolve() in seen:
            continue
        seen.add(p.resolve())
        df = pd.read_csv(p)
        for col in df.columns:
            df[col] = df[col].apply(
                lambda v: _round_metric_value(v, decimals)
            )
        df["Block"] = block_label
        frames.append(df)

    if not frames:
        print("  No metrics CSVs found to consolidate.")
        return None

    merged = pd.concat(frames, ignore_index=True, sort=False)
    preferred = [c for c in ("Model", "Input Data", "Block") if c in merged.columns]
    other = [c for c in merged.columns if c not in preferred]
    merged = merged[preferred + other]
    merged.to_csv(output_path, index=False)
    print(f"\nExperiment metrics outline saved to {output_path}")
    return merged


# ---------------------------------------------------------------------------
# Experiment config with blocks
# ---------------------------------------------------------------------------

def run_experiment(experiment_config_path: str | Path) -> Dict[str, Any]:
    experiment = load_experiment_config(experiment_config_path)

    from scripts.data import dataset_cache

    cache_datasets = bool(getattr(experiment, "cache_datasets", False))
    dataset_cache.clear()
    dataset_cache.set_enabled(cache_datasets)
    if cache_datasets:
        print(
            "Dataset caching enabled: feature/embedding datasets are held "
            "during a block and released afterwards; raw-audio datasets are "
            "retained across blocks and repeats."
        )

    experiment_out = ensure_dir(Path(experiment.output_dir) / experiment.name)
    environment_path = experiment_out / "environment.txt"
    log_environment_snapshot(environment_path)
    print(f"Environment snapshot saved to {environment_path}")

    print(f"\n{'=' * 60}")
    print(f"Experiment: {experiment.name}")
    print(f"Blocks:     {len(experiment.blocks)}")
    print(f"{'=' * 60}\n")

    all_results = {}
    metrics_paths: List[Tuple[Path, str]] = []

    for block_idx, block in enumerate(experiment.blocks):
        print(f"\n{'─' * 60}")
        print(f"Block {block_idx + 1}: {block.model_type}")
        print(f"  Benchmark: {block.benchmark_config.name}")
        print(f"  Models:    {', '.join(block.benchmark_config.models)}")
        print(f"  Partition: {block.partition.name}")
        print(f"  Datasets:  {', '.join(d.name for d in block.datasets)}")
        if block.filters.gender:
            print(f"  Gender:    {block.filters.gender}")
        if block.filters.include_labels:
            print(f"  Labels:    {len(block.filters.include_labels)} classes")
        print(f"{'─' * 60}\n")

        set_seed(block.benchmark_config.seed)

        block_out = ensure_dir(
            Path(experiment.output_dir)
            / experiment.name
            / block.benchmark_config.name
            / f"block_{block_idx + 1}"
        )
        block_metrics_path = Path(block.metrics_path) if block.metrics_path else block_out / "metrics.csv"
        metrics_paths.append((block_metrics_path, block.model_type))

        for dataset in block.datasets:
            resolved = resolve_dataset(
                dataset, block.partition, block.filters,
                frame_duration=block.frame_duration,
                overlap=block.overlap,
            )

            print(f"\n  Dataset: {dataset.name}")

            warnings = validate(resolved, block.benchmark_config)
            has_errors = False
            for level, msg in warnings:
                print(f"  [{level}] {msg}")
                if level == "error":
                    has_errors = True
            if has_errors:
                print("  Skipping due to errors.")
                continue

            n_repeats = max(1, int(getattr(block, "n_repeats", 1) or 1))

            for model_name in block.benchmark_config.models:
                print(f"  Model: {model_name}")

                model_info = get_model(model_name)
                model_out = ensure_dir(block_out / dataset.name / model_name)

                if n_repeats == 1:
                    resolved.metrics_path = str(block_metrics_path)
                    result = model_info["run"](
                        resolved, block.benchmark_config, str(model_out)
                    )
                else:
                    run_results = []
                    base_seed = block.benchmark_config.seed
                    for run_idx in range(1, n_repeats + 1):
                        print(f"    Repetition {run_idx}/{n_repeats}")
                        set_seed(base_seed + run_idx)
                        run_out = ensure_dir(model_out / f"run_{run_idx}")
                        resolved.metrics_path = str(run_out / "metrics.csv")
                        run_result = model_info["run"](
                            resolved, block.benchmark_config, str(run_out)
                        )
                        run_results.append(run_result)

                    result = _aggregate_repeated_run(
                        run_results,
                        resolved,
                        block_metrics_path,
                        dataset.name,
                    )

                key = f"{block.benchmark_config.name}/{dataset.name}/{model_name}"
                all_results[key] = result

                print(f"  Completed: {key}")

        # Feature/embedding datasets are only used within their own block, so
        # release them once the block is done (this is the traditional-ML case
        # the experiment wants to keep off the heap). Raw-audio datasets are
        # retained for the fine-tuning blocks that reuse them.
        if cache_datasets:
            released = dataset_cache.release_prefix("feature")
            if released:
                print(f"  Released {released} cached feature dataset(s).")
        dataset_cache.collect_garbage()

    summary_path = Path(experiment.output_dir) / f"{experiment.name}_summary.json"
    summary = {
        "experiment": experiment.name,
        "results": {
            key: {
                "val_balanced_accuracy": res.get("val_metrics", {}).get("balanced_accuracy"),
                "val_accuracy": res.get("val_metrics", {}).get("accuracy"),
            }
            for key, res in all_results.items()
        },
    }
    save_json(summary, summary_path)
    print(f"\nSummary saved to {summary_path}")

    outline_path = Path(experiment.output_dir) / f"{experiment.name}_full_metrics_outline.csv"
    _consolidate_metrics_table(metrics_paths, outline_path)

    if cache_datasets:
        cached = dataset_cache.clear()
        print(f"Cleared {cached} cached dataset(s).")
    dataset_cache.set_enabled(False)
    dataset_cache.collect_garbage()

    return all_results
