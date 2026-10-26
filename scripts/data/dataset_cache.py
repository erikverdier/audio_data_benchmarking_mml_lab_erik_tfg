"""In-process cache for prepared datasets.

Preparing a dataset is expensive: feature datasets parse large parquet files
and concatenate embeddings, while raw-audio datasets load/resample every
recording and run the model processor over every window. When several models
(or repeated runs) use the same dataset, caching it avoids rebuilding it and
keeps peak memory bounded because only the datasets that are actually reused
are held.

Caching is disabled by default (``set_enabled``), so existing experiments are
unaffected. When enabled, callers build datasets through ``get_or_build`` and
the experiment driver can evict groups of entries with ``release_prefix``
(e.g. drop the feature/embedding datasets once the traditional-ML block is
finished, while retaining the raw-audio datasets for the fine-tuning blocks).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

_CACHE: Dict[str, Any] = {}
_ENABLED = False


def set_enabled(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = enabled


def is_enabled() -> bool:
    return _ENABLED


def make_key(*parts: Any) -> str:
    return "|".join(str(p) for p in parts)


def feature_key(config) -> str:
    """Stable cache key for a feature/embedding dataset."""
    return make_key(
        "feature",
        config.name,
        config.feature_type,
        config.data_dir,
        config.parquet_file,
        config.train_data_path,
        config.val_data_path,
        repr(config.train_data_paths),
        repr(config.val_data_paths),
        config.feature_column,
        config.label_column,
        config.data_column_label,
        repr(config.train_data_ids),
        repr(config.test_data_ids),
        repr(config.include_labels),
        repr(config.exclude_labels),
        repr(config.label_map),
        config.pad_sequences,
        config.add_channel_dim,
    )


def audio_key(dataset_config, params: dict, stage: str) -> str:
    """Stable cache key for a raw-audio window dataset (train or val)."""
    return make_key(
        "audio",
        stage,
        dataset_config.name,
        dataset_config.audio_dir or dataset_config.data_dir,
        params.get("model_name"),
        params.get("sample_rate"),
        params.get("frame_duration"),
        params.get("overlap"),
        getattr(dataset_config, "tail_window_mode", None),
        getattr(dataset_config, "short_audio_handling", None),
        getattr(dataset_config, "enable_silence_trimming", None),
        getattr(dataset_config, "normalization_mode", None),
        repr(dataset_config.train_data_ids),
        repr(dataset_config.test_data_ids),
        repr(dataset_config.include_labels),
        repr(dataset_config.exclude_labels),
        repr(dataset_config.gender_split),
    )


def get_or_build(key: str, builder: Callable[[], Any]) -> Any:
    """Return the cached value for ``key`` or build and store it.

    When caching is disabled the builder is always invoked and nothing is
    retained.
    """
    if not _ENABLED:
        return builder()
    if key not in _CACHE:
        _CACHE[key] = builder()
    return _CACHE[key]


def cached_keys() -> List[str]:
    return sorted(_CACHE)


def release_prefix(prefix: str) -> int:
    """Drop every cached entry whose key starts with ``prefix``.

    Returns the number of entries released.
    """
    keys = [k for k in _CACHE if k.startswith(prefix)]
    for k in keys:
        _CACHE.pop(k, None)
    return len(keys)


def clear() -> int:
    """Drop the whole cache. Returns the number of entries released."""
    n = len(_CACHE)
    _CACHE.clear()
    return n


def collect_garbage() -> None:
    """Free Python objects and empty the CUDA caching allocator."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
