"""Measure the per-window representation latency of the feature sets.

The traditional-ML block reports an end-to-end latency (representation
extraction + classifier). The representation part is measured in-process by
``scripts.evaluation.representation_latency`` while the benchmark runs, but a
representation whose dependencies are missing in the current environment
(e.g. VGGish, which needs TensorFlow) cannot be built there.

This script measures those representations and stores the result in the
latency cache, so the benchmark can pick the values up:

    python scripts/measure_representation_latency.py \
        --parquet /path/to/vocalset_dataset_vggish_3.0s.parquet \
        --feature-type vggish

Run it in an environment that has the required dependency (the cache is
environment specific because the numbers are device specific).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.representation_latency import (
    DEFAULT_CACHE_PATH,
    build_extractor,
    canonical_name,
    measure_extractor_latency,
    sample_windows,
    update_cache,
)

DEFAULT_FEATURE_TYPES = [
    "mfcc",
    "opensmile-compare-2016",
    "vggish",
    "ms-clap-2023",
    "laion-clap-htsat-fused",
    "whisper-base",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        required=True,
        help="Embedding parquet that lists the source audio files.",
    )
    parser.add_argument(
        "--feature-type",
        action="append",
        dest="feature_types",
        help="Feature type to measure (repeatable). Default: all of them.",
    )
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--singer-column", default="singer")
    parser.add_argument(
        "--no-trimming",
        action="store_true",
        help="Measure on untrimmed windows.",
    )
    return parser.parse_args()


def build_dataset_config(args: argparse.Namespace, feature_type: str) -> DatasetConfig:
    parquet = Path(args.parquet)
    return DatasetConfig(
        name=parquet.stem,
        data_type="features",
        feature_type=feature_type,
        feature_column="embedding",
        label_column="vocal_technique",
        data_dir=str(parquet.parent),
        parquet_file=parquet.name,
        data_column_label=args.singer_column,
        sample_rate=16000,
        frame_duration=3.0,
        overlap=0.0,
        enable_silence_trimming=not args.no_trimming,
        silence_threshold_db=-40.0,
        normalization_mode="none",
        short_audio_handling="pad",
        tail_window_mode="pad",
    )


def main() -> None:
    args = parse_args()
    feature_types = args.feature_types or DEFAULT_FEATURE_TYPES

    results = {}
    for feature_type in feature_types:
        name = canonical_name(feature_type)
        print(f"\n=== {feature_type} ({name}) ===")
        try:
            extractor = build_extractor(name)
            dataset_config = build_dataset_config(args, feature_type)
            windows = sample_windows(
                dataset_config,
                feature_type,
                n_windows=args.n_windows,
            )
            stats = measure_extractor_latency(
                extractor,
                windows,
                warmups=args.warmups,
                iters=args.iters,
            )
            entry = {
                "mean_ms": round(stats["mean_ms"], 4),
                "std_ms": round(stats["std_ms"], 4),
                "calls_per_s": round(stats["calls_per_s"], 3),
                "n_windows": stats["n_windows"],
                "source": "live",
                "feature_type": feature_type,
                "device": extractor.device or "cpu",
            }
            update_cache(name, entry, args.cache)
            results[name] = entry
            print(
                f"  {stats['mean_ms']:.3f} ms/window "
                f"({stats['calls_per_s']:.1f} windows/s, "
                f"{stats['n_windows']} windows)"
            )
        except Exception as e:
            print(f"  [WARN] {type(e).__name__}: {e}")

    print("\n" + json.dumps(results, indent=2, sort_keys=True))
    print(f"\nCache written to {args.cache}")


if __name__ == "__main__":
    main()
