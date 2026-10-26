"""Measure end-to-end latency for an existing traditional-ML run.

The traditional-ML block now reports a batch-size-1 end-to-end latency
(representation extraction + classifier). Older runs predate those columns, and
re-fitting every classifier just to obtain two timings is wasteful. This script
reuses the fitted ``model.pkl`` files and the stored embeddings instead:

* the classifier latency is measured on the saved model,
* the representation latency is measured (or taken from the cache) with the
  same code path the benchmark uses,
* the results are written to a small CSV and, optionally, merged into the
  existing metrics outline so the heatmap can use them.

    python scripts/measure_traditional_ml_latency.py \
        --results-dir experiments/vocalset_non_overlapping/metrics/<run>/<experiment> \
        --update-outline
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from scripts.data.features import load_features_data
from scripts.evaluation.latency import measure_sklearn_classifier_latency_bs1
from scripts.evaluation.representation_latency import (
    DEFAULT_CACHE_PATH,
    representation_latency_columns,
)
from scripts.run_benchmark import load_experiment_config, resolve_dataset
from scripts.utils import load_pickle

DEFAULT_EXPERIMENT = (
    "experiments/vocalset_non_overlapping/"
    "vocalset_non_overlapping_full_experiment.py"
)
DEFAULT_RESULTS_DIR = (
    "experiments/vocalset_non_overlapping/metrics/"
    "vtc_vocalset_non_overlapping_silencetrimmed_allbutunfrozen/"
    "vtc_vocalset_non_overlapping"
)

FEATURE_BLOCK_TYPE = "features"

REPRESENTATION_COLUMN = "Feature Extraction Latency (ms/window) (bs=1)"
CLASSIFIER_COLUMN = "Classifier Latency (ms/window) (bs=1)"
END_TO_END_COLUMN = "End To End Latency (ms/window) (bs=1)"

OUTPUT_COLUMNS = [
    "Model",
    "Input Data",
    "Block",
    REPRESENTATION_COLUMN,
    "Representation Latency Source",
    CLASSIFIER_COLUMN,
    "Classifier Throughput (windows/s) (bs=1)",
    END_TO_END_COLUMN,
    "End To End Throughput (windows/s) (bs=1)",
]

# Directory names produced by earlier runs differ from the current config names.
REPRESENTATION_ALIASES = {
    "clap": "laion-clap",
    "laion-clap": "laion-clap",
    "laion_clap": "laion-clap",
    "laion-clap-htsat-fused": "laion-clap",
    "ms_clap": "ms-clap",
    "ms-clap": "ms-clap",
    "ms-clap-2023": "ms-clap",
}


def _representation_name(value: str) -> str:
    key = str(value).strip().lower()
    return REPRESENTATION_ALIASES.get(key, key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, default=Path(DEFAULT_EXPERIMENT))
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument(
        "--outline",
        type=Path,
        default=None,
        help="Metrics outline to update. Default: next to --results-dir.",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Where to write the latency table. Default: next to --results-dir.",
    )
    parser.add_argument("--n-windows", type=int, default=20)
    parser.add_argument("--classifier-samples", type=int, default=200)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument(
        "--no-update-outline",
        action="store_true",
        help="Only write the latency CSV, leave the outline untouched.",
    )
    return parser.parse_args()


def _find_feature_block(experiment):
    for block in experiment.blocks:
        if block.model_type == FEATURE_BLOCK_TYPE:
            return block
    raise ValueError(f"No '{FEATURE_BLOCK_TYPE}' block in the experiment")


def _find_block_dir(results_dir: Path) -> Path:
    candidates = sorted(results_dir.glob("trad_ml/block_*"))
    if not candidates:
        raise ValueError(f"No trad_ml/block_* directory under {results_dir}")
    return candidates[-1]


def _match_dataset_dir(block_dir: Path, dataset_name: str) -> Optional[Path]:
    wanted = _representation_name(dataset_name)
    for candidate in sorted(block_dir.iterdir()):
        if candidate.is_dir() and _representation_name(candidate.name) == wanted:
            return candidate
    return None


def _load_split(dataset_config, max_samples: int):
    data = load_features_data(dataset_config)
    X = data.X_val if getattr(data, "X_val", None) is not None and len(data.X_val) else data.X_train
    if X is None or len(X) == 0:
        return None
    return X[:max_samples]


def measure(args: argparse.Namespace) -> List[Dict[str, object]]:
    experiment = load_experiment_config(args.experiment)
    block = _find_feature_block(experiment)

    block_dir = _find_block_dir(args.results_dir)
    print(f"Experiment: {experiment.name}")
    print(f"Results:    {block_dir}")

    rows: List[Dict[str, object]] = []
    for dataset in block.datasets:
        dataset_dir = _match_dataset_dir(block_dir, dataset.name)
        if dataset_dir is None:
            print(f"[WARN] No results directory for '{dataset.name}', skipping")
            continue

        resolved = resolve_dataset(
            dataset,
            block.partition,
            block.filters,
            frame_duration=block.frame_duration,
            overlap=block.overlap,
        )

        print(f"\n=== {dataset.name} ({resolved.feature_type}) ===")
        representation = representation_latency_columns(
            resolved.feature_type,
            resolved,
            n_windows=args.n_windows,
            cache_path=args.cache,
        )
        if not representation:
            print("  [WARN] representation latency unavailable in this environment")

        X = _load_split(resolved, args.classifier_samples)
        if X is None:
            print("  [WARN] no samples available for classifier timing")
            continue

        representation_ms = representation.get(REPRESENTATION_COLUMN)

        for model_dir in sorted(dataset_dir.iterdir()):
            model_path = model_dir / "model.pkl"
            if not model_dir.is_dir() or not model_path.exists():
                continue

            model = load_pickle(model_path)
            try:
                classifier = measure_sklearn_classifier_latency_bs1(model, X)
            except Exception as e:
                print(f"  [WARN] {model_dir.name}: {type(e).__name__}: {e}")
                continue

            classifier_ms = float(classifier["mean_ms"])
            row: Dict[str, object] = {
                "Model": model_dir.name,
                "Input Data": dataset.name,
                "Block": FEATURE_BLOCK_TYPE,
                REPRESENTATION_COLUMN: representation_ms,
                "Representation Latency Source": representation.get(
                    "Representation Latency Source"
                ),
                CLASSIFIER_COLUMN: round(classifier_ms, 4),
                "Classifier Throughput (windows/s) (bs=1)": round(
                    float(classifier["calls_per_s"]), 3
                ),
            }

            if representation_ms is not None:
                end_to_end = float(representation_ms) + classifier_ms
                row[END_TO_END_COLUMN] = round(end_to_end, 4)
                row["End To End Throughput (windows/s) (bs=1)"] = round(
                    1000.0 / end_to_end if end_to_end > 0 else 0.0, 3
                )

            rows.append(row)
            print(
                f"  {model_dir.name:4s} classifier {classifier_ms:8.4f} ms"
                + (
                    f" | representation {float(representation_ms):8.4f} ms"
                    f" | end-to-end {row.get(END_TO_END_COLUMN, float('nan')):8.4f} ms"
                    if representation_ms is not None
                    else ""
                )
            )

    return rows


def update_outline(outline_path: Path, rows: List[Dict[str, object]]) -> None:
    if not outline_path.exists():
        print(f"[WARN] Outline not found: {outline_path}")
        return

    df = pd.read_csv(outline_path)
    if "Block" not in df.columns or "Input Data" not in df.columns:
        print("[WARN] Outline is missing 'Block'/'Input Data' columns, skipping")
        return

    latency_columns = [
        c for c in OUTPUT_COLUMNS if c not in ("Model", "Input Data", "Block")
    ]
    for column in latency_columns:
        if column not in df.columns:
            df[column] = None

    updated = 0
    for row in rows:
        mask = (
            (df["Block"] == FEATURE_BLOCK_TYPE)
            & (df["Model"] == row["Model"])
            & (df["Input Data"].map(_representation_name) == _representation_name(row["Input Data"]))
        )
        if not mask.any():
            print(
                f"[WARN] No outline row for {row['Model']} / {row['Input Data']}"
            )
            continue
        for column in latency_columns:
            df.loc[mask, column] = row.get(column)
        updated += int(mask.sum())

    backup_path = outline_path.with_suffix(outline_path.suffix + ".bak")
    shutil.copy2(outline_path, backup_path)
    df.to_csv(outline_path, index=False)
    print(f"\nUpdated {updated} outline row(s); backup at {backup_path}")


def main() -> None:
    args = parse_args()

    rows = measure(args)
    if not rows:
        print("No latency rows produced.")
        return

    table = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    out_csv = args.out_csv or args.results_dir.parent / "trad_ml_latency.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nLatency table written to {out_csv}")

    print("\n" + table.to_string(index=False))

    if not args.no_update_outline:
        outline = args.outline or (
            args.results_dir.parent
            / f"{args.results_dir.name}_full_metrics_outline.csv"
        )
        update_outline(outline, rows)


if __name__ == "__main__":
    main()
