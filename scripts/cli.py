from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio-benchmark",
        description="Audio data benchmarking tool - run ML models on audio features, raw audio, or spectrograms",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a benchmark experiment")
    run_parser.add_argument(
        "--experiment", "-e",
        help="Path to experiment config file (.py)",
    )
    run_parser.add_argument(
        "--feature-selection-experiment", "-f",
        help="Path to a feature-selection experiment config file (.py, .yml, .yaml, .json)",
    )
    run_parser.add_argument(
        "--dataset-config", "-d",
        help="Path to dataset config file (.py, .yml, .yaml, .json)",
    )
    run_parser.add_argument(
        "--benchmark-config", "-b",
        help="Path to benchmark config file (.py, .yml, .yaml, .json)",
    )

    list_parser = subparsers.add_parser("list-models", help="List all registered models")
    list_parser.add_argument(
        "--data-type", "-t",
        choices=["features", "raw_audio", "images"],
        help="Filter models by supported data type",
    )

    validate_parser = subparsers.add_parser("validate", help="Validate config files without running")
    validate_parser.add_argument(
        "--dataset-config", "-d",
        help="Path to dataset config file",
    )
    validate_parser.add_argument(
        "--benchmark-config", "-b",
        help="Path to benchmark config file",
    )
    validate_parser.add_argument(
        "--experiment", "-e",
        help="Path to experiment config file (.py)",
    )

    return parser


def main(argv: list = None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _run_command(args)
    elif args.command == "list-models":
        _list_models_command(args)
    elif args.command == "validate":
        _validate_command(args)


def _run_command(args):
    if args.feature_selection_experiment:
        from scripts.run_feature_selection import run_feature_selection_experiment
        run_feature_selection_experiment(args.feature_selection_experiment)
    elif args.experiment:
        from scripts.run_benchmark import run_experiment
        run_experiment(args.experiment)
    elif args.dataset_config and args.benchmark_config:
        from scripts.run_benchmark import run_benchmark
        run_benchmark(
            dataset_config=args.dataset_config,   # MODIFIED: changed from args.dataset_config to args.dataset_config
            benchmark_config=args.benchmark_config, # MODIFIED: changed from args.benchmark_config to args.benchmark_config
        )
    else:
        print("Error: specify --feature-selection-experiment, --experiment, OR --dataset-config + --benchmark-config")
        sys.exit(1)


def _list_models_command(args):
    from scripts.registry import list_models
    models = list_models()
    if args.data_type:
        models = [m for m in models if args.data_type in m["supported_data_types"]]

    if not models:
        print("No models found.")
        return

    print(f"{'Model Name':<20} {'Data Types':<30} Description")
    print("-" * 80)
    for m in models:
        types_str = ", ".join(m["supported_data_types"])
        desc = m.get("description", "")
        print(f"{m['name']:<20} {types_str:<30} {desc}")


def _validate_command(args):
    if args.experiment:
        from scripts.run_benchmark import load_experiment_config
        try:
            exp = load_experiment_config(args.experiment)
            print(f"Experiment '{exp.name}' loaded successfully.")
            print(f"  Blocks: {len(exp.blocks)}")
            for i, block in enumerate(exp.blocks):
                print(f"  Block {i+1}: {block.model_type} | "
                      f"models={block.benchmark_config.models} | "
                      f"datasets={[d.name for d in block.datasets]}")
        except Exception as e:
            print(f"Error loading experiment config: {e}")
            sys.exit(1)
        return

    if not (args.dataset_config and args.benchmark_config):
        print("Error: specify --experiment OR --dataset-config + --benchmark-config")
        sys.exit(1)

    from scripts.run_benchmark import load_dataset_config, load_benchmark_config
    from scripts.validate_config import validate

    try:
        dataset_config = load_dataset_config(args.dataset_config)
        bench_config = load_benchmark_config(args.benchmark_config)
    except Exception as e:
        print(f"Error loading configs: {e}")
        sys.exit(1)

    warnings = validate(dataset_config, bench_config)
    if not warnings:
        print("Configuration is valid.")
    else:
        has_errors = False
        for level, msg in warnings:
            print(f"[{level.upper()}] {msg}")
            if level == "error":
                has_errors = True
        if has_errors:
            sys.exit(1)


if __name__ == "__main__":
    main()
