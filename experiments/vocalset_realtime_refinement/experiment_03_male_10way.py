
from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)

from configs.benchmarking_500ms_50overlap.my_dataset_config import opensmile
from configs.benchmarking_500ms_50overlap.my_benchmark_config import trad_ml


# 10/17 techniques

VOCALSET_10 = [
    "vibrato",
    "straight",
    "belt",
    "breathy",
    "lip_trill",
    "inhaled",
    "spoken",
    "trill",
    "trillo",
    "vocal_fry",
]


# Singer-independent train-test partition

TRAIN_SINGERS = [
    "male1", "male11", "male2", "male4",
    "male6", "male7", "male8", "male9",
]

TEST_SINGERS = [
    "male3", "male5", "male10",
]

partition = PartitionConfig(
    name="vocalset_singer_independent",
    train_data_ids=TRAIN_SINGERS,
    test_data_ids=TEST_SINGERS,
    data_column_label="singer",
)


# Experiment 03: male-only, 10 techniques

experiment_03_male_10way = ExperimentConfig(
    name="experiment_03_male_10way",
    output_dir=(
        "/Users/erikverdiermontero/Downloads/"
        "TFG_benchmarking/results"
    ),
    blocks=[
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml,
            datasets=[opensmile],
            partition=partition,
            frame_duration=0.5,
            overlap=0.5,
            filters=ExperimentFilters(
                gender="male",             # Only male
                include_labels=VOCALSET_10,
            ),
        ),
    ],
)
