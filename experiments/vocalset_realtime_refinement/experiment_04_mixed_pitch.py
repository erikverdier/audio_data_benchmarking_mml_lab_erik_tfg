
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
    "female1", "female3", "female4", "female5", "female6",
    "female7", "female9", "male1", "male11", "male2", "male4",
    "male6", "male7", "male8", "male9",
]

TEST_SINGERS = [
    "female2", "female8", "male3", "male5", "male10",
]

partition = PartitionConfig(
    name="vocalset_singer_independent",
    train_data_ids=TRAIN_SINGERS,
    test_data_ids=TEST_SINGERS,
    data_column_label="singer",
)

PMT_LABELS = [
    "vibrato",
    "straight",
    "trill",
    "trillo",
]

TMT_LABELS = [
    "belt",
    "breathy",
    "spoken",
    "inhaled",
    "vocal_fry",
]

# Experiment 04: mixed-gender pitch modulation (PMT)

experiment_04_mixed_pitch = ExperimentConfig(
    name="experiment_04_mixed_pitch",
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
                gender="mixed",
                include_labels=PMT_LABELS,  # Pitch
            ),
        ),
    ],
)
