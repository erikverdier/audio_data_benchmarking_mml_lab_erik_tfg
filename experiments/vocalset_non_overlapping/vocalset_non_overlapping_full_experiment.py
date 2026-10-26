from scripts.config.partition_config import PartitionConfig
from scripts.config.experiment_config import (
    ExperimentConfig,
    ExperimentBlock,
    ExperimentFilters,
)
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_experiments_constants import (
    VOCALSET_10,
    TRAIN_SINGERS,
    VAL_SINGERS,
)
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_data_featureset_embedding_configs import (
    make_opensmile_embedding_config,
    make_vggish_embedding_config,
    make_whisper_embedding_config,
    make_laion_clap_embedding_config,
    make_ms_clap_embedding_config,
    make_mfcc_config,
)
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_raw_audio_data_config import (
    make_whisper_audio_config,
    make_clap_audio_config,
)
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_benchmark_configs import (
    trad_ml_benchmark,
    whisper_finetuning_benchmark,
    clap_finetuning_benchmark,
    full_whisper_finetuning_benchmark,
    full_clap_finetuning_benchmark,
)

# --- Singer-independent partition (shared across all blocks) ---

partition = PartitionConfig(
    name="vocalset_singer_independent",
    train_data_ids=TRAIN_SINGERS,
    test_data_ids=VAL_SINGERS,
    data_column_label="singer",
)

# --- Datasets ---

opensmile = make_opensmile_embedding_config()
vggish = make_vggish_embedding_config()
whisper_emb = make_whisper_embedding_config()
laion_clap_emb = make_laion_clap_embedding_config()
ms_clap_emb = make_ms_clap_embedding_config()
mfccs = make_mfcc_config()

feature_datasets = [
    opensmile,
    vggish,
    whisper_emb,
    laion_clap_emb,
    ms_clap_emb,
    mfccs,
]

whisper_audio = make_whisper_audio_config()
clap_audio = make_clap_audio_config()

FINETUNING_METRICS_PATH = (
    "experiments/vocalset_non_overlapping/metrics/vtc_vocalset_non_overlapping_silencetrimmed/finetuning_metrics/metrics.csv"
)

OUTPUT_DIR = "experiments/vocalset_non_overlapping/metrics/vtc_vocalset_non_overlapping_silencetrimmed/"

# --- Experiment ---
# Non-overlapping 3 s windows, with silence trimming, zero-padded tails.
vtc_non_overlap_silence_trim_experiment = ExperimentConfig(
    name="vtc_vocalset_non_overlapping",
    output_dir=OUTPUT_DIR,
    cache_datasets=True,
    blocks=[
        # 1. Traditional ML over all representations, VOCALSET_10, mixed gender.
        ExperimentBlock(
            model_type="features",
            benchmark_config=trad_ml_benchmark,
            datasets=feature_datasets,
            partition=partition,
            frame_duration=3.0,
            overlap=0.0,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
        # 2. CLAP frozen encoder, VOCALSET_10, mixed gender.
        ExperimentBlock(
            model_type="clap_frozen_encoder",
            benchmark_config=clap_finetuning_benchmark,
            datasets=[clap_audio],
            partition=partition,
            frame_duration=3.0,
            overlap=0.0,
            n_repeats=10,
            metrics_path=FINETUNING_METRICS_PATH,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        ),
        # 3. Whisper frozen encoder, VOCALSET_10, mixed gender.
        ExperimentBlock(
            model_type="whisper_frozen_encoder",
            benchmark_config=whisper_finetuning_benchmark,
            datasets=[whisper_audio],
            partition=partition,
            frame_duration=3.0,
            overlap=0.0,
            n_repeats=10,
            metrics_path=FINETUNING_METRICS_PATH,
            filters=ExperimentFilters(
                gender="mixed",
                include_labels=VOCALSET_10,
            ),
        )
        # # 4. CLAP full fine-tuning, VOCALSET_10, mixed gender, 10 repeats.
        # ExperimentBlock(
        #     model_type="clap_full_fine_tuning",
        #     benchmark_config=full_clap_finetuning_benchmark,
        #     datasets=[clap_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.0,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     n_repeats=10,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
        # # 5. Whisper full fine-tuning, VOCALSET_10, mixed gender, 10 repeats.
        # ExperimentBlock(
        #     model_type="whisper_full_fine_tuning",
        #     benchmark_config=full_whisper_finetuning_benchmark,
        #     datasets=[whisper_audio],
        #     partition=partition,
        #     frame_duration=3.0,
        #     overlap=0.0,
        #     metrics_path=FINETUNING_METRICS_PATH,
        #     n_repeats=10,
        #     filters=ExperimentFilters(
        #         gender="mixed",
        #         include_labels=VOCALSET_10,
        #     ),
        # ),
    ],
)
