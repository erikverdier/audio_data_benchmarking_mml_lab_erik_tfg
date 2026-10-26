from scripts.config.dataset_config import DatasetConfig
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_experiments_constants import (
    EMBEDDINGS_DATA_DIR,
)


def make_data_config_for_vtc_embeddings(
    name="opensmile",
    feature_type="opensmile-compare-2016",
    data_dir=EMBEDDINGS_DATA_DIR,
    parquet_file="vocalset_dataset_opensmile-compare-2016_3.0s.parquet",
    data_column_label="singer",
):
    # The framing fields describe how the stored embeddings were produced
    # (3 s, non-overlapping, silence trimmed, zero-padded tail). They are not
    # used for loading the embeddings, but the end-to-end latency
    # measurement rebuilds real windows with them so the representation cost
    # is measured on the very windows the embeddings were built from.
    return DatasetConfig(
        name=name,
        data_type="features",
        feature_type=feature_type,
        feature_column="embedding",
        label_column="vocal_technique",
        data_dir=data_dir,
        parquet_file=parquet_file,
        data_column_label=data_column_label,
        sample_rate=16000,
        frame_duration=3.0,
        overlap=0.0,
        enable_silence_trimming=True,
        silence_threshold_db=-40.0,
        normalization_mode="none",
        short_audio_handling="pad",
        tail_window_mode="pad",
    )


def make_opensmile_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="opensmile",
        feature_type="opensmile-compare-2016",
        parquet_file="vocalset_dataset_opensmile-compare-2016_3.0s.parquet",
    )


def make_vggish_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="vggish",
        feature_type="vggish",
        parquet_file="vocalset_dataset_vggish_3.0s.parquet",
    )


def make_whisper_embedding_config():
    return make_data_config_for_vtc_embeddings(
        name="whisper",
        feature_type="whisper-base",
        parquet_file="vocalset_dataset_whisper_whisper-base_3.0s.parquet",
    )


def make_laion_clap_embedding_config():
    """LAION-CLAP (HTSAT-fused), 512-d, 48 kHz."""
    return make_data_config_for_vtc_embeddings(
        name="laion-clap",
        feature_type="laion-clap-htsat-fused",
        parquet_file="vocalset_dataset_laion-clap-htsat-fused_3.0s.parquet",
    )


def make_ms_clap_embedding_config():
    """Microsoft CLAP 2023, 1024-d, 44.1 kHz."""
    return make_data_config_for_vtc_embeddings(
        name="ms-clap",
        feature_type="ms-clap-2023",
        parquet_file="vocalset_dataset_ms-clap-2023_3.0s.parquet",
    )


def make_mfcc_config():
    return make_data_config_for_vtc_embeddings(
        name="mfcc",
        feature_type="mfcc",
        parquet_file="vocalset_dataset_mfcc-13d_3.0s.parquet",
    )
