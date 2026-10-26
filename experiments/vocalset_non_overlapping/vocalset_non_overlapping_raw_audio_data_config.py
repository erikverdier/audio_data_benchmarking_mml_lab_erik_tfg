from scripts.config.dataset_config import DatasetConfig
from experiments.vocalset_non_overlapping.vocalset_non_overlapping_experiments_constants import (
    RAW_AUDIO_DATA_DIR,
)

# All blocks in this experiment use non-overlapping 3 s windows with
# silence trimming and zero-padded tails, matching the extraction pipeline.
_FRAMING = dict(
    frame_duration=3.0,
    overlap=0.0,
    tail_window_mode="pad",
    short_audio_handling="pad",
    enable_silence_trimming=True,
    normalization_mode="none",
)


def make_whisper_audio_config(audio_dir=RAW_AUDIO_DATA_DIR):
    return DatasetConfig(
        name="whisper_audio",
        data_type="raw_audio",
        audio_dir=audio_dir,
        sample_rate=16000,
        level_names=[
            "singer",
            "exercise",
            "technique"
        ],
        label_level="technique",
        **_FRAMING,
    )


def make_clap_audio_config(audio_dir=RAW_AUDIO_DATA_DIR):
    return DatasetConfig(
        name="clap_audio",
        data_type="raw_audio",
        audio_dir=audio_dir,
        sample_rate=48000,
        level_names=["singer", "exercise", "technique"],
        label_level="technique",
        **_FRAMING,
    )
