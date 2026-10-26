"""Frozen-encoder-only rerun of the VocalSet non-overlapping experiment.

Only the frozen CLAP and frozen Whisper blocks are executed, so the encoder
heads are retrained with the current learning rate while the traditional-ML
and full-fine-tuning results are reused from the previous run.
"""

import experiments.vocalset_non_overlapping.vocalset_non_overlapping_full_experiment as _full
from scripts.config.experiment_config import ExperimentConfig

_FROZEN_BLOCK_TYPES = ("clap_frozen_encoder", "whisper_frozen_encoder")

vtc_non_overlap_frozen_only_experiment = ExperimentConfig(
    name=_full.vtc_non_overlap_silence_trim_experiment.name,
    output_dir=_full.OUTPUT_DIR,
    cache_datasets=True,
    blocks=[
        block
        for block in _full.vtc_non_overlap_silence_trim_experiment.blocks
        if block.model_type in _FROZEN_BLOCK_TYPES
    ],
)
