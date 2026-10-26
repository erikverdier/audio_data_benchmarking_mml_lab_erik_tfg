"""Frozen-CLAP-only rerun of the VocalSet non-overlapping experiment.

The frozen Whisper head was already trained with the standard learning rate and
has no BatchNorm drift issue, so only the frozen CLAP block is rerun (with the
batch-norm fix and the corrected head learning rate).
"""

import experiments.vocalset_non_overlapping.vocalset_non_overlapping_full_experiment as _full
from scripts.config.experiment_config import ExperimentConfig

_FROZEN_CLAP_BLOCK = "clap_frozen_encoder"

vtc_non_overlap_frozen_clap_only_experiment = ExperimentConfig(
    name=_full.vtc_non_overlap_silence_trim_experiment.name,
    output_dir=_full.OUTPUT_DIR,
    cache_datasets=True,
    blocks=[
        block
        for block in _full.vtc_non_overlap_silence_trim_experiment.blocks
        if block.model_type == _FROZEN_CLAP_BLOCK
    ],
)
