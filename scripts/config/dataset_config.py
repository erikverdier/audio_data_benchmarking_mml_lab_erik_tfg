'''
    Each dataset type will need to be instantianted with its own dataset config.
'''

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Tuple


@dataclass
class DatasetConfig:
    name: str
    data_type: str  # "features", "raw_audio", "images"

    data_dir: Optional[str] = None

    feature_type: Optional[str] = None
    train_data_path: Optional[str] = None
    val_data_path: Optional[str] = None
    train_data_paths: Optional[Dict[str, str]] = None
    val_data_paths: Optional[Dict[str, str]] = None

    feature_column: str = "Features"
    label_column: str = "Class"
    data_column_label: Optional[str] = None

    parquet_file: Optional[str] = None

    audio_dir: Optional[str] = None
    sample_rate: int = 16000
    frame_duration: float = 3.0
    overlap: float = 0.25

    # Raw-audio framing options (defaults preserve the original behaviour).
    # "shift": slide the final partial window back to the end of the audio
    #   (original behaviour). "pad": emit the trailing partial window as an
    #   extra zero-padded frame, matching the embedding-extraction repo.
    tail_window_mode: str = "shift"
    short_audio_handling: str = "pad"
    enable_silence_trimming: bool = False
    silence_threshold_db: float = -40.0
    silence_min_duration_ms: float = 250.0
    normalization_mode: str = "none"

    image_dir: Optional[str] = None
    image_size: Tuple[int, int] = (224, 224)

    level_names: Optional[List[str]] = None
    label_level: Optional[str] = None
    label_map: Optional[Dict[str, int]] = None
    include_labels: Optional[List[str]] = None
    exclude_labels: Optional[List[str]] = None

    gender_split: Optional[str] = None
    train_data_ids: Optional[List[str]] = None
    test_data_ids: Optional[List[str]] = None
    metrics_path: Optional[str] = None

    pad_sequences: bool = False
    max_sequence_length: Optional[int] = None
    add_channel_dim: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> DatasetConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> DatasetConfig:
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> DatasetConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))
