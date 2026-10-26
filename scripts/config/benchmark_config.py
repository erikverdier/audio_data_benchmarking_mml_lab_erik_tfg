'''
    Each benchmark type (audio input, feature set input, image input) will need to be instantianted with its own benchmark config.
'''
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List, Any


@dataclass
class BenchmarkConfig:
    name: str
    models: List[str]
    model_params: Optional[Dict[str, Dict[str, Any]]] = None
    output_dir: str = "./results"
    seed: int = 42

    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 10
    learning_rate: float = 0.001

    train_split: float = 1.0
    val_split: float = 1.0

    # When True the model's inference latency is measured and reported.
    # Defaults False so existing experiments are unaffected.
    measure_latency: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> BenchmarkConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save_yaml(self, path: str | Path) -> None:
        import yaml
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load_yaml(cls, path: str | Path) -> BenchmarkConfig:
        import yaml
        with open(path) as f:
            return cls.from_dict(yaml.safe_load(f))

    def save_json(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> BenchmarkConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def get_model_params(self, model_name: str) -> dict:
        if self.model_params and model_name in self.model_params:
            return self.model_params[model_name]
        return {}
