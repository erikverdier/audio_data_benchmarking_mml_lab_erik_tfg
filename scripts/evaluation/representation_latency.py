"""Per-window representation latency for the traditional-ML pipeline.

The traditional-ML block consumes embeddings that were extracted offline, so
the latency it publishes only covers the classifier. The fine-tuning blocks
publish an end-to-end latency (encoder + head) instead, so the cost of
producing the representation has to be added to the traditional-ML numbers
before the two families can be compared.

This module measures the *model forward* stage of every representation, i.e.
the work that actually produces the stored embedding, with audio decoding and
file IO excluded. The fine-tuning blocks pre-compute their model inputs in the
dataset, so their measured latency excludes exactly the same overhead.

    mfcc        librosa MFCC over the window
    opensmile   openSMILE ComParE 2016 functionals over the window
    vggish      VGGish audio tower over the window
    ms-clap     MS-CLAP audio encoder over the window
    laion-clap  CLAP.get_audio_features over the pre-computed mel
    whisper     Whisper encoder + mean pooling over the pre-computed mel

For the two neural representations the front-end (mel spectrogram) is
computed once per window outside the timed region, which mirrors the
fine-tuning measurement where the mel tensors are pre-computed in the
dataset and only the encoder forward is timed.

Measured values can be cached to JSON so that an extractor which cannot be
built in the current environment (e.g. VGGish, which needs TensorFlow) can
still be reported. Set the REPRESENTATION_LATENCY_CACHE environment variable
to point the cache somewhere else.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

DEFAULT_CACHE_PATH = Path(
    os.environ.get(
        "REPRESENTATION_LATENCY_CACHE",
        ".representation_latency_cache.json",
    )
)

# feature_type (as used in the dataset configs) -> canonical representation name
CANONICAL_NAMES: Dict[str, str] = {
    "mfcc": "mfcc",
    "mfcc-13d": "mfcc",
    "opensmile": "opensmile",
    "opensmile-compare-2016": "opensmile",
    "vggish": "vggish",
    "ms-clap": "ms-clap",
    "ms-clap-2023": "ms-clap",
    "laion-clap": "laion-clap",
    "laion-clap-htsat-fused": "laion-clap",
    "whisper": "whisper",
    "whisper-base": "whisper",
}

# canonical name -> sample rate the representation is computed at
SAMPLE_RATES: Dict[str, int] = {
    "mfcc": 16000,
    "opensmile": 16000,
    "vggish": 16000,
    "ms-clap": 16000,
    "whisper": 16000,
    "laion-clap": 48000,
}

LAION_CLAP_MODEL = "laion/clap-htsat-fused"
WHISPER_MODEL = "openai/whisper-base"
MS_CLAP_VERSION = "2023"
VGGISH_URL = "https://tfhub.dev/google/vggish/1"


class RepresentationLatencyError(RuntimeError):
    """Raised when a representation cannot be measured in this environment."""


@dataclass
class RepresentationExtractor:
    """Splits a representation into an untimed front-end and a timed forward.

    ``prepare`` does the work that is shared with the fine-tuning blocks
    (mel spectrogram computation) and is executed outside the timed region.
    ``forward`` is the part that is timed and that produces the stored
    embedding.
    """

    name: str
    sample_rate: int
    prepare: Callable[[np.ndarray], Any]
    forward: Callable[[Any], Any]
    device: Optional[str] = None


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def _resolve_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def sync_device(device: Optional[str]) -> None:
    """Synchronises the CUDA device so timings are not read too early."""
    try:
        import torch

        if device is not None and str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------


def _identity(window: np.ndarray) -> np.ndarray:
    return np.asarray(window, dtype=np.float32)


def _build_mfcc() -> RepresentationExtractor:
    try:
        import librosa
    except ImportError as e:  # pragma: no cover - librosa is a hard dependency
        raise RepresentationLatencyError("librosa is not installed") from e

    def forward(frame: np.ndarray) -> np.ndarray:
        mfcc = librosa.feature.mfcc(
            y=np.asarray(frame, dtype=np.float32),
            sr=16000,
            n_mfcc=13,
        )
        return mfcc.mean(axis=1).astype(np.float32)

    return RepresentationExtractor("mfcc", 16000, _identity, forward)


def _build_opensmile() -> RepresentationExtractor:
    try:
        import opensmile
    except ImportError as e:
        raise RepresentationLatencyError(
            "openSMILE is not installed in this environment "
            "(pip install opensmile)"
        ) from e

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    def forward(frame: np.ndarray) -> np.ndarray:
        result = smile.process_signal(np.asarray(frame, dtype=np.float32), 16000)
        return result.values[0].astype(np.float32)

    return RepresentationExtractor("opensmile", 16000, _identity, forward, "cpu")


def _build_vggish() -> RepresentationExtractor:
    try:
        import tensorflow as tf
        import tensorflow_hub as hub
    except ImportError as e:
        raise RepresentationLatencyError(
            "tensorflow_hub is not installed in this environment "
            "(pip install tensorflow-hub)"
        ) from e

    model = hub.load(VGGISH_URL)
    device = "gpu" if tf.config.list_physical_devices("GPU") else "cpu"

    def forward(frame: np.ndarray) -> np.ndarray:
        return model(np.asarray(frame, dtype=np.float32)).numpy()

    return RepresentationExtractor("vggish", 16000, _identity, forward, device)


def _build_ms_clap() -> RepresentationExtractor:
    try:
        import librosa
        from msclap import CLAP
    except ImportError as e:
        raise RepresentationLatencyError(
            "msclap is not installed in this environment (pip install msclap)"
        ) from e

    import torch

    use_cuda = torch.cuda.is_available()
    model = CLAP(version=MS_CLAP_VERSION, use_cuda=use_cuda)

    target_rate = int(model.args.sampling_rate)
    target_samples = int(float(model.args.duration) * target_rate)

    def prepare(frame: np.ndarray) -> Any:
        audio = np.asarray(frame, dtype=np.float32)
        if target_rate != 16000:
            audio = librosa.resample(audio, orig_sr=16000, target_sr=target_rate)
        if len(audio) < target_samples:
            audio = np.tile(audio, int(np.ceil(target_samples / len(audio))))
        audio = audio[:target_samples]
        tensor = torch.from_numpy(np.ascontiguousarray(audio)).reshape(1, 1, -1)
        if use_cuda:
            tensor = tensor.cuda()
        return tensor

    def forward(prepared) -> np.ndarray:
        emb = model._get_audio_embeddings(prepared)
        if hasattr(emb, "detach"):
            emb = emb.detach().cpu().numpy()
        if getattr(emb, "ndim", 1) > 1:
            emb = emb[0]
        return np.asarray(emb, dtype=np.float32)

    return RepresentationExtractor(
        "ms-clap",
        16000,
        prepare,
        forward,
        device="cuda" if use_cuda else "cpu",
    )


def _build_laion_clap() -> RepresentationExtractor:
    try:
        import torch
        from transformers import ClapModel, ClapProcessor
    except ImportError as e:
        raise RepresentationLatencyError(
            "transformers/torch are required for the LAION-CLAP representation"
        ) from e

    device = _resolve_device()
    processor = ClapProcessor.from_pretrained(LAION_CLAP_MODEL)
    model = ClapModel.from_pretrained(LAION_CLAP_MODEL).to(device)
    model.eval()

    def prepare(frame: np.ndarray):
        inputs = processor(
            audio=[np.asarray(frame, dtype=np.float32)],
            sampling_rate=48000,
            return_tensors="pt",
        )
        # Same layout as the fine-tuning dataset: (batch, frames, mel bins)
        input_features = inputs.input_features.squeeze(0).unsqueeze(0).to(device)
        is_longer = inputs.is_longer.squeeze(0).unsqueeze(0).to(device)
        return input_features, is_longer

    @torch.no_grad()
    def forward(prepared) -> np.ndarray:
        input_features, is_longer = prepared
        features = model.get_audio_features(
            input_features=input_features,
            is_longer=is_longer,
        )
        if features.ndim > 1:
            features = features[0]
        return features.cpu().numpy().astype(np.float32)

    return RepresentationExtractor("laion-clap", 48000, prepare, forward, device)


def _build_whisper() -> RepresentationExtractor:
    try:
        import torch
        from transformers import WhisperFeatureExtractor, WhisperModel
    except ImportError as e:
        raise RepresentationLatencyError(
            "transformers/torch are required for the Whisper representation"
        ) from e

    device = _resolve_device()
    feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPER_MODEL)
    encoder = WhisperModel.from_pretrained(WHISPER_MODEL).encoder.to(device)
    encoder.eval()

    def prepare(frame: np.ndarray):
        inputs = feature_extractor(
            np.asarray(frame, dtype=np.float32),
            sampling_rate=16000,
            return_tensors="pt",
        )
        return inputs.input_features.to(device)

    @torch.no_grad()
    def forward(prepared) -> np.ndarray:
        hidden = encoder(input_features=prepared).last_hidden_state
        return hidden.mean(dim=1).cpu().numpy().astype(np.float32)

    return RepresentationExtractor("whisper", 16000, prepare, forward, device)


_BUILDERS: Dict[str, Callable[[], RepresentationExtractor]] = {
    "mfcc": _build_mfcc,
    "opensmile": _build_opensmile,
    "vggish": _build_vggish,
    "ms-clap": _build_ms_clap,
    "laion-clap": _build_laion_clap,
    "whisper": _build_whisper,
}

_BUILT: Dict[str, RepresentationExtractor] = {}


def build_extractor(name: str) -> RepresentationExtractor:
    """Builds (and caches in-process) the extractor for a canonical name."""
    if name not in _BUILDERS:
        raise RepresentationLatencyError(f"Unknown representation '{name}'")
    if name not in _BUILT:
        _BUILT[name] = _BUILDERS[name]()
    return _BUILT[name]


def canonical_name(feature_type: Optional[str]) -> Optional[str]:
    """Maps a dataset ``feature_type`` to the canonical representation name."""
    if not feature_type:
        return None
    key = str(feature_type).strip().lower()
    return CANONICAL_NAMES.get(key, key)


def _normalise_id(value: Any) -> str:
    """Case-insensitive, extension-insensitive key for a data identifier."""
    text = str(value).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text


# ---------------------------------------------------------------------------
# Window reconstruction
# ---------------------------------------------------------------------------


def sample_windows(
    dataset_config,
    feature_type: str,
    n_windows: int = 20,
    seed: int = 0,
) -> List[np.ndarray]:
    """Rebuilds real windows from the audio files listed in the embedding
    parquet, using the same framing (and silence trimming) as the extraction
    pipeline, so the timed windows are the ones the embeddings were built from.

    Only rows from the validation split are used (same singers the metrics are
    computed on).
    """
    import pandas as pd
    import pyarrow.parquet as pq

    from scripts.data.raw_audio import load_and_frame_audio

    parquet_path = Path(dataset_config.data_dir) / dataset_config.parquet_file
    if not parquet_path.exists():
        raise FileNotFoundError(f"Embedding parquet not found: {parquet_path}")

    schema = pq.ParquetFile(parquet_path).schema.names
    singer_col = dataset_config.data_column_label or "singer"
    columns = [c for c in ("filepath", "frame_index") if c in schema]
    if singer_col in schema:
        columns.append(singer_col)

    df = pd.read_parquet(parquet_path, columns=columns)

    val_ids = {
        _normalise_id(v) for v in (dataset_config.test_data_ids or [])
    }
    if val_ids and singer_col in df.columns:
        df = df[df[singer_col].map(_normalise_id).isin(val_ids)]

    if len(df) == 0:
        raise ValueError("No validation rows found in the embedding parquet")

    rng = np.random.default_rng(seed)
    n = min(n_windows, len(df))
    picks = rng.choice(len(df), size=n, replace=False)

    sample_rate = SAMPLE_RATES.get(canonical_name(feature_type) or "", 16000)

    windows: List[np.ndarray] = []
    for idx in picks:
        row = df.iloc[int(idx)]
        frames = load_and_frame_audio(
            row["filepath"],
            sample_rate=sample_rate,
            frame_duration=dataset_config.frame_duration or 3.0,
            overlap=dataset_config.overlap if dataset_config.overlap is not None else 0.0,
            enable_silence_trimming=bool(dataset_config.enable_silence_trimming),
            silence_threshold_db=dataset_config.silence_threshold_db,
            silence_min_duration_ms=dataset_config.silence_min_duration_ms,
            normalization_mode=dataset_config.normalization_mode,
            short_audio_handling=dataset_config.short_audio_handling,
            tail_window_mode=dataset_config.tail_window_mode,
        )
        if not frames:
            continue
        frame_index = int(row["frame_index"]) if "frame_index" in row else 0
        window = frames[min(frame_index, len(frames) - 1)][0]
        windows.append(np.asarray(window, dtype=np.float32))

    return windows


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def measure_extractor_latency(
    extractor: RepresentationExtractor,
    windows: List[np.ndarray],
    warmups: int = 2,
    iters: int = 5,
) -> Dict[str, Any]:
    """Times the forward stage of ``extractor`` over ``windows`` at batch 1.

    The front-end runs once per window outside the timed region, mirroring
    the fine-tuning measurement where the model inputs are pre-computed.
    """
    if not windows:
        raise ValueError("No windows to measure")

    per_window_ms: List[float] = []

    for window in windows:
        prepared = extractor.prepare(window)

        for _ in range(warmups):
            extractor.forward(prepared)

        sync_device(extractor.device)
        start = time.perf_counter()
        for _ in range(iters):
            extractor.forward(prepared)
        sync_device(extractor.device)
        elapsed = time.perf_counter() - start

        per_window_ms.append((elapsed * 1000.0) / max(1, iters))

    mean_ms = float(np.mean(per_window_ms))
    return {
        "mean_ms": mean_ms,
        "std_ms": float(np.std(per_window_ms)),
        "calls_per_s": 1000.0 / mean_ms if mean_ms > 0 else 0.0,
        "n_windows": len(windows),
    }


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


def load_cache(cache_path: Optional[Path] = None) -> Dict[str, Any]:
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def save_cache(cache: Dict[str, Any], cache_path: Optional[Path] = None) -> Path:
    path = Path(cache_path) if cache_path else DEFAULT_CACHE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cache, fh, indent=2, sort_keys=True)
    return path


def update_cache(
    name: str,
    entry: Dict[str, Any],
    cache_path: Optional[Path] = None,
) -> Path:
    cache = load_cache(cache_path)
    cache[name] = entry
    return save_cache(cache, cache_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def representation_latency_columns(
    feature_type: Optional[str],
    dataset_config=None,
    n_windows: int = 20,
    warmups: int = 10,
    iters: int = 50,
    cache_path: Optional[Path] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Returns the representation-latency columns for a traditional-ML run.

    Returns an empty dict when the representation cannot be measured, so a
    missing optional dependency never breaks a benchmark run.
    """
    name = canonical_name(feature_type)
    if not name:
        return {}

    column = "Feature Extraction Latency (ms/window) (bs=1)"

    if use_cache:
        cached = load_cache(cache_path).get(name)
        if cached and "mean_ms" in cached:
            return {
                column: round(float(cached["mean_ms"]), 4),
                "Representation Latency Source": cached.get("source", "cache"),
            }

    if dataset_config is None:
        return {}

    try:
        extractor = build_extractor(name)
        windows = sample_windows(
            dataset_config,
            feature_type,
            n_windows=n_windows,
        )
        stats = measure_extractor_latency(
            extractor, windows, warmups=warmups, iters=iters
        )
    except Exception as e:
        print(
            f"[WARN] Could not measure representation latency for '{name}': "
            f"{type(e).__name__}: {e}"
        )
        return {}

    entry = {
        "mean_ms": round(stats["mean_ms"], 4),
        "std_ms": round(stats["std_ms"], 4),
        "calls_per_s": round(stats["calls_per_s"], 3),
        "n_windows": stats["n_windows"],
        "source": "live",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "feature_type": feature_type,
        "device": extractor.device or "cpu",
    }
    try:
        update_cache(name, entry, cache_path)
    except Exception:
        pass

    return {
        column: entry["mean_ms"],
        "Representation Latency Source": "live",
    }
