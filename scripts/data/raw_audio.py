from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from scripts.config.dataset_config import DatasetConfig


@dataclass
class AudioMetadata:
    filepath: str
    label: str
    singer_id: str
    gender: str
    technique: str
    exercise: Optional[str] = None


def _normalise_id(value: Any) -> str:
    """Case-insensitive, extension-insensitive key for a data identifier."""
    text = str(value).strip().lower()
    if "." in text:
        text = text.rsplit(".", 1)[0]
    return text


def _split_ids(
    df: pd.DataFrame,
    id_column: str,
    train_ids: Optional[List[str]],
    val_ids: List[str],
) -> Dict[str, pd.DataFrame]:
    """Split ``df`` on ``id_column``; train is the complement of val when no
    explicit train list is given."""
    keys = df[id_column].map(_normalise_id)
    val_keys = {_normalise_id(i) for i in val_ids}
    val_df = df[keys.isin(val_keys)]
    if train_ids:
        train_keys = {_normalise_id(i) for i in train_ids}
        train_df = df[keys.isin(train_keys)]
    else:
        train_df = df[~keys.isin(val_keys)]
    return {"train": train_df, "val": val_df}


def scan_audio_files(
    audio_dir: str | Path,
    extension: str = ".wav",
    train_singers: Optional[List[str]] = None,
    val_singers: Optional[List[str]] = None,
    gender_filter: Optional[str] = None,
    include_labels: Optional[List[str]] = None,
    exclude_labels: Optional[List[str]] = None,
    level_names: Optional[List[str]] = None,
    label_level: Optional[str] = None,
    train_files: Optional[List[str]] = None,
    val_files: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    audio_dir = Path(audio_dir)
    level_names = level_names or ["singer", "technique", "exercise"]
    label_level = label_level or "technique"

    if "singer" not in level_names:
        raise ValueError("level_names must include 'singer'")
    if label_level not in level_names:
        raise ValueError(
            f"label_level {label_level!r} must be present in level_names"
        )

    all_files = list(audio_dir.rglob(f"*{extension}"))

    records = []
    for fp in all_files:
        rel = fp.relative_to(audio_dir)
        parts = list(rel.parts)
        levels = dict(zip(level_names, parts))
        singer_id = levels["singer"]
        label = levels.get(label_level, "unknown")
        technique = levels.get("technique", label)
        exercise = levels.get("exercise")
        gender = "female" if singer_id.startswith("f") else "male" if singer_id.startswith("m") else "unknown"

        if gender_filter and gender_filter != "mixed" and gender != gender_filter:
            continue
        if include_labels and label not in include_labels:
            continue
        if exclude_labels and label in exclude_labels:
            continue

        record = {
            "filepath": str(fp),
            "filename": fp.name,
            "singer_id": singer_id,
            "gender": gender,
            "technique": technique,
            "label": label,
        }
        if exercise is not None:
            record["exercise"] = exercise
        records.append(record)

    df = pd.DataFrame(records)

    if val_files:
        return _split_ids(df, "filename", train_files, val_files)

    if train_singers is not None and val_singers is not None:
        return _split_ids(df, "singer_id", train_singers, val_singers)

    from sklearn.model_selection import train_test_split
    singers = df["singer_id"].unique()
    train_ids, val_ids = train_test_split(
        singers, test_size=0.25, random_state=42
    )
    return _split_ids(df, "singer_id", list(train_ids), list(val_ids))


def load_audio_file(path: str | Path, target_sr: int = 16000) -> np.ndarray:
    import librosa
    audio, _ = librosa.load(path, sr=target_sr, mono=True)
    return audio


def _trim_silence_smart(
    audio: np.ndarray,
    threshold_db: float = -40.0,
) -> np.ndarray:
    """Trim edge silence using librosa's trim. Ported from the
    embedding-extraction repo so both pipelines behave identically."""
    import librosa

    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-6:
        return audio
    trimmed, _ = librosa.effects.trim(audio, top_db=abs(threshold_db), ref=np.max)
    return trimmed


def _normalize_audio(audio: np.ndarray, mode: str = "none") -> np.ndarray:
    if mode in (None, "none"):
        return audio
    if mode == "peak":
        peak = np.max(np.abs(audio))
        return audio / peak if peak > 0 else audio
    if mode == "rms":
        rms = np.sqrt(np.mean(audio**2))
        return audio / rms if rms > 0 else audio
    if mode == "lufs":
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(48000)
            loudness = meter.integrated_loudness(audio)
            return pyln.normalize.loudness(audio, loudness, -23.0)
        except Exception:
            return audio
    return audio


def load_and_frame_audio(
    audio_file: str | Path,
    sample_rate: int = 16000,
    frame_duration: float = 3.0,
    overlap: float = 0.0,
    enable_silence_trimming: bool = False,
    silence_threshold_db: float = -40.0,
    silence_min_duration_ms: float = 250.0,
    normalization_mode: str = "none",
    short_audio_handling: str = "pad",
    tail_window_mode: str = "pad",
) -> List[tuple]:
    """Load an audio file and split it into fixed-length frames.

    Mirrors ``load_and_frame_audio`` in the embedding-extraction repo so that
    fine-tuning windows are directly comparable to extracted embeddings.

    Returns a list of ``(frame, padded)`` tuples where ``frame`` is a float32
    array of ``frame_duration * sample_rate`` samples and ``padded`` is True
    for the zero-padded tail (and for short files zero-padded to a full
    window). Returns an empty list on load error or empty audio.

    ``tail_window_mode``:
        - ``"pad"``: trailing partial window is zero-padded into one extra
          frame (extraction behaviour).
        - ``"shift"``: trailing partial window is covered by sliding the final
          window back to the end of the audio (original fine-tuning
          behaviour).
    """
    import librosa

    try:
        audio, sr = librosa.load(str(audio_file), sr=None, mono=True)
        if sr != sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
            sr = sample_rate
    except Exception as e:
        print(f"[WARN] Error loading {audio_file}: {e}")
        return []

    audio = np.asarray(audio, dtype=np.float32)

    if enable_silence_trimming:
        audio = _trim_silence_smart(audio, silence_threshold_db)

    audio = _normalize_audio(audio, normalization_mode)

    if len(audio) == 0:
        return []

    frame_len = int(sample_rate * frame_duration)
    hop_len = max(1, int(frame_len * (1 - overlap)))

    if len(audio) < frame_len:
        if short_audio_handling == "tile":
            repeat = int(np.ceil(frame_len / len(audio)))
            frame = np.tile(audio, repeat)[:frame_len].astype(np.float32)
            return [(frame, False)]
        padded = np.zeros(frame_len, dtype=np.float32)
        padded[: len(audio)] = audio
        return [(padded, True)]

    starts = list(range(0, len(audio) - frame_len + 1, hop_len))
    frames = [(audio[s: s + frame_len], False) for s in starts]

    if tail_window_mode == "pad":
        covered_end = (starts[-1] + frame_len) if starts else 0
        if covered_end < len(audio):
            tail = np.zeros(frame_len, dtype=np.float32)
            tail[: len(audio) - covered_end] = audio[covered_end:]
            frames.append((tail, True))
    elif tail_window_mode == "shift":
        last_full_start = len(audio) - frame_len
        if not starts or starts[-1] != last_full_start:
            frames.append((audio[last_full_start: last_full_start + frame_len], False))

    return frames


def frame_audio(
    audio: np.ndarray,
    sample_rate: int,
    frame_duration: float,
    overlap: float,
) -> List[np.ndarray]:
    frame_len = int(sample_rate * frame_duration)
    hop_len = int(frame_len * (1 - overlap))
    if len(audio) < frame_len:
        padding = frame_len - len(audio)
        audio = np.pad(audio, (0, padding))
    frames = []
    for start in range(0, len(audio) - frame_len + 1, hop_len):
        frames.append(audio[start: start + frame_len])
    return frames
