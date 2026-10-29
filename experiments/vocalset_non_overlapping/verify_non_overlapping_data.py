"""Verify the non-overlapping VocalSet embeddings against the raw audio.

Three independent checks:

1. Row counts per extractor parquet (total, padded tails, VOCALSET_10 after
   filtering, and the singer-independent train/val split).
2. Framing parity: re-frames sampled raw recordings with the same
   ``load_and_frame_audio`` helper used by the fine-tuning models and compares
   the per-file ``(frame_index, padded)`` sequence with the parquet.
3. MFCC parity: recomputes the mean-MFCC embedding for sampled frames and
   compares it with the stored parquet embedding.

Run from the repository root:

    python -m experiments.vocalset_non_overlapping.verify_non_overlapping_data
    python -m experiments.vocalset_non_overlapping.verify_non_overlapping_data --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.vocalset_non_overlapping.vocalset_non_overlapping_experiments_constants import (
    VOCALSET_10,
    TRAIN_SINGERS,
    VAL_SINGERS,
    RAW_AUDIO_DATA_DIR,
    EMBEDDINGS_DATA_DIR,
)

EXPECTED_BY_EXTRACTOR = {
    "laion-clap": {
        "total": 11664,
        "padded": 3610,
        "filtered": 6371,
        "train": 4667,
        "val": 1704,
    },
}
EXPECTED_DEFAULT = {
    "total": 11697,
    "padded": 3611,
    "filtered": 6378,
    "train": 4671,
    "val": 1707,
}


def _expected_for(path: Path) -> dict[str, int]:
    for extractor, expected in EXPECTED_BY_EXTRACTOR.items():
        if extractor in path.stem:
            return expected
    return EXPECTED_DEFAULT


# Sample rate used by the extraction pipeline when framing each extractor.
EXTRACTOR_SAMPLE_RATE = {
    "laion-clap": 48000,
}
DEFAULT_SAMPLE_RATE = 16000


def _ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


def check_counts(parquet_dir: Path) -> bool:
    print("\n=== 1. Row counts ===")
    all_ok = True
    parquets = sorted(parquet_dir.glob("*.parquet"))
    if not parquets:
        print(f"  No parquet files found in {parquet_dir}")
        return False

    for path in parquets:
        df = pd.read_parquet(
            path,
            columns=["filename", "singer", "vocal_technique", "frame_index", "padded"],
        )
        filtered = df[df["vocal_technique"].isin(VOCALSET_10)]
        train = filtered[filtered["singer"].isin(TRAIN_SINGERS)]
        val = filtered[filtered["singer"].isin(VAL_SINGERS)]

        actual = {
            "total": len(df),
            "padded": int(df["padded"].sum()),
            "filtered": len(filtered),
            "train": len(train),
            "val": len(val),
        }
        expected = _expected_for(path)
        conds = {k: actual[k] == expected[k] for k in expected}
        all_ok &= all(conds.values())
        status = "PASS" if all(conds.values()) else "FAIL"
        detail = ", ".join(
            f"{k}={actual[k]}" + ("" if conds[k] else f"(exp {expected[k]})")
            for k in expected
        )
        print(f"  [{status}] {path.name}: {detail}")
    return all_ok


def _padded_sequence(df: pd.DataFrame) -> list[bool]:
    df = df.sort_values("frame_index")
    return df["padded"].astype(bool).tolist()


def check_framing(parquet_dir: Path, n_files: int, full: bool) -> bool:
    print("\n=== 2. Framing parity (raw audio -> parquet) ===")
    from scripts.data.raw_audio import load_and_frame_audio

    path = parquet_dir / "vocalset_dataset_mfcc-13d_3.0s.parquet"
    if not path.exists():
        print(f"  Missing {path}")
        return False
    df = pd.read_parquet(path, columns=["filepath", "frame_index", "padded"])

    filepaths = list(dict.fromkeys(df["filepath"].tolist()))
    if not full:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(filepaths), size=min(n_files, len(filepaths)), replace=False)
        filepaths = [filepaths[i] for i in idx]

    mismatches = 0
    checked_windows = 0
    for fp in filepaths:
        sub = df[df["filepath"] == fp]
        expected = _padded_sequence(sub)
        frames = load_and_frame_audio(
            fp,
            sample_rate=DEFAULT_SAMPLE_RATE,
            frame_duration=3.0,
            overlap=0.0,
            enable_silence_trimming=True,
            silence_threshold_db=-40.0,
            normalization_mode="none",
            short_audio_handling="pad",
            tail_window_mode="pad",
        )
        actual = [bool(p) for _, p in frames]
        checked_windows += len(expected)
        if actual != expected:
            mismatches += 1
            if mismatches <= 5:
                print(
                    f"  MISMATCH {Path(fp).name}: "
                    f"expected {len(expected)} windows {expected}, "
                    f"got {len(actual)} {actual}"
                )

    ok = mismatches == 0
    print(
        f"  [{_ok(ok)}] checked {len(filepaths)} files / {checked_windows} windows, "
        f"{mismatches} mismatches"
    )
    return ok


def check_mfcc(parquet_dir: Path, n_frames: int) -> bool:
    print("\n=== 3. MFCC parity (recomputed vs parquet) ===")
    import librosa

    path = parquet_dir / "vocalset_dataset_mfcc-13d_3.0s.parquet"
    if not path.exists():
        print(f"  Missing {path}")
        return False
    df = pd.read_parquet(
        path, columns=["filepath", "frame_index", "padded", "embedding"]
    )
    df = df[df["padded"] == False]  # noqa: E712  (non-padded frames are exact)

    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(n_frames, len(df)), replace=False)
    sample = df.iloc[idx]

    from scripts.data.raw_audio import load_and_frame_audio

    max_abs = 0.0
    max_rel = 0.0
    checked = 0
    for _, row in sample.iterrows():
        frames = load_and_frame_audio(
            row["filepath"],
            sample_rate=DEFAULT_SAMPLE_RATE,
            frame_duration=3.0,
            overlap=0.0,
            enable_silence_trimming=True,
            silence_threshold_db=-40.0,
            normalization_mode="none",
            short_audio_handling="pad",
            tail_window_mode="pad",
        )
        fi = int(row["frame_index"])
        if fi >= len(frames):
            print(f"  SKIP {Path(row['filepath']).name}: frame_index {fi} out of range")
            continue
        frame = frames[fi][0]
        recomputed = librosa.feature.mfcc(
            y=frame.astype(np.float32), sr=DEFAULT_SAMPLE_RATE, n_mfcc=13
        ).mean(axis=1).astype(np.float32)
        raw = row["embedding"]
        if isinstance(raw, (bytes, bytearray)):
            stored = np.frombuffer(raw, dtype=np.float32)
        else:
            stored = np.asarray(raw, dtype=np.float32)
        if recomputed.shape != stored.shape:
            print(f"  SHAPE MISMATCH: {recomputed.shape} vs {stored.shape}")
            continue
        abs_err = float(np.max(np.abs(recomputed - stored)))
        rel = abs_err / (float(np.max(np.abs(stored))) + 1e-8)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel)
        checked += 1

    ok = checked > 0 and max_abs < 1e-3
    print(
        f"  [{_ok(ok)}] checked {checked} frames, max abs err={max_abs:.6g}, "
        f"max rel err={max_rel:.6g}"
    )
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet-dir", default=EMBEDDINGS_DATA_DIR)
    parser.add_argument(
        "--files",
        type=int,
        default=25,
        help="number of raw recordings to re-frame (ignored with --full)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="number of frames to recompute MFCC for",
    )
    parser.add_argument("--full", action="store_true", help="check every recording")
    args = parser.parse_args(argv)

    parquet_dir = Path(args.parquet_dir)
    print(f"Parquet dir: {parquet_dir}")
    print(f"Audio dir:   {RAW_AUDIO_DATA_DIR}")

    results = [
        check_counts(parquet_dir),
        check_framing(parquet_dir, args.files, args.full),
        check_mfcc(parquet_dir, args.frames),
    ]

    print("\n========================================")
    if all(results):
        print("ALL CHECKS PASSED")
        return 0
    print("SOME CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
