from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.config.benchmark_config import BenchmarkConfig
from scripts.config.dataset_config import DatasetConfig
from scripts.evaluation.metrics import (
    compute_metrics,
    save_metrics_csv,
)
from scripts.utils import ensure_dir

MODEL_METADATA = {
    "name": "clap",
    "supported_data_types": ["raw_audio"],
    "description": "CLAP embeddings + classifier (head_only or full finetuning)",
    "default_params": {
        "finetune_mode": "head_only",
        "learning_rate": 0.001,
        "encoder_learning_rate": 1e-6,
        "batch_size": 32,
        "epochs": 100,
        "patience": 10,
        "frame_duration": 3.0,
        "overlap": 0.25,
        "sample_rate": 44100,
        "model_name": "laion/clap-htsat-fused",
    },
}


def _make_windows(audio: np.ndarray, frame_len: int, hop_len: int) -> list[np.ndarray]:
    """Slice `audio` into overlapping fixed-length windows.

    - If audio is shorter than one frame, pad it and return a single window.
    - Otherwise, step by hop_len, and make sure the final partial window at
      the tail is still covered (by adding one last window aligned to the
      end of the audio) instead of being dropped.
    """
    if len(audio) < frame_len:
        return [np.pad(audio, (0, frame_len - len(audio)))]

    starts = list(range(0, len(audio) - frame_len + 1, hop_len))
    last_full_start = len(audio) - frame_len
    if not starts or starts[-1] != last_full_start:
        starts.append(last_full_start)

    return [audio[s : s + frame_len] for s in starts]


class ClapForClassification(nn.Module):
    """CLAP audio encoder + projection + linear classifier.

    Same architecture is used for both frozen (head_only) and full
    fine-tuning modes; the only difference is which parameters have
    requires_grad=True. Uses get_audio_features() so that both modes
    share the same CLAP projected 512-D audio embedding.
    """

    def __init__(self, model_name: str, n_classes: int):
        super().__init__()
        from transformers import ClapModel

        self.clap = ClapModel.from_pretrained(model_name)

        # CLAP's final projected audio embedding dimension (512 for
        # laion/clap-htsat-fused).
        embedding_dim = self.clap.config.projection_dim

        self.classifier = nn.Linear(embedding_dim, n_classes)

    def forward(self, input_features, is_longer=None):
        audio_features = self.clap.get_audio_features(
            input_features=input_features,
            is_longer=is_longer,
        )
        return self.classifier(audio_features)


class ClapFinetuneDataset(Dataset):
    """Splits each raw audio source file into overlapping frame_duration-second
    windows and produces one training sample per window.

    All windows and their labels are precomputed in __init__ so that each
    __getitem__ returns a fixed-length feature tensor for a single window.
    """

    def __init__(
        self,
        filepaths: List[str],
        labels: List[int],
        model_name: str,
        sample_rate: int = 48000,
        frame_duration: float = 3.0,
        overlap: float = 0.25,
        enable_silence_trimming: bool = False,
        silence_threshold_db: float = -40.0,
        silence_min_duration_ms: float = 250.0,
        normalization_mode: str = "none",
        short_audio_handling: str = "pad",
        tail_window_mode: str = "shift",
    ):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.overlap = overlap
        self.enable_silence_trimming = enable_silence_trimming
        self.silence_threshold_db = silence_threshold_db
        self.silence_min_duration_ms = silence_min_duration_ms
        self.normalization_mode = normalization_mode
        self.short_audio_handling = short_audio_handling
        self.tail_window_mode = tail_window_mode

        from transformers import ClapProcessor
        self.processor = ClapProcessor.from_pretrained(model_name)

        # Expands each source file into its 3-second windows.
        self._items = []
        for filepath, label in zip(filepaths, labels):
            self._items.extend(self._windows_for_file(filepath, label))

    def _windows_for_file(self, filepath: str, label: int) -> List[tuple]:
        from scripts.data.raw_audio import load_and_frame_audio

        frames = load_and_frame_audio(
            filepath,
            sample_rate=self.sample_rate,
            frame_duration=self.frame_duration,
            overlap=self.overlap,
            enable_silence_trimming=self.enable_silence_trimming,
            silence_threshold_db=self.silence_threshold_db,
            silence_min_duration_ms=self.silence_min_duration_ms,
            normalization_mode=self.normalization_mode,
            short_audio_handling=self.short_audio_handling,
            tail_window_mode=self.tail_window_mode,
        )

        items = []
        for window, _padded in frames:
            inputs = self.processor(
                audio=[window],
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            )
            input_features = inputs.input_features.squeeze(0)
            is_longer = inputs.is_longer.squeeze(0)
            items.append((input_features, is_longer, label))
        return items

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        input_features, is_longer, label = self._items[idx]
        return input_features, is_longer, label


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    # CLAP's audio encoder contains BatchNorm2d layers. In train() mode they
    # normalise with batch statistics and keep updating their running
    # statistics, so a "frozen" encoder would still drift away from the
    # extracted embeddings. For the frozen (head-only) mode the encoder is
    # forced back to eval() so the head trains on exactly the same features
    # the traditional-ML block consumes.
    if getattr(model, "keep_encoder_eval", False):
        model.clap.eval()
    total_loss = 0
    for input_features, is_longer, labels in dataloader:
        input_features = input_features.to(device)
        is_longer = is_longer.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_features, is_longer)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    total_loss = 0
    with torch.no_grad():
        for input_features, is_longer, labels in dataloader:
            input_features = input_features.to(device)
            is_longer = is_longer.to(device)
            labels = labels.to(device)

            outputs = model(input_features, is_longer)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    return (
        total_loss / len(dataloader),
        np.array(all_preds),
        np.array(all_labels),
        np.concatenate(all_probs, axis=0) if all_probs else None,
    )


def _build_datasets(dataset_config, params):
    """Scan/split raw audio and build train + val ClapFinetuneDataset windows.
    Returns (train_dataset, val_dataset, class_names)."""
    from scripts.data.raw_audio import scan_audio_files

    singer_map = _get_split_args(dataset_config)
    file_df = scan_audio_files(
        audio_dir=dataset_config.audio_dir or dataset_config.data_dir,
        gender_filter=dataset_config.gender_split,
        include_labels=dataset_config.include_labels,
        exclude_labels=dataset_config.exclude_labels,
        level_names=dataset_config.level_names,
        label_level=dataset_config.label_level,
        **singer_map,
    )

    classes = _resolve_classes(file_df, dataset_config)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_df = file_df["train"]
    val_df = file_df["val"]
    train_df = train_df[train_df["label"].isin(classes)]
    val_df = val_df[val_df["label"].isin(classes)]

    window_kwargs = dict(
        model_name=params["model_name"],
        sample_rate=params["sample_rate"],
        frame_duration=params["frame_duration"],
        overlap=params["overlap"],
        enable_silence_trimming=getattr(dataset_config, "enable_silence_trimming", False),
        silence_threshold_db=getattr(dataset_config, "silence_threshold_db", -40.0),
        silence_min_duration_ms=getattr(dataset_config, "silence_min_duration_ms", 250.0),
        normalization_mode=getattr(dataset_config, "normalization_mode", "none"),
        short_audio_handling=getattr(dataset_config, "short_audio_handling", "pad"),
        tail_window_mode=getattr(dataset_config, "tail_window_mode", "shift"),
    )

    from scripts.data.dataset_cache import audio_key, get_or_build

    train_dataset = get_or_build(
        audio_key(dataset_config, params, "train"),
        lambda: ClapFinetuneDataset(
            filepaths=train_df["filepath"].tolist(),
            labels=[class_to_idx[l] for l in train_df["label"]],
            **window_kwargs,
        ),
    )

    val_dataset = get_or_build(
        audio_key(dataset_config, params, "val"),
        lambda: ClapFinetuneDataset(
            filepaths=val_df["filepath"].tolist(),
            labels=[class_to_idx[l] for l in val_df["label"]],
            **window_kwargs,
        ),
    )

    return train_dataset, val_dataset, classes


def _measure_latency_clap(model, val_dataset, device, params) -> Dict[str, float]:
    """Times encoder, classifier, and end-to-end forward at batch 1 and the
    training batch size, returning per-window latency columns."""
    import torch
    from scripts.evaluation.latency import measure_clap_components, to_latency_columns

    n = len(val_dataset)
    if n == 0:
        return {}

    batch_sizes = []
    for bs in (1, params.get("batch_size", 1)):
        if bs and bs not in batch_sizes:
            batch_sizes.append(int(bs))

    columns: Dict[str, float] = {}
    for bs in batch_sizes:
        k = min(bs, n)
        feats = torch.stack([val_dataset[i][0] for i in range(k)]).to(device)
        is_longer = torch.stack([val_dataset[i][1] for i in range(k)]).to(device)
        stats = measure_clap_components(model, feats, is_longer, device)
        for key, value in to_latency_columns(stats, k).items():
            columns[f"{key} (bs={bs})"] = value
    return columns


def _run_clap(
    dataset_config, bench_config, output_dir, device, params, freeze_encoder
) -> Dict[str, Any]:
    """Shared training loop for both frozen (head_only) and full fine-tuning.

    Both modes instantiate the identical ClapForClassification architecture
    over 3-second windows; freeze_encoder only controls requires_grad on the
    CLAP audio encoder + projection (freezing makes the unused CLAP text side
    frozen too).
    """
    start_time = time.perf_counter()
    early_stopping = params.get("early_stopping", True)

    train_dataset, val_dataset, classes = _build_datasets(
        dataset_config, params
    )

    feature_extraction_time = time.perf_counter() - start_time

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=0,
        # CLAP's audio encoder has a BatchNorm2d over the mel bins, which
        # cannot compute per-channel statistics from a batch of one. Drop the
        # trailing partial batch so training never sees a size-1 batch.
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    if len(train_loader) == 0:
        raise ValueError(
            f"CLAP training needs at least one full batch: got {len(train_dataset)} "
            f"windows with batch_size={params['batch_size']} and drop_last=True."
        )

    model = ClapForClassification(
        model_name=params["model_name"],
        n_classes=len(classes),
    ).to(device)

    if freeze_encoder:
        # Freeze the entire CLAP (audio encoder + projection). Only the
        # classifier is trainable.
        for param in model.clap.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True
        # Keep the encoder in eval() mode so its BatchNorm running statistics
        # are not updated and the features match the extracted embeddings.
        model.keep_encoder_eval = True

        optimizer = torch.optim.Adam(
            model.classifier.parameters(),
            lr=params["learning_rate"],
        )
    else:
        # Full fine-tuning: freeze CLAP's unused text side, train the audio
        # encoder + projection + classifier.
        for param in model.clap.parameters():
            param.requires_grad = False
        for param in model.clap.audio_model.parameters():
            param.requires_grad = True
        for param in model.clap.audio_projection.parameters():
            param.requires_grad = True
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = torch.optim.AdamW([
            {"params": model.clap.audio_model.parameters(), "lr": params.get("encoder_learning_rate", 1e-6)},
            {"params": model.clap.audio_projection.parameters(), "lr": params.get("encoder_learning_rate", 1e-6)},
            {"params": model.classifier.parameters(), "lr": params["learning_rate"]},
        ])

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"loss": [], "val_loss": [], "val_accuracy": []}

    from scripts.evaluation.metrics_utils import ResourceTracker
    resource_tracker = ResourceTracker()
    resource_tracker.start()

    training_start = time.perf_counter()
    actual_epochs = 0
    for epoch in range(params["epochs"]):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        actual_epochs = epoch + 1

        history["loss"].append(train_loss)

        # When early stopping is disabled the validation split is never used for
        # model selection, so skip the per-epoch validation pass entirely and
        # only evaluate once at the end (for reporting).
        if not early_stopping:
            print(f"Epoch {epoch + 1}/{params['epochs']} - train_loss: {train_loss:.4f}")
            continue

        val_loss, val_preds, val_labels, _ = validate(model, val_loader, device)
        val_acc = (val_preds == val_labels).mean()
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(float(val_acc))

        print(f"Epoch {epoch + 1}/{params['epochs']} - "
              f"train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))
        else:
            patience_counter += 1
            if patience_counter >= params["patience"]:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    training_time = time.perf_counter() - training_start
    resource_stats = resource_tracker.stop()

    if early_stopping:
        model.load_state_dict(torch.load(str(output_dir / "best_model.pt")))
    model.eval()

    eval_start = time.perf_counter()
    _, train_preds, train_labels, train_probs = validate(model, train_loader, device)
    _, val_preds, val_labels, val_probs = validate(model, val_loader, device)
    eval_time = time.perf_counter() - eval_start

    train_metrics = compute_metrics(train_labels, train_preds, train_probs, classes)
    val_metrics = compute_metrics(val_labels, val_preds, val_probs, classes)

    embedding_dim = model.clap.config.projection_dim
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    complexity = {
        "Backbone": f"CLAP ({params['model_name']})",
        "Classifier Head": f"Linear({embedding_dim}, {len(classes)})",
        "Total Params": total_params,
        "Trainable Params": trainable_params,
        "Non-trainable Params": total_params - trainable_params,
        "Encoder Frozen": freeze_encoder,
    }

    model_name = "clap_head_only" if freeze_encoder else "clap_full"

    total_time = time.perf_counter() - start_time
    latency_extra = {}
    if getattr(bench_config, "measure_latency", False):
        try:
            latency_extra = _measure_latency_clap(model, val_dataset, device, params)
        except Exception as e:
            print(f"[WARN] CLAP latency measurement failed: {e}")

    metrics_path = Path(dataset_config.metrics_path or output_dir / "metrics.csv")
    save_metrics_csv(
        metrics_path,
        model_name,
        dataset_config.name,
        train_metrics,
        val_metrics,
        extra={
            "Epochs": actual_epochs,
            "Max Epochs": params.get("epochs", bench_config.epochs),
            "Batch Size": params.get("batch_size", bench_config.batch_size),
            "Feature Extraction Time (s)": round(feature_extraction_time, 4),
            "Encoder Training Time (s)": round(training_time, 4),
            "Eval Inference Time (s)": round(eval_time, 4),
            "Total Time (s)": round(total_time, 4),
            "Train+Eval Time (s)": round(total_time, 4),
            "Early Stopping": early_stopping,
            "frame_duration": dataset_config.frame_duration,
            "overlap": dataset_config.overlap,
            **resource_stats,
            **latency_extra,
        },
        complexity=complexity,
    )

    return {
        "model_name": model_name,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "history": history,
        "complexity": complexity,
        "resource_stats": resource_stats,
        "actual_epochs": actual_epochs,
        "max_epochs": params.get("epochs", bench_config.epochs),
        "batch_size": params.get("batch_size", bench_config.batch_size),
        "train_time_s": round(total_time, 4),
        "feature_extraction_time_s": round(feature_extraction_time, 4),
        "training_time_s": round(training_time, 4),
        "eval_time_s": round(eval_time, 4),
        "latency": latency_extra,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(
    dataset_config: DatasetConfig,
    bench_config: BenchmarkConfig,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = MODEL_METADATA["default_params"].copy()
    params.update(bench_config.get_model_params("clap"))

    params["frame_duration"] = dataset_config.frame_duration
    params["overlap"] = dataset_config.overlap
    params["sample_rate"] = dataset_config.sample_rate

    finetune_mode = params.get("finetune_mode", "head_only")

    if finetune_mode == "head_only":
        return _run_clap(dataset_config, bench_config, output_dir, device, params, freeze_encoder=True)
    elif finetune_mode == "full":
        return _run_clap(dataset_config, bench_config, output_dir, device, params, freeze_encoder=False)
    else:
        raise ValueError(f"Unknown finetune_mode: {finetune_mode}. Use 'head_only' or 'full'.")


def _get_split_args(dataset_config: DatasetConfig) -> Dict[str, Any]:
    """Maps the partition ids onto the ``scan_audio_files`` split arguments.

    When the partition's column is ``filename`` the split is per file (test
    set = the listed files, train set = everything left over); otherwise it
    falls back to the singer ids.
    """
    if not dataset_config.test_data_ids:
        return {}
    if dataset_config.data_column_label == "filename":
        return {
            "train_files": dataset_config.train_data_ids,
            "val_files": dataset_config.test_data_ids,
        }
    return {
        "train_singers": dataset_config.train_data_ids,
        "val_singers": dataset_config.test_data_ids,
    }


def _resolve_classes(file_df, dataset_config):
    if dataset_config.label_map:
        return sorted(dataset_config.label_map.keys(), key=lambda c: dataset_config.label_map[c])
    all_labels = set(file_df["train"]["label"].unique()) | set(file_df["val"]["label"].unique())
    return sorted(all_labels)
