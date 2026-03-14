"""Training callbacks: checkpointing, early stopping, CSV logging, LR tracking."""

import os
import json
import csv
import heapq
from collections import OrderedDict

import torch

import config


class ModelCheckpoint:
    """
    Save the top-K models based on a monitored metric.

    - Saves best model weights to disk after each epoch
    - Keeps only the top K checkpoints, deletes worse ones
    - Tracks the overall best for easy restore
    """

    def __init__(self, dirpath=config.MODEL_DIR, filename_prefix="checkpoint",
                 monitor="val_acc", mode="max", save_top_k=config.CHECKPOINT_SAVE_TOP_K):
        self.dirpath = dirpath
        self.filename_prefix = filename_prefix
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k

        os.makedirs(dirpath, exist_ok=True)

        # Min-heap of (score, path) — for mode="max" we negate scores
        self._heap = []
        self.best_score = None
        self.best_path = None

    def _is_better(self, current, best):
        if self.mode == "max":
            return current > best
        return current < best

    def _score_key(self, score):
        return score if self.mode == "max" else -score

    def __call__(self, epoch, model, metrics):
        score = metrics[self.monitor]
        path = os.path.join(
            self.dirpath,
            f"{self.filename_prefix}_epoch{epoch:03d}_{self.monitor}={score:.4f}.pth",
        )

        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "metrics": metrics,
        }, path)

        # Track best
        if self.best_score is None or self._is_better(score, self.best_score):
            self.best_score = score
            self.best_path = path
            print(f"  [Checkpoint] New best {self.monitor}={score:.4f} saved to {os.path.basename(path)}")

        # Manage top-K
        heap_score = self._score_key(score)
        if len(self._heap) < self.save_top_k:
            heapq.heappush(self._heap, (heap_score, path))
        else:
            # If this score is better than the worst in heap, replace it
            worst_score, worst_path = self._heap[0]
            if heap_score > worst_score:
                heapq.heapreplace(self._heap, (heap_score, path))
                if os.path.exists(worst_path):
                    os.remove(worst_path)
                    print(f"  [Checkpoint] Removed old checkpoint: {os.path.basename(worst_path)}")
            else:
                # This checkpoint is worse than all top-K, delete it
                if os.path.exists(path) and path != self.best_path:
                    os.remove(path)

    def load_best(self, model, device=None):
        """Load the best checkpoint into the model."""
        if self.best_path and os.path.exists(self.best_path):
            checkpoint = torch.load(self.best_path, map_location=device, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"  [Checkpoint] Restored best model from {os.path.basename(self.best_path)} "
                  f"({self.monitor}={self.best_score:.4f})")
            return checkpoint["metrics"]
        return None


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    - Waits `patience` epochs for improvement beyond `min_delta`
    - Optionally restores the best model weights on stop
    """

    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE,
                 min_delta=config.MIN_DELTA, monitor="val_acc", mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.best_score = None
        self.counter = 0
        self.should_stop = False
        self.best_epoch = 0

    def _is_improvement(self, current, best):
        if self.mode == "max":
            return current > best + self.min_delta
        return current < best - self.min_delta

    def __call__(self, epoch, metrics):
        score = metrics[self.monitor]

        if self.best_score is None or self._is_improvement(score, self.best_score):
            self.best_score = score
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            remaining = self.patience - self.counter
            print(f"  [EarlyStopping] No improvement for {self.counter}/{self.patience} epochs "
                  f"(best {self.monitor}={self.best_score:.4f} at epoch {self.best_epoch})")

            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EarlyStopping] Stopping training. "
                      f"Best {self.monitor}={self.best_score:.4f} at epoch {self.best_epoch}")


class CSVLogger:
    """Log metrics to a CSV file after each epoch."""

    def __init__(self, log_dir=config.LOG_DIR, filename="training_log.csv"):
        os.makedirs(log_dir, exist_ok=True)
        self.filepath = os.path.join(log_dir, filename)
        self._header_written = False

    def __call__(self, epoch, metrics):
        row = {"epoch": epoch, **metrics}

        write_header = not os.path.exists(self.filepath)
        with open(self.filepath, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def get_history(self):
        """Read back the full training history."""
        if not os.path.exists(self.filepath):
            return []
        with open(self.filepath, "r") as f:
            reader = csv.DictReader(f)
            return [{k: float(v) if k != "epoch" else int(float(v))
                     for k, v in row.items()} for row in reader]


class LRTracker:
    """Track and log learning rates from optimizer param groups."""

    def __call__(self, optimizer):
        lrs = {}
        for i, group in enumerate(optimizer.param_groups):
            name = group.get("name", f"group_{i}")
            lrs[f"lr_{name}"] = group["lr"]
        return lrs


class CallbackRunner:
    """
    Orchestrates all callbacks in a single interface.

    Usage:
        runner = CallbackRunner(stage="proto")
        for epoch in ...:
            metrics = {train_loss, train_acc, val_acc, ...}
            runner.on_epoch_end(epoch, model, metrics, optimizer)
            if runner.should_stop:
                break
        runner.load_best_model(model, device)
    """

    def __init__(self, stage="proto"):
        prefix = f"{stage}"
        self.checkpoint = ModelCheckpoint(
            filename_prefix=prefix, monitor="val_acc", mode="max",
        )
        self.early_stopping = EarlyStopping(monitor="val_acc", mode="max")
        self.csv_logger = CSVLogger(filename=f"{stage}_log.csv")
        self.lr_tracker = LRTracker()
        self.stage = stage

    @property
    def should_stop(self):
        return self.early_stopping.should_stop

    def on_epoch_end(self, epoch, model, metrics, optimizer=None):
        # Add LR info to metrics
        if optimizer:
            lr_info = self.lr_tracker(optimizer)
            metrics = {**metrics, **lr_info}

        # Run callbacks
        self.csv_logger(epoch, metrics)
        self.checkpoint(epoch, model, metrics)
        self.early_stopping(epoch, metrics)

    def load_best_model(self, model, device=None):
        return self.checkpoint.load_best(model, device)
