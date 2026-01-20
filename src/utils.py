"""
Utility Functions

Helper functions used across the project for:
- Configuration loading
- Logging setup
- Data loading
- Reproducibility (random seeds)
- Device detection (CPU/GPU)
"""

import json
import logging
import os
import random
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import torch
import yaml


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the project.

    Args:
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR)

    Returns:
        Configured logger
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler()]

    if log_file:
        # Create logs directory if needed
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )

    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.

    Sets seeds for:
    - Python's random module
    - NumPy
    - PyTorch (CPU and GPU)

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For complete reproducibility (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(use_gpu: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch.

    Args:
        use_gpu: Whether to use GPU if available

    Returns:
        torch.device object
    """
    if use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def load_processed_data(data_dir: str = "data/processed") -> Dict[str, np.ndarray]:
    """
    Load preprocessed data from files.

    Args:
        data_dir: Path to processed data directory

    Returns:
        Dictionary with:
        - features: Numerical features matrix
        - token_sequences: Token ID sequences
        - labels: Bug labels
        - train_idx, val_idx, test_idx: Data split indices
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Processed data directory not found: {data_path}\n"
            "Please run preprocessing first: python -m src.preprocessing"
        )

    # Load arrays
    features = np.load(data_path / "features.npz")['features']
    token_sequences = np.load(data_path / "token_sequences.npz")['sequences']
    labels = np.load(data_path / "labels.npy")
    train_idx = np.load(data_path / "train_indices.npy")
    val_idx = np.load(data_path / "val_indices.npy")
    test_idx = np.load(data_path / "test_indices.npy")

    return {
        'features': features,
        'token_sequences': token_sequences,
        'labels': labels,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }


def load_metadata(data_dir: str = "data/processed") -> Dict[str, Any]:
    """
    Load preprocessing metadata.

    Args:
        data_dir: Path to processed data directory

    Returns:
        Metadata dictionary
    """
    metadata_path = Path(data_dir) / "metadata.json"

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    return metadata


def get_train_val_test_data(data_dir: str = "data/processed") -> Tuple:
    """
    Get train, validation, and test datasets.

    Args:
        data_dir: Path to processed data directory

    Returns:
        Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        where X is features and y is labels
    """
    data = load_processed_data(data_dir)

    features = data['features']
    labels = data['labels']
    train_idx = data['train_idx']
    val_idx = data['val_idx']
    test_idx = data['test_idx']

    X_train = features[train_idx]
    y_train = labels[train_idx]

    X_val = features[val_idx]
    y_val = labels[val_idx]

    X_test = features[test_idx]
    y_test = labels[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve.

    Usage:
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(epochs):
            val_loss = train_and_validate()
            if early_stopping(val_loss):
                print("Early stopping triggered")
                break
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0001):
        """
        Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.

        Args:
            val_loss: Current validation loss

        Returns:
            True if training should stop
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop

    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_loss = None
        self.should_stop = False


class AverageMeter:
    """
    Computes and stores the average and current value.

    Useful for tracking metrics during training.

    Usage:
        meter = AverageMeter()
        for batch in batches:
            loss = compute_loss()
            meter.update(loss, batch_size)
        print(f"Average loss: {meter.avg}")
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.

        Args:
            val: Value to add
            n: Weight (e.g., batch size)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    Pretty print metrics dictionary.

    Args:
        metrics: Dictionary of metric name -> value
        prefix: Optional prefix for output
    """
    if prefix:
        print(f"\n{prefix}")
        print("-" * 40)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")


def ensure_dir(path: str) -> Path:
    """
    Ensure directory exists, create if not.

    Args:
        path: Directory path

    Returns:
        Path object
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path
