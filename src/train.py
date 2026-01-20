"""
Training Module for Bug Prediction Model

Features:
- Training loop with early stopping
- Learning rate scheduling
- Model checkpointing
- Progress logging
- Support for both feature-only and hybrid models
"""

import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from src.model import create_model, create_data_loaders
from src.utils import (
    set_seed, get_device, load_processed_data,
    EarlyStopping, AverageMeter, format_time
)


class Trainer:
    """Training manager for bug prediction models."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.train_config = self.config['training']
        self.model_config = self.config['model']

        # Setup
        set_seed(self.train_config.get('random_seed', 42))
        self.device = get_device(self.train_config.get('use_gpu', True))

        # Paths
        self.models_dir = Path(self.config['paths']['models'])
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.models_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # Will be set during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None

    def load_data(self, use_sequences: bool = True) -> Tuple:
        """Load preprocessed data and create data loaders."""
        data = load_processed_data(self.config['paths']['data_processed'])

        features = data['features']
        labels = data['labels']
        train_idx = data['train_idx']
        val_idx = data['val_idx']
        test_idx = data['test_idx']

        token_sequences = data['token_sequences'] if use_sequences else None

        train_loader, val_loader, test_loader = create_data_loaders(
            features=features,
            labels=labels,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            token_sequences=token_sequences,
            batch_size=self.train_config.get('batch_size', 64)
        )

        return train_loader, val_loader, test_loader

    def setup_model(self, model_type: str = "feature_only",
                    num_features: int = 32,
                    vocab_size: int = 5000):
        """Create model, optimizer, scheduler, and criterion."""
        # Get LSTM settings from config
        use_lstm = self.model_config.get('use_lstm', False)
        embedding_dim = self.model_config.get('embedding_dim', 64)

        self.model = create_model(
            model_type=model_type,
            num_features=num_features,
            vocab_size=vocab_size,
            dropout=self.model_config.get('dropout_rates', [0.3])[0],
            use_lstm=use_lstm,
            embedding_dim=embedding_dim
        )
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.train_config.get('learning_rate', 0.001),
            weight_decay=self.train_config.get('weight_decay', 0.0001)
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.train_config.get('scheduler_factor', 0.5),
            patience=self.train_config.get('scheduler_patience', 3)
        )

        self.criterion = nn.BCELoss()

        # Print model info
        params = sum(p.numel() for p in self.model.parameters())
        print(f"Model: {model_type}")
        print(f"Parameters: {params:,}")
        print(f"Device: {self.device}")

    def train_epoch(self, train_loader, use_sequences: bool = False) -> float:
        """Train for one epoch."""
        self.model.train()
        loss_meter = AverageMeter()

        for batch in train_loader:
            if use_sequences:
                features, tokens, labels = batch
                features = features.to(self.device)
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features, tokens)
            else:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(features)

            loss = self.criterion(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), labels.size(0))

        return loss_meter.avg

    def validate(self, val_loader, use_sequences: bool = False) -> Tuple[float, float]:
        """Validate model and return loss and accuracy."""
        self.model.eval()
        loss_meter = AverageMeter()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                if use_sequences:
                    features, tokens, labels = batch
                    features = features.to(self.device)
                    tokens = tokens.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features, tokens)
                else:
                    features, labels = batch
                    features = features.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(features)

                loss = self.criterion(outputs, labels)
                loss_meter.update(loss.item(), labels.size(0))

                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0
        return loss_meter.avg, accuracy

    def train(self,
              model_type: str = "feature_only",
              num_features: int = 32,
              vocab_size: int = 5000,
              epochs: int = None) -> Dict:
        """
        Full training loop.

        Returns:
            Dictionary with training history and best metrics
        """
        print("\n" + "="*60)
        print("TRAINING BUG PREDICTION MODEL")
        print("="*60 + "\n")

        use_sequences = model_type == "hybrid"

        # Load data
        print("Loading data...")
        train_loader, val_loader, test_loader = self.load_data(use_sequences)
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")

        # Setup model
        print("\nSetting up model...")
        self.setup_model(model_type, num_features, vocab_size)

        # Training settings
        epochs = epochs or self.train_config.get('epochs', 50)
        patience = self.train_config.get('early_stopping_patience', 5)
        early_stopping = EarlyStopping(patience=patience)

        # History
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
        best_val_loss = float('inf')
        best_epoch = 0

        # Training loop
        print(f"\nStarting training for {epochs} epochs...")
        print("-" * 60)
        start_time = time.time()

        for epoch in range(epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(train_loader, use_sequences)

            # Validate
            val_loss, val_acc = self.validate(val_loader, use_sequences)

            # Update scheduler
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_acc)
            history['learning_rate'].append(current_lr)

            # Check for best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                self.save_model(self.models_dir / "best_model.pth")

            # Print progress
            epoch_time = time.time() - epoch_start
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.1f}s")

            # Early stopping
            if early_stopping(val_loss):
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        total_time = time.time() - start_time
        print("-" * 60)
        print(f"Training complete in {format_time(total_time)}")
        print(f"Best model at epoch {best_epoch} (val_loss: {best_val_loss:.4f})")

        # Load best model and evaluate on test set
        print("\nEvaluating on test set...")
        self.load_model(self.models_dir / "best_model.pth")
        test_loss, test_acc = self.validate(test_loader, use_sequences)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")

        # Save training history
        history['best_epoch'] = best_epoch
        history['best_val_loss'] = best_val_loss
        history['test_loss'] = test_loss
        history['test_accuracy'] = test_acc
        history['total_time'] = total_time

        with open(self.models_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)

        print(f"\nModel saved to: {self.models_dir / 'best_model.pth'}")
        print(f"History saved to: {self.models_dir / 'training_history.json'}")

        return history

    def save_model(self, path: Path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load_model(self, path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    trainer = Trainer(config_path="config.yaml")

    # Get model type from config
    model_type = trainer.model_config.get('model_type', 'feature_only')

    # Get num_features from metadata (generated by preprocessing)
    import json
    metadata_path = Path(trainer.config['paths']['data_processed']) / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        num_features = metadata.get('n_features', 32)
    else:
        num_features = trainer.model_config.get('num_features', 32)

    vocab_size = trainer.config['preprocessing'].get('vocab_size', 5000)
    epochs = trainer.train_config.get('epochs', 50)

    print(f"\nTraining {model_type.upper()} model...")
    if model_type == "hybrid":
        use_lstm = trainer.model_config.get('use_lstm', False)
        print(f"LSTM enabled: {use_lstm}")

    # Train model
    history = trainer.train(
        model_type=model_type,
        num_features=num_features,
        vocab_size=vocab_size,
        epochs=epochs
    )

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
