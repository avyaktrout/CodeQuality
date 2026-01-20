"""
Neural Network Models for Bug Prediction

This module contains PyTorch neural network architectures for predicting
whether Python functions contain bugs.

Model Types:
1. FeatureOnlyModel: Uses only numerical features (32 dims)
2. HybridModel: Combines numerical features + token sequences
3. SimpleModel: Basic feedforward for testing

Architecture Overview:
----------------------
The HybridModel processes two types of inputs:

1. Numerical Features (32 dimensions):
   - Code metrics: LOC, parameters, complexity, nesting depth
   - AST features: loops, conditionals, function calls, etc.
   - Processed through dense layers

2. Token Sequences (200 tokens):
   - Tokenized code converted to integer IDs
   - Processed through embedding layer
   - Aggregated via global average pooling or LSTM

Both streams are concatenated and fed through classification layers.

Why This Architecture?
---------------------
- Numerical features capture measurable code properties
- Token sequences capture actual code patterns and semantics
- Combination provides both structural and semantic understanding
- Simpler than transformers, faster to train, easier to debug
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureOnlyModel(nn.Module):
    """
    Simple feedforward network using only numerical features.

    Good for:
    - Quick experiments
    - Baseline comparisons
    - When token sequences aren't available

    Architecture:
        Input (32) → Dense(64) → ReLU → Dropout
                  → Dense(32) → ReLU → Dropout
                  → Dense(1) → Sigmoid
    """

    def __init__(self, input_dim: int = 32, dropout: float = 0.3):
        """
        Initialize the model.

        Args:
            input_dim: Number of input features
            dropout: Dropout probability
        """
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            # Layer 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(dropout),

            # Layer 3
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            # Output
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Numerical features tensor (batch_size, input_dim)

        Returns:
            Bug probability tensor (batch_size, 1)
        """
        return self.network(features)


class HybridModel(nn.Module):
    """
    Hybrid model combining numerical features and token sequences.

    This model processes two types of inputs:
    1. Numerical features through dense layers
    2. Token sequences through embedding + aggregation

    The outputs are concatenated and classified.

    Architecture:
        Features (32) → Dense(64) → Dense(32) → Feature Embedding (32)
                                                         ↓
        Tokens (200) → Embedding(vocab, 64) → AvgPool → Token Embedding (64)
                                                         ↓
                                    Concatenate [32 + 64] = 96
                                                         ↓
                                    Dense(64) → Dense(32) → Dense(1) → Sigmoid
    """

    def __init__(self,
                 num_features: int = 32,
                 vocab_size: int = 5000,
                 embedding_dim: int = 64,
                 seq_length: int = 200,
                 dropout: float = 0.3,
                 use_lstm: bool = False):
        """
        Initialize the hybrid model.

        Args:
            num_features: Number of numerical features
            vocab_size: Size of token vocabulary
            embedding_dim: Dimension of token embeddings
            seq_length: Length of token sequences (for reference)
            dropout: Dropout probability
            use_lstm: Whether to use LSTM (True) or GlobalAvgPool (False)
        """
        super().__init__()

        self.use_lstm = use_lstm

        # Feature processing branch
        self.feature_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
        )

        # Token processing branch
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD token
        )

        if use_lstm:
            self.sequence_encoder = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=64,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
            token_output_dim = 64
        else:
            # Global average pooling
            self.sequence_encoder = None
            token_output_dim = embedding_dim

        # Combined classifier
        combined_dim = 32 + token_output_dim  # Features + Tokens

        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self,
                features: torch.Tensor,
                token_sequences: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: Numerical features (batch_size, num_features)
            token_sequences: Token IDs (batch_size, seq_length)

        Returns:
            Bug probability (batch_size, 1)
        """
        # Process numerical features
        feature_embedding = self.feature_encoder(features)  # (batch, 32)

        # Process token sequences
        token_embeds = self.embedding(token_sequences)  # (batch, seq_len, embed_dim)

        if self.use_lstm:
            # LSTM encoding
            lstm_out, (hidden, _) = self.sequence_encoder(token_embeds)
            token_embedding = hidden.squeeze(0)  # (batch, 64)
        else:
            # Global average pooling over sequence
            # Mask out padding tokens (0)
            mask = (token_sequences != 0).unsqueeze(-1).float()  # (batch, seq_len, 1)
            masked_embeds = token_embeds * mask
            token_embedding = masked_embeds.sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (batch, embed_dim)

        # Concatenate and classify
        combined = torch.cat([feature_embedding, token_embedding], dim=1)
        output = self.classifier(combined)

        return output


class SimpleModel(nn.Module):
    """
    Very simple model for testing and debugging.

    Single hidden layer, minimal complexity.
    Use this to verify the training pipeline works.
    """

    def __init__(self, input_dim: int = 32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(features))
        x = torch.sigmoid(self.fc2(x))
        return x


# =============================================================================
# MODEL FACTORY
# =============================================================================

def create_model(model_type: str = "feature_only",
                 num_features: int = 32,
                 vocab_size: int = 5000,
                 **kwargs) -> nn.Module:
    """
    Factory function to create models.

    Args:
        model_type: Type of model ("feature_only", "hybrid", "simple")
        num_features: Number of numerical features
        vocab_size: Vocabulary size for token embeddings
        **kwargs: Additional model-specific arguments

    Returns:
        PyTorch model instance
    """
    if model_type == "feature_only":
        return FeatureOnlyModel(input_dim=num_features, **kwargs)
    elif model_type == "hybrid":
        return HybridModel(
            num_features=num_features,
            vocab_size=vocab_size,
            **kwargs
        )
    elif model_type == "simple":
        return SimpleModel(input_dim=num_features)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# DATASET CLASS
# =============================================================================

class BugDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for bug prediction.

    Handles both feature-only and hybrid (features + tokens) modes.
    """

    def __init__(self,
                 features: torch.Tensor,
                 labels: torch.Tensor,
                 token_sequences: torch.Tensor = None):
        """
        Initialize dataset.

        Args:
            features: Numerical features (N, num_features)
            labels: Bug labels (N,)
            token_sequences: Optional token sequences (N, seq_length)
        """
        self.features = features
        self.labels = labels
        self.token_sequences = token_sequences
        self.has_sequences = token_sequences is not None

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        if self.has_sequences:
            return (
                self.features[idx],
                self.token_sequences[idx],
                self.labels[idx]
            )
        else:
            return (
                self.features[idx],
                self.labels[idx]
            )


def create_data_loaders(features,
                        labels,
                        train_idx,
                        val_idx,
                        test_idx,
                        token_sequences=None,
                        batch_size: int = 64):
    """
    Create train, validation, and test data loaders.

    Args:
        features: Feature matrix (N, num_features)
        labels: Label array (N,)
        train_idx: Training indices
        val_idx: Validation indices
        test_idx: Test indices
        token_sequences: Optional token sequences (N, seq_length)
        batch_size: Batch size

    Returns:
        train_loader, val_loader, test_loader
    """
    # Convert to tensors
    features = torch.FloatTensor(features)
    labels = torch.FloatTensor(labels).unsqueeze(1)

    if token_sequences is not None:
        token_sequences = torch.LongTensor(token_sequences)

    # Create datasets
    if token_sequences is not None:
        train_dataset = BugDataset(
            features[train_idx],
            labels[train_idx],
            token_sequences[train_idx]
        )
        val_dataset = BugDataset(
            features[val_idx],
            labels[val_idx],
            token_sequences[val_idx]
        )
        test_dataset = BugDataset(
            features[test_idx],
            labels[test_idx],
            token_sequences[test_idx]
        )
    else:
        train_dataset = BugDataset(features[train_idx], labels[train_idx])
        val_dataset = BugDataset(features[val_idx], labels[val_idx])
        test_dataset = BugDataset(features[test_idx], labels[test_idx])

    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True  # For BatchNorm stability
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test models with dummy data
    print("Testing models...")

    batch_size = 4
    num_features = 32
    vocab_size = 5000
    seq_length = 200

    # Dummy data
    features = torch.randn(batch_size, num_features)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_length))

    # Test FeatureOnlyModel
    print("\n1. FeatureOnlyModel")
    model1 = FeatureOnlyModel(input_dim=num_features)
    out1 = model1(features)
    print(f"   Input: {features.shape}")
    print(f"   Output: {out1.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model1.parameters()):,}")

    # Test HybridModel
    print("\n2. HybridModel (AvgPool)")
    model2 = HybridModel(num_features=num_features, vocab_size=vocab_size, use_lstm=False)
    out2 = model2(features, tokens)
    print(f"   Input: features {features.shape}, tokens {tokens.shape}")
    print(f"   Output: {out2.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model2.parameters()):,}")

    # Test HybridModel with LSTM
    print("\n3. HybridModel (LSTM)")
    model3 = HybridModel(num_features=num_features, vocab_size=vocab_size, use_lstm=True)
    out3 = model3(features, tokens)
    print(f"   Input: features {features.shape}, tokens {tokens.shape}")
    print(f"   Output: {out3.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model3.parameters()):,}")

    # Test SimpleModel
    print("\n4. SimpleModel")
    model4 = SimpleModel(input_dim=num_features)
    out4 = model4(features)
    print(f"   Input: {features.shape}")
    print(f"   Output: {out4.shape}")
    print(f"   Parameters: {sum(p.numel() for p in model4.parameters()):,}")

    print("\nAll models working!")
