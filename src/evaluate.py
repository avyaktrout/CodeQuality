"""
Evaluation Module for Bug Prediction Model

Provides:
- Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Error analysis
- CLI interface for testing code snippets

Usage:
    python -m src.evaluate --test              # Evaluate on test set
    python -m src.evaluate --code "def f(): return x/0"  # Test single function
    python -m src.evaluate --file mycode.py    # Analyze a file
"""

import argparse
import ast
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

from src.model import create_model, create_data_loaders
from src.preprocessing import CodeFeatureExtractor, CodeTokenizer
from src.utils import get_device, load_processed_data, load_metadata


class ModelEvaluator:
    """Evaluates bug prediction models."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device(False)  # Use CPU for evaluation
        self.models_dir = Path(self.config['paths']['models'])
        self.output_dir = Path(self.config['paths'].get('logs', 'logs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_model(self, model_path: Path, model_type: str = "feature_only",
                   num_features: int = 32, vocab_size: int = 5000):
        """Load trained model."""
        # Get LSTM settings from config
        model_config = self.config.get('model', {})
        use_lstm = model_config.get('use_lstm', False)
        embedding_dim = model_config.get('embedding_dim', 64)

        self.model = create_model(
            model_type, num_features, vocab_size,
            use_lstm=use_lstm, embedding_dim=embedding_dim
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Loaded {model_type} model from {model_path}")

    def get_predictions(self, data_loader, use_sequences: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions and true labels."""
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                if use_sequences:
                    features, tokens, labels = batch
                    features = features.to(self.device)
                    tokens = tokens.to(self.device)
                    outputs = self.model(features, tokens)
                else:
                    features, labels = batch
                    features = features.to(self.device)
                    outputs = self.model(features)

                all_probs.extend(outputs.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())

        return np.array(all_probs), np.array(all_labels)

    def calculate_metrics(self, y_true: np.ndarray, y_prob: np.ndarray,
                          threshold: float = 0.5) -> Dict:
        """Calculate all classification metrics."""
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'threshold': threshold
        }

        # ROC-AUC (only if both classes present)
        if len(np.unique(y_true)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        else:
            metrics['roc_auc'] = None

        return metrics

    def print_metrics(self, metrics: Dict, title: str = "Metrics"):
        """Print metrics in a nice format."""
        print(f"\n{title}")
        print("-" * 40)
        for key, value in metrics.items():
            if value is not None and isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            elif value is not None:
                print(f"  {key}: {value}")

    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                              save_path: Path = None):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        classes = ['Clean', 'Buggy']
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=classes, yticklabels=classes,
               xlabel='Predicted', ylabel='True',
               title='Confusion Matrix')

        # Add text annotations
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved confusion matrix to {save_path}")

        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                       save_path: Path = None):
        """Plot ROC curve."""
        if len(np.unique(y_true)) < 2:
            print("Cannot plot ROC curve: only one class in data")
            return

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved ROC curve to {save_path}")

        plt.show()

    def plot_training_history(self, history_path: Path, save_path: Path = None):
        """Plot training history (loss and accuracy curves)."""
        with open(history_path, 'r') as f:
            history = json.load(f)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Loss curves
        ax1 = axes[0]
        ax1.plot(history['train_loss'], label='Train Loss')
        ax1.plot(history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy curve
        ax2 = axes[1]
        ax2.plot(history['val_accuracy'], label='Val Accuracy', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved training history plot to {save_path}")

        plt.show()

    def evaluate(self, model_type: str = "feature_only") -> Dict:
        """Run full evaluation pipeline."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)

        use_sequences = model_type == "hybrid"

        # Load data
        print("\nLoading data...")
        data = load_processed_data(self.config['paths']['data_processed'])
        metadata = load_metadata(self.config['paths']['data_processed'])

        train_loader, val_loader, test_loader = create_data_loaders(
            features=data['features'],
            labels=data['labels'],
            train_idx=data['train_idx'],
            val_idx=data['val_idx'],
            test_idx=data['test_idx'],
            token_sequences=data['token_sequences'] if use_sequences else None,
            batch_size=64
        )

        # Load model
        model_path = self.models_dir / "best_model.pth"
        if not model_path.exists():
            print(f"Model not found: {model_path}")
            print("Please train a model first: python -m src.train")
            return {}

        self.load_model(
            model_path,
            model_type=model_type,
            num_features=metadata['n_features'],
            vocab_size=metadata.get('vocab_size', 5000)
        )

        # Get predictions
        print("\nGetting predictions...")
        y_prob, y_true = self.get_predictions(test_loader, use_sequences)
        y_pred = (y_prob >= 0.5).astype(int)

        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_prob)
        self.print_metrics(metrics, "Test Set Metrics")

        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['Clean', 'Buggy']))

        # Visualizations
        print("\nGenerating visualizations...")

        # Confusion matrix
        self.plot_confusion_matrix(
            y_true, y_pred,
            save_path=self.output_dir / "confusion_matrix.png"
        )

        # ROC curve
        if len(np.unique(y_true)) > 1:
            self.plot_roc_curve(
                y_true, y_prob,
                save_path=self.output_dir / "roc_curve.png"
            )

        # Training history
        history_path = self.models_dir / "training_history.json"
        if history_path.exists():
            self.plot_training_history(
                history_path,
                save_path=self.output_dir / "training_history.png"
            )

        # Save metrics
        with open(self.output_dir / "evaluation_metrics.json", 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to {self.output_dir / 'evaluation_metrics.json'}")

        return metrics


class CodePredictor:
    """Predicts bug probability for individual code snippets."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize predictor with trained model."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device(False)  # Use CPU for inference
        self.model = None
        self.extractor = None
        self.tokenizer = None
        self.model_type = self.config['model'].get('model_type', 'feature_only')
        self._load_components()

    def _load_components(self):
        """Load model, feature extractor, and tokenizer."""
        # Load metadata
        processed_dir = Path(self.config['paths']['data_processed'])
        metadata_path = processed_dir / "metadata.json"

        num_features = 32
        vocab_size = 5000
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                num_features = metadata.get('n_features', 32)
                vocab_size = metadata.get('vocab_size', 5000)

        # Load model
        model_config = self.config['model']
        use_lstm = model_config.get('use_lstm', False)
        embedding_dim = model_config.get('embedding_dim', 64)

        self.model = create_model(
            model_type=self.model_type,
            num_features=num_features,
            vocab_size=vocab_size,
            use_lstm=use_lstm,
            embedding_dim=embedding_dim
        )

        model_path = Path(self.config['paths']['models']) / "best_model.pth"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Load feature extractor
        self.extractor = CodeFeatureExtractor()
        scaler_path = processed_dir / "scaler_params.json"
        if scaler_path.exists():
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            self.extractor.scaler_mean = np.array(scaler_params['mean'])
            self.extractor.scaler_std = np.array(scaler_params['scale'])
        else:
            self.extractor.scaler_mean = None
            self.extractor.scaler_std = None

        # Load tokenizer for hybrid model
        if self.model_type == "hybrid":
            vocab_path = processed_dir / "vocabulary.json"
            self.tokenizer = CodeTokenizer(
                vocab_size=self.config['preprocessing'].get('vocab_size', 5000),
                max_length=self.config['preprocessing'].get('max_length', 200)
            )
            if vocab_path.exists():
                with open(vocab_path, 'r') as f:
                    vocab_data = json.load(f)
                if 'token_to_id' in vocab_data:
                    vocab = vocab_data['token_to_id']
                else:
                    vocab = vocab_data
                self.tokenizer.token_to_id = vocab
                self.tokenizer.id_to_token = {v: k for k, v in vocab.items()}
                self.tokenizer.is_fitted = True

    def predict(self, code: str) -> Dict:
        """
        Predict bug probability for Python code.

        Args:
            code: Python function code as string

        Returns:
            Dictionary with prediction results
        """
        # Validate syntax
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                'error': f"Syntax error: {e}",
                'probability': None,
                'prediction': None
            }

        # Extract features
        features_dict = self.extractor.extract_all_features(code)
        if features_dict is None:
            return {
                'error': "Failed to extract features",
                'probability': None,
                'prediction': None
            }

        # Convert to feature vector
        feature_names = self.extractor.get_feature_names()
        features = np.array([features_dict.get(name, 0) for name in feature_names])

        # Normalize
        if self.extractor.scaler_mean is not None:
            features = (features - self.extractor.scaler_mean) / (self.extractor.scaler_std + 1e-8)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict with ML model
        with torch.no_grad():
            if self.model_type == "hybrid" and self.tokenizer is not None:
                tokens = self.tokenizer.transform([code])
                tokens_tensor = torch.LongTensor(tokens).to(self.device)
                ml_probability = self.model(features_tensor, tokens_tensor).item()
            else:
                ml_probability = self.model(features_tensor).item()

        # Rule-based detection for known bug patterns
        rule_boost = self._detect_bug_patterns(features_dict, code)

        # Combine ML prediction with rule-based detection
        probability = min(0.95, ml_probability + rule_boost)

        return {
            'probability': probability,
            'prediction': 'BUGGY' if probability >= 0.5 else 'CLEAN',
            'confidence': abs(probability - 0.5) * 2,
            'features': features_dict
        }

    def _detect_bug_patterns(self, features_dict: Dict, code: str = "") -> float:
        """
        Rule-based detection for known Python bug patterns.
        Returns a probability boost (0.0 to 0.7) based on detected patterns.
        """
        import re
        boost = 0.0

        # Mutable default arguments (classic Python gotcha)
        if features_dict.get('num_mutable_defaults', 0) > 0:
            boost += 0.45

        # eval/exec usage (security risk)
        if features_dict.get('num_eval_exec', 0) > 0:
            boost += 0.40

        # Bare except (swallows all errors including KeyboardInterrupt)
        if features_dict.get('num_bare_except', 0) > 0:
            boost += 0.35

        # Deep nesting (code smell, hard to maintain)
        if features_dict.get('max_nesting_depth', 0) >= 4:
            boost += 0.25

        # High cyclomatic complexity
        if features_dict.get('cyclomatic_complexity', 0) >= 10:
            boost += 0.20

        # Global variable usage
        if features_dict.get('num_global', 0) > 0:
            boost += 0.15

        # Code-based pattern detection for semantic bugs
        if code:
            # Off-by-one patterns: range(n + 1) or range(n - 1) with array access
            if re.search(r'range\s*\(\s*\w+\s*[+-]\s*1\s*\)', code):
                if re.search(r'\[\s*\w+\s*\]', code):  # Has array indexing
                    boost += 0.40

            # Missing null check: method calls on parameters without None check
            if re.search(r'\.strip\s*\(|\.lower\s*\(|\.upper\s*\(|\.split\s*\(', code):
                if not re.search(r'if\s+\w+\s*:|if\s+not\s+\w+|is\s+not\s+None|is\s+None|if\s+\w+\s+is|if\s+\w+\s*(!|=)=', code):
                    boost += 0.35

            # Potential index error: accessing [0] or [-1] without length check
            if re.search(r'\[\s*0\s*\]|\[\s*-1\s*\]', code):
                if not re.search(r'if\s+.*len\s*\(|if\s+not\s+\w+\s*:|if\s+\w+\s*:', code):
                    boost += 0.25

            # Division without zero check (only flag division by variables, not constants like // 2)
            # Match "/ variable" but not "// anything" (integer division)
            if re.search(r'(?<!/)/\s*[a-zA-Z_]\w*(?!\s*/)', code):
                if not re.search(r'if\s+\w+\s*[=!]=\s*0|if\s+\w+\s*:|if\s+not\s+\w+|if\s+\w+\s*>', code):
                    boost += 0.35

        return min(0.7, boost)  # Cap the boost

    def analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze all functions in a Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return [{'error': f"Syntax error in file: {e}"}]

        results = []
        lines = content.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20

                func_lines = lines[start_line:end_line]
                func_code = '\n'.join(func_lines)

                result = self.predict(func_code)
                result['function_name'] = node.name
                result['line_number'] = node.lineno
                results.append(result)

        return results


def print_prediction(result: Dict, show_features: bool = False):
    """Print prediction result."""
    print("\n" + "=" * 60)

    if 'error' in result and result.get('probability') is None:
        print(f"ERROR: {result['error']}")
        return

    prob = result['probability']
    pred = result['prediction']
    conf = result.get('confidence', 0)

    # Color coding
    if pred == 'BUGGY':
        color = '\033[91m'  # Red
    else:
        color = '\033[92m'  # Green
    reset = '\033[0m'

    print(f"Prediction: {color}{pred}{reset}")
    print(f"Bug Probability: {prob:.2%}")
    print(f"Confidence: {conf:.2%}")

    if result.get('function_name'):
        print(f"Function: {result['function_name']} (line {result.get('line_number', '?')})")

    if show_features and result.get('features'):
        print("\nExtracted Features:")
        for name, value in sorted(result['features'].items()):
            if value != 0:
                print(f"  {name}: {value}")

    print("=" * 60)


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate bug prediction model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.evaluate --test                    # Evaluate on test set
  python -m src.evaluate --code "def f(): return x/0"   # Test single function
  python -m src.evaluate --file mycode.py          # Analyze a file
  python -m src.evaluate --code "..." --features   # Show extracted features
        """
    )

    parser.add_argument('--test', '-t', action='store_true',
                        help='Run evaluation on test set')
    parser.add_argument('--code', '-c', type=str,
                        help='Python code string to analyze')
    parser.add_argument('--file', '-f', type=str,
                        help='Python file to analyze')
    parser.add_argument('--features', action='store_true',
                        help='Show extracted features')

    args = parser.parse_args()

    if args.test:
        # Run full test set evaluation
        evaluator = ModelEvaluator(config_path="config.yaml")
        model_type = evaluator.config['model'].get('model_type', 'feature_only')
        print(f"Evaluating {model_type.upper()} model on test set...")
        evaluator.evaluate(model_type=model_type)

    elif args.code:
        # Analyze single code snippet
        predictor = CodePredictor(config_path="config.yaml")
        print(f"\nAnalyzing code snippet...")
        print("-" * 60)
        print(args.code)
        result = predictor.predict(args.code)
        print_prediction(result, show_features=args.features)

    elif args.file:
        # Analyze file
        predictor = CodePredictor(config_path="config.yaml")
        print(f"\nAnalyzing file: {args.file}")
        results = predictor.analyze_file(args.file)

        buggy_count = 0
        for result in results:
            print_prediction(result, show_features=args.features)
            if result.get('prediction') == 'BUGGY':
                buggy_count += 1

        print(f"\nSummary: {buggy_count}/{len(results)} functions flagged as potentially buggy")

    else:
        # Default: show help
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick test - analyzing a sample function:")
        print("=" * 60)

        # Demo with a sample function
        try:
            predictor = CodePredictor(config_path="config.yaml")
            sample_code = '''def process_items(items):
    for i in range(len(items) + 1):  # Off-by-one error
        print(items[i])
'''
            print(sample_code)
            result = predictor.predict(sample_code)
            print_prediction(result)
        except FileNotFoundError:
            print("No trained model found. Run 'python -m src.train' first.")


if __name__ == "__main__":
    main()
