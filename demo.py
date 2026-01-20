"""
Bug Prediction Demo

Interactive command-line tool for predicting whether Python functions contain bugs.

Usage:
    python demo.py                    # Interactive mode
    python demo.py --file mycode.py   # Analyze a file
    python demo.py --example          # Show example predictions

Features:
- Predicts bug probability for Python functions
- Shows top contributing features
- Supports interactive input, file analysis, or examples
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.model import create_model
from src.preprocessing import CodeFeatureExtractor, CodeTokenizer
from src.utils import get_device


class BugPredictor:
    """Predicts bugs in Python functions using trained model."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize predictor with trained model."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device(False)  # Use CPU for inference
        self.model = None
        self.extractor = None
        self.tokenizer = None
        self.model_type = self.config['model'].get('model_type', 'feature_only')
        self._load_model()
        self._load_extractor()
        if self.model_type == "hybrid":
            self._load_tokenizer()

    def _load_model(self):
        """Load the trained model."""
        model_path = Path(self.config['paths']['models']) / "best_model.pth"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train a model first: python -m src.train"
            )

        # Load metadata to get feature count
        processed_dir = Path(self.config['paths']['data_processed'])
        metadata_path = processed_dir / "metadata.json"

        num_features = 32  # Default
        vocab_size = 5000
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                num_features = metadata.get('n_features', 32)
                vocab_size = metadata.get('vocab_size', 5000)

        # Get model settings from config
        model_config = self.config['model']
        use_lstm = model_config.get('use_lstm', False)
        embedding_dim = model_config.get('embedding_dim', 64)

        # Create and load model
        self.model = create_model(
            model_type=self.model_type,
            num_features=num_features,
            vocab_size=vocab_size,
            use_lstm=use_lstm,
            embedding_dim=embedding_dim
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded {self.model_type} model from {model_path}")
        if self.model_type == "hybrid":
            print(f"  LSTM enabled: {use_lstm}")

    def _load_tokenizer(self):
        """Load the tokenizer with fitted vocabulary."""
        processed_dir = Path(self.config['paths']['data_processed'])
        vocab_path = processed_dir / "vocabulary.json"

        vocab_size = self.config['preprocessing'].get('vocab_size', 5000)
        max_length = self.config['preprocessing'].get('max_length', 200)

        self.tokenizer = CodeTokenizer(vocab_size=vocab_size, max_length=max_length)

        if vocab_path.exists():
            import json
            with open(vocab_path, 'r') as f:
                vocab_data = json.load(f)
            # Handle nested vocabulary structure
            if 'token_to_id' in vocab_data:
                vocab = vocab_data['token_to_id']
            else:
                vocab = vocab_data
            self.tokenizer.token_to_id = vocab
            self.tokenizer.id_to_token = {v: k for k, v in vocab.items()}
            self.tokenizer.is_fitted = True
            print(f"Loaded vocabulary ({len(vocab)} tokens)")
        else:
            print("Warning: No vocabulary found, hybrid model may not work correctly")

    def _load_extractor(self):
        """Load the feature extractor with fitted scaler."""
        processed_dir = Path(self.config['paths']['data_processed'])
        scaler_path = processed_dir / "scaler_params.json"

        self.extractor = CodeFeatureExtractor()

        # Load scaler if available
        if scaler_path.exists():
            import json
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)
            self.extractor.scaler_mean = np.array(scaler_params['mean'])
            self.extractor.scaler_std = np.array(scaler_params['scale'])
            print("Loaded feature scaler")
        else:
            print("Warning: No scaler found, predictions may be less accurate")
            self.extractor.scaler_mean = None
            self.extractor.scaler_std = None

    def predict(self, code: str) -> Dict:
        """
        Predict bug probability for a Python function.

        Args:
            code: Python function code as string

        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features_dict = self.extractor.extract_all_features(code)

        if features_dict is None:
            return {
                'error': "Failed to parse code. Make sure it's valid Python.",
                'probability': None,
                'prediction': None
            }

        # Convert to feature vector
        feature_names = self.extractor.get_feature_names()
        features = np.array([features_dict.get(name, 0) for name in feature_names])

        # Normalize if scaler available
        if self.extractor.scaler_mean is not None:
            features = (features - self.extractor.scaler_mean) / (self.extractor.scaler_std + 1e-8)

        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

        # Predict based on model type
        with torch.no_grad():
            if self.model_type == "hybrid" and self.tokenizer is not None:
                # Tokenize code for hybrid model
                tokens = self.tokenizer.transform([code])
                tokens_tensor = torch.LongTensor(tokens).to(self.device)
                probability = self.model(features_tensor, tokens_tensor).item()
            else:
                # Feature-only model
                probability = self.model(features_tensor).item()

        # Get top contributing features
        top_features = self._get_top_features(features_dict, feature_names)

        return {
            'probability': probability,
            'prediction': 'BUGGY' if probability >= 0.5 else 'CLEAN',
            'confidence': abs(probability - 0.5) * 2,  # 0 to 1 scale
            'top_features': top_features,
            'all_features': features_dict
        }

    def _get_top_features(self, features_dict: Dict, feature_names: List[str],
                          top_n: int = 5) -> List[Tuple[str, float]]:
        """Get top features that might indicate bugs."""
        # Features that typically correlate with bugs
        bug_indicators = {
            'cyclomatic_complexity': 1.5,
            'max_nesting_depth': 2.0,
            'num_nested_functions': 1.5,
            'num_bare_except': 3.0,
            'num_mutable_defaults': 3.0,
            'num_global': 2.0,
            'num_eval_exec': 3.0,
            'num_assertions': -0.5,  # Assertions are good
            'has_docstring': -1.0,   # Docstrings are good
            'num_comments': -0.5,    # Comments are good
        }

        scored_features = []
        for name in feature_names:
            value = features_dict.get(name, 0)
            weight = bug_indicators.get(name, 1.0)

            # Higher values with positive weights indicate potential bugs
            if value > 0:
                score = value * weight
                scored_features.append((name, value, score))

        # Sort by score descending
        scored_features.sort(key=lambda x: x[2], reverse=True)

        # Return top features with their values
        return [(name, value) for name, value, _ in scored_features[:top_n]]

    def analyze_file(self, file_path: str) -> List[Dict]:
        """Analyze all functions in a Python file."""
        import ast

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
                # Extract function code
                start_line = node.lineno - 1
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 20

                func_lines = lines[start_line:end_line]
                func_code = '\n'.join(func_lines)

                result = self.predict(func_code)
                result['function_name'] = node.name
                result['line_number'] = node.lineno
                results.append(result)

        return results


def print_prediction(result: Dict, code: str = None):
    """Pretty print prediction result."""
    print("\n" + "=" * 60)

    if 'error' in result and result.get('probability') is None:
        print(f"ERROR: {result['error']}")
        return

    prob = result['probability']
    pred = result['prediction']
    conf = result['confidence']

    # Color coding (ANSI escape codes)
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

    if result.get('top_features'):
        print("\nTop Contributing Features:")
        for name, value in result['top_features']:
            print(f"  - {name}: {value}")

    print("=" * 60)


def run_examples():
    """Run predictions on example functions."""
    predictor = BugPredictor()

    examples = [
        # Clean function
        ("""
def calculate_average(numbers):
    \"\"\"Calculate the average of a list of numbers.\"\"\"
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)
""", "Clean function with validation"),

        # Buggy: Off-by-one error
        ("""
def get_last_items(items, n):
    result = []
    for i in range(n + 1):  # Bug: should be range(n)
        result.append(items[i])
    return result
""", "Off-by-one error in loop"),

        # Buggy: Missing null check
        ("""
def process_data(data):
    result = data.strip().lower()
    parts = result.split(',')
    return parts[0]
""", "Missing null check"),

        # Buggy: Mutable default argument
        ("""
def append_to_list(item, items=[]):
    items.append(item)
    return items
""", "Mutable default argument"),

        # Clean: Well-structured function
        ("""
def binary_search(arr, target):
    \"\"\"Binary search for target in sorted array.\"\"\"
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1
""", "Well-structured binary search"),

        # Buggy: Deep nesting
        ("""
def complex_logic(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    for i in range(a):
                        for j in range(b):
                            if i == j:
                                return i * j
    return 0
""", "Deeply nested logic"),
    ]

    print("\n" + "=" * 60)
    print("BUG PREDICTION EXAMPLES")
    print("=" * 60)

    for code, description in examples:
        print(f"\n>>> {description}")
        print("-" * 40)
        print(code.strip())

        result = predictor.predict(code)
        print_prediction(result)
        print()


def interactive_mode():
    """Run interactive prediction mode."""
    predictor = BugPredictor()

    print("\n" + "=" * 60)
    print("INTERACTIVE BUG PREDICTION")
    print("=" * 60)
    print("\nPaste your Python function below.")
    print("Enter a blank line followed by 'END' to submit.")
    print("Type 'quit' to exit.\n")

    while True:
        print("-" * 40)
        print("Enter Python function (or 'quit'):")

        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                break

            if line.strip().lower() == 'quit':
                print("Goodbye!")
                return

            if line.strip().upper() == 'END':
                break

            lines.append(line)

        if not lines:
            continue

        code = '\n'.join(lines)
        result = predictor.predict(code)
        print_prediction(result, code)
        print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Predict bugs in Python functions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                     # Interactive mode
  python demo.py --example           # Show example predictions
  python demo.py --file mycode.py    # Analyze a file
  python demo.py --code "def f(x): return x+1"  # Single function
        """
    )

    parser.add_argument('--file', '-f', type=str,
                        help='Python file to analyze')
    parser.add_argument('--code', '-c', type=str,
                        help='Python code string to analyze')
    parser.add_argument('--example', '-e', action='store_true',
                        help='Show example predictions')

    args = parser.parse_args()

    if args.example:
        run_examples()
    elif args.file:
        predictor = BugPredictor()
        results = predictor.analyze_file(args.file)

        print(f"\nAnalyzing: {args.file}")
        print("=" * 60)

        buggy_count = 0
        for result in results:
            print_prediction(result)
            if result.get('prediction') == 'BUGGY':
                buggy_count += 1

        print(f"\nSummary: {buggy_count}/{len(results)} functions flagged as potentially buggy")

    elif args.code:
        predictor = BugPredictor()
        result = predictor.predict(args.code)
        print_prediction(result, args.code)

    else:
        interactive_mode()


if __name__ == "__main__":
    main()
