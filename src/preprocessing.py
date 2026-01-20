"""
Preprocessing Module

This module transforms raw Python function code into features for neural network training.

Feature Types:
1. Basic Code Features (5): LOC, parameters, complexity, function calls, nesting
2. AST-Based Features (15+): Structural patterns from abstract syntax trees
3. Text Tokenization: Convert code to token sequences

Why Both Numerical Features AND Text Tokens?
--------------------------------------------
Numerical features capture:
- Structural properties (complexity, size)
- Known bug patterns (deep nesting, many branches)
- Interpretable metrics for analysis

Text tokens capture:
- Actual code semantics and patterns
- Variable naming conventions
- API usage patterns
- Context that metrics miss

Together, the neural network learns from both high-level structure AND low-level details.

What is Cyclomatic Complexity?
------------------------------
Measures the number of independent paths through code.

Simple calculation: Count decision points + 1
- Each if, elif, for, while, and, or, except adds +1
- Base complexity = 1

Example:
    def example(x):           # Base: 1
        if x > 0:             # +1 = 2
            for i in range(x): # +1 = 3
                if i % 2:     # +1 = 4
                    print(i)
        return x
    # Cyclomatic complexity = 4

Higher complexity = more paths to test = more potential bugs.

Why Padding is Necessary?
-------------------------
Neural networks require fixed-size inputs:
- Matrix operations need consistent dimensions
- Batch processing requires same-sized tensors

Solution:
- Short functions: Pad with special <PAD> token (ID=0)
- Long functions: Truncate to first N tokens
- All functions become same-shaped tensors
"""

import ast
import json
import re
import tokenize
from collections import Counter
from io import StringIO
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# =============================================================================
# AST VISITORS FOR FEATURE EXTRACTION
# =============================================================================

class ComplexityVisitor(ast.NodeVisitor):
    """
    Calculates cyclomatic complexity by counting decision points.

    Cyclomatic Complexity = 1 + (number of decision points)

    Decision points:
    - if, elif (not else - it doesn't add a new path)
    - for, while loops
    - and, or in boolean expressions
    - except handlers
    - conditional expressions (ternary)
    - assert statements
    """

    def __init__(self):
        self.complexity = 1  # Base complexity

    def visit_If(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        self.complexity += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        # Each 'and' or 'or' adds complexity
        # n operands means n-1 operators
        self.complexity += len(node.values) - 1
        self.generic_visit(node)

    def visit_IfExp(self, node):
        # Ternary expression: x if condition else y
        self.complexity += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.complexity += 1
        self.generic_visit(node)


class NestingDepthVisitor(ast.NodeVisitor):
    """
    Calculates maximum nesting depth of code blocks.

    Deep nesting is often a code smell and bug indicator.
    """

    def __init__(self):
        self.max_depth = 0
        self.current_depth = 0

    def _enter_block(self):
        self.current_depth += 1
        self.max_depth = max(self.max_depth, self.current_depth)

    def _exit_block(self):
        self.current_depth -= 1

    def visit_If(self, node):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_For(self, node):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_While(self, node):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_Try(self, node):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()

    def visit_With(self, node):
        self._enter_block()
        self.generic_visit(node)
        self._exit_block()


class ASTFeatureVisitor(ast.NodeVisitor):
    """
    Extracts detailed AST-based features from code.

    Counts various node types that may indicate code quality:
    - Assignments, loops, conditionals
    - Function calls, comparisons
    - Exception handling, assertions
    - List comprehensions, lambda expressions
    - Bug patterns (mutable defaults, bare except, eval/exec)
    """

    def __init__(self, function_name: str = None):
        self.function_name = function_name
        self.features = {
            'num_assignments': 0,
            'num_aug_assignments': 0,  # +=, -=, etc.
            'num_for_loops': 0,
            'num_while_loops': 0,
            'num_if_statements': 0,
            'num_function_calls': 0,
            'num_method_calls': 0,
            'num_comparisons': 0,
            'num_boolean_ops': 0,
            'num_binary_ops': 0,
            'num_unary_ops': 0,
            'num_try_except': 0,
            'num_raise': 0,
            'num_assertions': 0,
            'num_returns': 0,
            'num_yields': 0,
            'num_list_comps': 0,
            'num_dict_comps': 0,
            'num_set_comps': 0,
            'num_generator_exps': 0,
            'num_lambda': 0,
            'num_subscripts': 0,  # array[index] access
            'num_attributes': 0,  # obj.attribute access
            'has_recursion': 0,   # Calls itself
            'num_string_formats': 0,  # f-strings, .format()
            # Bug pattern indicators
            'num_bare_except': 0,  # except: without exception type
            'num_eval_exec': 0,    # eval() or exec() calls
            'num_global': 0,       # global variable declarations
        }

    def visit_Assign(self, node):
        self.features['num_assignments'] += 1
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        self.features['num_aug_assignments'] += 1
        self.generic_visit(node)

    def visit_For(self, node):
        self.features['num_for_loops'] += 1
        self.generic_visit(node)

    def visit_While(self, node):
        self.features['num_while_loops'] += 1
        self.generic_visit(node)

    def visit_If(self, node):
        self.features['num_if_statements'] += 1
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check if it's a method call (obj.method()) or function call
        if isinstance(node.func, ast.Attribute):
            self.features['num_method_calls'] += 1
            # Check for .format() string formatting
            if node.func.attr == 'format':
                self.features['num_string_formats'] += 1
        elif isinstance(node.func, ast.Name):
            self.features['num_function_calls'] += 1
            # Check for recursion
            if self.function_name and node.func.id == self.function_name:
                self.features['has_recursion'] = 1
            # Check for eval/exec (security risk)
            if node.func.id in ('eval', 'exec'):
                self.features['num_eval_exec'] += 1
        self.generic_visit(node)

    def visit_Compare(self, node):
        self.features['num_comparisons'] += 1
        self.generic_visit(node)

    def visit_BoolOp(self, node):
        self.features['num_boolean_ops'] += 1
        self.generic_visit(node)

    def visit_BinOp(self, node):
        self.features['num_binary_ops'] += 1
        self.generic_visit(node)

    def visit_UnaryOp(self, node):
        self.features['num_unary_ops'] += 1
        self.generic_visit(node)

    def visit_Try(self, node):
        self.features['num_try_except'] += 1
        self.generic_visit(node)

    def visit_Raise(self, node):
        self.features['num_raise'] += 1
        self.generic_visit(node)

    def visit_Assert(self, node):
        self.features['num_assertions'] += 1
        self.generic_visit(node)

    def visit_Return(self, node):
        self.features['num_returns'] += 1
        self.generic_visit(node)

    def visit_Yield(self, node):
        self.features['num_yields'] += 1
        self.generic_visit(node)

    def visit_YieldFrom(self, node):
        self.features['num_yields'] += 1
        self.generic_visit(node)

    def visit_ListComp(self, node):
        self.features['num_list_comps'] += 1
        self.generic_visit(node)

    def visit_DictComp(self, node):
        self.features['num_dict_comps'] += 1
        self.generic_visit(node)

    def visit_SetComp(self, node):
        self.features['num_set_comps'] += 1
        self.generic_visit(node)

    def visit_GeneratorExp(self, node):
        self.features['num_generator_exps'] += 1
        self.generic_visit(node)

    def visit_Lambda(self, node):
        self.features['num_lambda'] += 1
        self.generic_visit(node)

    def visit_Subscript(self, node):
        self.features['num_subscripts'] += 1
        self.generic_visit(node)

    def visit_Attribute(self, node):
        self.features['num_attributes'] += 1
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        # f-string
        self.features['num_string_formats'] += 1
        self.generic_visit(node)

    def visit_ExceptHandler(self, node):
        # Check for bare except (no exception type specified)
        if node.type is None:
            self.features['num_bare_except'] += 1
        self.generic_visit(node)

    def visit_Global(self, node):
        # Global variable declaration
        self.features['num_global'] += len(node.names)
        self.generic_visit(node)


# =============================================================================
# FEATURE EXTRACTORS
# =============================================================================

class CodeFeatureExtractor:
    """
    Extracts numerical features from Python code.

    Features:
    1. Basic metrics: LOC, parameters, etc.
    2. Complexity metrics: cyclomatic complexity, nesting depth
    3. AST-based counts: various node types
    """

    def __init__(self):
        self.feature_names = []
        self._build_feature_names()

    def _build_feature_names(self):
        """Build list of all feature names"""
        # Basic features
        basic = [
            'lines_of_code',
            'num_parameters',
            'num_defaults',  # Parameters with default values
            'has_varargs',   # *args
            'has_kwargs',    # **kwargs
        ]

        # Complexity features
        complexity = [
            'cyclomatic_complexity',
            'max_nesting_depth',
        ]

        # AST features (from ASTFeatureVisitor)
        ast_features = [
            'num_assignments',
            'num_aug_assignments',
            'num_for_loops',
            'num_while_loops',
            'num_if_statements',
            'num_function_calls',
            'num_method_calls',
            'num_comparisons',
            'num_boolean_ops',
            'num_binary_ops',
            'num_unary_ops',
            'num_try_except',
            'num_raise',
            'num_assertions',
            'num_returns',
            'num_yields',
            'num_list_comps',
            'num_dict_comps',
            'num_set_comps',
            'num_generator_exps',
            'num_lambda',
            'num_subscripts',
            'num_attributes',
            'has_recursion',
            'num_string_formats',
            # Bug pattern indicators
            'num_bare_except',
            'num_eval_exec',
            'num_global',
            'num_mutable_defaults',
            'has_docstring',
        ]

        self.feature_names = basic + complexity + ast_features

    def extract_features(self, code: str, function_name: str = None) -> Dict[str, float]:
        """
        Extract all features from a code string.

        Args:
            code: Python function source code
            function_name: Name of the function (for recursion detection)

        Returns:
            Dictionary of feature name -> value
        """
        features = {}

        # Try to parse the code
        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Return zeros for unparseable code
            return {name: 0.0 for name in self.feature_names}

        # Basic features
        lines = code.split('\n')
        features['lines_of_code'] = len([l for l in lines if l.strip()])

        # Extract function-specific features if it's a function definition
        func_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_node = node
                break

        if func_node:
            features['num_parameters'] = len(func_node.args.args)
            features['num_defaults'] = len(func_node.args.defaults)
            features['has_varargs'] = 1 if func_node.args.vararg else 0
            features['has_kwargs'] = 1 if func_node.args.kwarg else 0
            function_name = func_node.name

            # Check for mutable default arguments (common Python bug!)
            mutable_defaults = 0
            for default in func_node.args.defaults:
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    mutable_defaults += 1
            features['num_mutable_defaults'] = mutable_defaults

            # Check for docstring
            features['has_docstring'] = 1 if ast.get_docstring(func_node) else 0
        else:
            features['num_parameters'] = 0
            features['num_defaults'] = 0
            features['has_varargs'] = 0
            features['has_kwargs'] = 0
            features['num_mutable_defaults'] = 0
            features['has_docstring'] = 0

        # Cyclomatic complexity
        complexity_visitor = ComplexityVisitor()
        complexity_visitor.visit(tree)
        features['cyclomatic_complexity'] = complexity_visitor.complexity

        # Nesting depth
        nesting_visitor = NestingDepthVisitor()
        nesting_visitor.visit(tree)
        features['max_nesting_depth'] = nesting_visitor.max_depth

        # AST features
        ast_visitor = ASTFeatureVisitor(function_name)
        ast_visitor.visit(tree)
        features.update(ast_visitor.features)

        return features

    def extract_batch(self, codes: List[str],
                      function_names: List[str] = None) -> np.ndarray:
        """
        Extract features from multiple code samples.

        Args:
            codes: List of code strings
            function_names: Optional list of function names

        Returns:
            Feature matrix of shape (n_samples, n_features)
        """
        if function_names is None:
            function_names = [None] * len(codes)

        features_list = []
        for code, func_name in tqdm(zip(codes, function_names),
                                     total=len(codes),
                                     desc="Extracting features"):
            features = self.extract_features(code, func_name)
            features_list.append([features.get(name, 0.0) for name in self.feature_names])

        return np.array(features_list, dtype=np.float32)

    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()

    def extract_all_features(self, code: str, function_name: str = None) -> Optional[Dict[str, float]]:
        """
        Alias for extract_features with None return on failure.

        Args:
            code: Python function source code
            function_name: Name of the function (for recursion detection)

        Returns:
            Dictionary of feature name -> value, or None if parsing fails
        """
        try:
            ast.parse(code)
        except SyntaxError:
            return None

        return self.extract_features(code, function_name)


class CodeTokenizer:
    """
    Tokenizes Python code into token sequences.

    Why Tokenization?
    -----------------
    Neural networks can't process raw text. We convert code to numbers:
    1. Split code into tokens (keywords, operators, identifiers)
    2. Build vocabulary of most common tokens
    3. Map each token to a unique ID
    4. Pad/truncate to fixed length

    Why Padding?
    ------------
    Neural networks need fixed-size inputs for batch processing.
    - Short code: Pad with <PAD> token (ID=0)
    - Long code: Truncate to max_length
    """

    # Special tokens
    PAD_TOKEN = '<PAD>'
    UNK_TOKEN = '<UNK>'  # Unknown token

    def __init__(self, vocab_size: int = 5000, max_length: int = 200):
        """
        Initialize tokenizer.

        Args:
            vocab_size: Maximum vocabulary size
            max_length: Fixed sequence length (pad/truncate to this)
        """
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_to_id = {}
        self.id_to_token = {}
        self.is_fitted = False

    def _tokenize_code(self, code: str) -> List[str]:
        """
        Tokenize Python code using Python's tokenize module.

        Returns list of token strings.
        """
        tokens = []

        try:
            # Use Python's tokenize module for proper tokenization
            readline = StringIO(code).readline
            for tok in tokenize.generate_tokens(readline):
                # tok.type: token type (NAME, NUMBER, OP, etc.)
                # tok.string: actual token string

                if tok.type == tokenize.NAME:
                    # Keywords and identifiers
                    tokens.append(tok.string)
                elif tok.type == tokenize.OP:
                    # Operators and punctuation
                    tokens.append(tok.string)
                elif tok.type == tokenize.NUMBER:
                    # Replace numbers with placeholder
                    tokens.append('<NUM>')
                elif tok.type == tokenize.STRING:
                    # Replace strings with placeholder
                    tokens.append('<STR>')
                # Skip comments, newlines, encoding, etc.

        except (tokenize.TokenError, IndentationError, SyntaxError):
            # Fallback: simple whitespace tokenization
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)
        except Exception:
            tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*|[^\s]', code)

        return tokens

    def fit(self, codes: List[str]):
        """
        Build vocabulary from code samples.

        Args:
            codes: List of code strings
        """
        print("Building vocabulary...")

        # Count all tokens
        token_counts = Counter()
        for code in tqdm(codes, desc="Tokenizing"):
            tokens = self._tokenize_code(code)
            token_counts.update(tokens)

        # Select top vocab_size - 2 tokens (reserve space for special tokens)
        most_common = token_counts.most_common(self.vocab_size - 2)

        # Build vocabulary
        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        for i, (token, _) in enumerate(most_common):
            self.token_to_id[token] = i + 2

        # Reverse mapping
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

        self.is_fitted = True
        print(f"Vocabulary size: {len(self.token_to_id)}")

    def transform(self, codes: List[str]) -> np.ndarray:
        """
        Convert code samples to padded token sequences.

        Args:
            codes: List of code strings

        Returns:
            Array of shape (n_samples, max_length) with token IDs
        """
        if not self.is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")

        sequences = []
        for code in tqdm(codes, desc="Converting to sequences"):
            tokens = self._tokenize_code(code)

            # Convert tokens to IDs
            ids = [self.token_to_id.get(tok, 1) for tok in tokens]  # 1 = UNK

            # Pad or truncate
            if len(ids) < self.max_length:
                ids = ids + [0] * (self.max_length - len(ids))  # Pad with 0
            else:
                ids = ids[:self.max_length]  # Truncate

            sequences.append(ids)

        return np.array(sequences, dtype=np.int32)

    def fit_transform(self, codes: List[str]) -> np.ndarray:
        """Fit vocabulary and transform in one step."""
        self.fit(codes)
        return self.transform(codes)

    def save(self, path: str):
        """Save vocabulary to JSON file."""
        with open(path, 'w') as f:
            json.dump({
                'vocab_size': self.vocab_size,
                'max_length': self.max_length,
                'token_to_id': self.token_to_id,
            }, f, indent=2)

    def load(self, path: str):
        """Load vocabulary from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)

        self.vocab_size = data['vocab_size']
        self.max_length = data['max_length']
        self.token_to_id = data['token_to_id']
        self.id_to_token = {int(v): k for k, v in self.token_to_id.items()}
        self.is_fitted = True


# =============================================================================
# MAIN PREPROCESSING PIPELINE
# =============================================================================

class Preprocessor:
    """
    Main preprocessing pipeline.

    Combines:
    1. Feature extraction (numerical features)
    2. Tokenization (sequence features)
    3. Data splitting (train/val/test)
    4. Normalization
    """

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize preprocessor with configuration.

        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Initialize components
        self.feature_extractor = CodeFeatureExtractor()

        preprocessing_config = self.config.get('preprocessing', {})
        self.tokenizer = CodeTokenizer(
            vocab_size=preprocessing_config.get('vocab_size', 5000),
            max_length=preprocessing_config.get('max_length', 200)
        )

        self.scaler = StandardScaler()

        # Paths
        self.data_raw_dir = Path(self.config['paths']['data_raw'])
        self.data_processed_dir = Path(self.config['paths']['data_processed'])
        self.data_processed_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """Load raw data from CSV."""
        csv_path = self.data_raw_dir / "functions.csv"

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {csv_path}\n"
                "Please run data collection first: python -m src.data_collection"
            )

        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} samples")
        print(f"  Buggy (has_bug=1): {df['has_bug'].sum()}")
        print(f"  Clean (has_bug=0): {(df['has_bug'] == 0).sum()}")

        return df

    def process(self):
        """
        Run the full preprocessing pipeline.

        Steps:
        1. Load raw data
        2. Extract numerical features
        3. Tokenize code
        4. Split data (train/val/test)
        5. Normalize features
        6. Save processed data
        """
        print("\n" + "="*70)
        print("PREPROCESSING PIPELINE")
        print("="*70 + "\n")

        # Step 1: Load data
        df = self.load_data()
        codes = df['code'].tolist()
        labels = df['has_bug'].values
        function_names = df['function_name'].tolist()

        # Step 2: Extract numerical features
        print("\n--- Step 2: Extracting numerical features ---")
        features = self.feature_extractor.extract_batch(codes, function_names)
        print(f"Feature matrix shape: {features.shape}")

        # Step 3: Tokenize code
        print("\n--- Step 3: Tokenizing code ---")
        token_sequences = self.tokenizer.fit_transform(codes)
        print(f"Token sequence shape: {token_sequences.shape}")

        # Step 4: Split data
        print("\n--- Step 4: Splitting data ---")
        train_config = self.config.get('training', {})
        train_ratio = train_config.get('train_split', 0.7)
        val_ratio = train_config.get('val_split', 0.15)
        test_ratio = train_config.get('test_split', 0.15)
        random_seed = train_config.get('random_seed', 42)

        indices = np.arange(len(df))

        # First split: train vs (val + test)
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=train_ratio,
            random_state=random_seed,
            stratify=labels  # Maintain class balance
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            random_state=random_seed,
            stratify=labels[temp_idx]
        )

        print(f"Train: {len(train_idx)} samples")
        print(f"Validation: {len(val_idx)} samples")
        print(f"Test: {len(test_idx)} samples")

        # Step 5: Normalize features (fit on train only)
        print("\n--- Step 5: Normalizing features ---")
        features_train = features[train_idx]
        self.scaler.fit(features_train)
        features_normalized = self.scaler.transform(features)
        print("Features normalized using StandardScaler")

        # Step 6: Save everything
        print("\n--- Step 6: Saving processed data ---")

        # Save features
        np.savez_compressed(
            self.data_processed_dir / "features.npz",
            features=features_normalized
        )
        print(f"  Saved: features.npz ({features_normalized.shape})")

        # Save token sequences
        np.savez_compressed(
            self.data_processed_dir / "token_sequences.npz",
            sequences=token_sequences
        )
        print(f"  Saved: token_sequences.npz ({token_sequences.shape})")

        # Save labels
        np.save(self.data_processed_dir / "labels.npy", labels)
        print(f"  Saved: labels.npy ({labels.shape})")

        # Save indices
        np.save(self.data_processed_dir / "train_indices.npy", train_idx)
        np.save(self.data_processed_dir / "val_indices.npy", val_idx)
        np.save(self.data_processed_dir / "test_indices.npy", test_idx)
        print(f"  Saved: train/val/test indices")

        # Save vocabulary
        self.tokenizer.save(self.data_processed_dir / "vocabulary.json")
        print(f"  Saved: vocabulary.json")

        # Save feature names
        with open(self.data_processed_dir / "feature_names.json", 'w') as f:
            json.dump(self.feature_extractor.feature_names, f, indent=2)
        print(f"  Saved: feature_names.json")

        # Save scaler parameters
        scaler_params = {
            'mean': self.scaler.mean_.tolist(),
            'scale': self.scaler.scale_.tolist(),
            'feature_names': self.feature_extractor.feature_names
        }
        with open(self.data_processed_dir / "scaler_params.json", 'w') as f:
            json.dump(scaler_params, f, indent=2)
        print(f"  Saved: scaler_params.json")

        # Save metadata
        metadata = {
            'n_samples': len(df),
            'n_features': features.shape[1],
            'n_tokens': token_sequences.shape[1],
            'vocab_size': len(self.tokenizer.token_to_id),
            'n_train': len(train_idx),
            'n_val': len(val_idx),
            'n_test': len(test_idx),
            'class_distribution': {
                'buggy': int(labels.sum()),
                'clean': int((labels == 0).sum())
            },
            'feature_names': self.feature_extractor.feature_names
        }
        with open(self.data_processed_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved: metadata.json")

        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"\nOutput directory: {self.data_processed_dir}")
        print("\nFiles created:")
        for f in sorted(self.data_processed_dir.glob("*")):
            print(f"  - {f.name}")

        return {
            'features': features_normalized,
            'token_sequences': token_sequences,
            'labels': labels,
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx
        }


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    preprocessor = Preprocessor(config_path="config.yaml")
    preprocessor.process()
