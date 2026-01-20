"""
FastAPI Service for Bug Prediction

Provides REST API for predicting bugs in Python code.

Usage:
    uvicorn src.api:app --reload
    # Then visit http://localhost:8000/docs for API documentation

Endpoints:
    POST /predict - Predict bug probability for Python code
    GET /health - Service health check
    GET /examples - Get example code snippets
"""

import ast
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

from src.model import create_model
from src.preprocessing import CodeFeatureExtractor, CodeTokenizer
from src.utils import get_device

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# Pydantic Models
# =============================================================================

class PredictRequest(BaseModel):
    """Request model for /predict endpoint."""
    code: str = Field(..., min_length=1, max_length=10000,
                      description="Python code to analyze")

    @validator('code')
    def validate_python_syntax(cls, v):
        """Validate that code is syntactically valid Python."""
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")
        return v


class FeatureInfo(BaseModel):
    """Feature information in prediction response."""
    name: str
    value: float


class PredictResponse(BaseModel):
    """Response model for /predict endpoint."""
    has_bug: bool
    probability: float
    confidence: float
    prediction: str
    top_features: List[FeatureInfo]


class HealthResponse(BaseModel):
    """Response model for /health endpoint."""
    status: str
    model_loaded: bool
    model_type: str


class ExampleCode(BaseModel):
    """Example code snippet."""
    name: str
    code: str
    description: str
    expected: str


# =============================================================================
# Bug Predictor Service
# =============================================================================

class BugPredictorService:
    """Singleton service for bug prediction."""

    def __init__(self):
        self.model = None
        self.extractor = None
        self.tokenizer = None
        self.config = None
        self.device = None
        self.model_type = None
        self._initialized = False

    def initialize(self, config_path: str = "config.yaml"):
        """Initialize the predictor with trained model."""
        if self._initialized:
            return

        logger.info("Initializing Bug Predictor Service...")

        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.device = get_device(False)  # Use CPU for inference
        self.model_type = self.config['model'].get('model_type', 'feature_only')

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

        # Create and load model
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
            logger.info(f"Loaded {self.model_type} model from {model_path}")
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
                logger.info(f"Loaded vocabulary ({len(vocab)} tokens)")

        self._initialized = True
        logger.info("Bug Predictor Service initialized successfully!")

    def predict(self, code: str) -> Dict:
        """Predict bug probability for Python code."""
        if not self._initialized:
            raise RuntimeError("Service not initialized")

        # Extract features
        features_dict = self.extractor.extract_all_features(code)
        if features_dict is None:
            raise ValueError("Failed to extract features from code")

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
        # (supplements ML model which was trained on different bug types)
        rule_boost = self._detect_bug_patterns(features_dict, code)

        # Combine ML prediction with rule-based detection
        # Rule boost can push probability up to 0.85 max
        probability = min(0.95, ml_probability + rule_boost)

        # Get top contributing features
        top_features = self._get_top_features(features_dict, feature_names)

        return {
            'has_bug': probability >= 0.5,
            'probability': probability,
            'confidence': abs(probability - 0.5) * 2,
            'prediction': 'BUGGY' if probability >= 0.5 else 'CLEAN',
            'top_features': top_features
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

    def _get_top_features(self, features_dict: Dict, feature_names: List[str],
                          top_n: int = 5) -> List[Dict[str, float]]:
        """Get top features that might indicate bugs."""
        bug_indicators = {
            'cyclomatic_complexity': 1.5,
            'max_nesting_depth': 2.0,
            'num_nested_functions': 1.5,
            'num_bare_except': 3.0,
            'num_mutable_defaults': 3.0,
            'num_global': 2.0,
            'num_eval_exec': 3.0,
            'num_assertions': -0.5,
            'has_docstring': -1.0,
            'num_comments': -0.5,
        }

        scored_features = []
        for name in feature_names:
            value = features_dict.get(name, 0)
            weight = bug_indicators.get(name, 1.0)
            if value > 0:
                score = value * weight
                scored_features.append({'name': name, 'value': float(value), 'score': score})

        scored_features.sort(key=lambda x: x['score'], reverse=True)
        return [{'name': f['name'], 'value': f['value']} for f in scored_features[:top_n]]


# =============================================================================
# FastAPI Application
# =============================================================================

# Create service instance
predictor_service = BugPredictorService()

# Create FastAPI app
app = FastAPI(
    title="Bug Prediction API",
    description="Predict bugs in Python code using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize service on startup."""
    try:
        predictor_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize service: {e}")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve web interface."""
    static_path = Path(__file__).parent / "static" / "index.html"
    if static_path.exists():
        return static_path.read_text()
    return """
    <html>
        <head><title>Bug Prediction API</title></head>
        <body>
            <h1>Bug Prediction API</h1>
            <p>Visit <a href="/docs">/docs</a> for API documentation.</p>
        </body>
    </html>
    """


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    return HealthResponse(
        status="healthy" if predictor_service._initialized else "initializing",
        model_loaded=predictor_service._initialized,
        model_type=predictor_service.model_type or "unknown"
    )


@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """
    Predict bug probability for Python code.

    - **code**: Python function or code snippet to analyze
    """
    if not predictor_service._initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = predictor_service.predict(request.code)
        return PredictResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/examples", response_model=List[ExampleCode])
async def get_examples():
    """Get example code snippets for testing."""
    return [
        ExampleCode(
            name="Clean Function",
            code='''def calculate_average(numbers):
    """Calculate the average of a list of numbers."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)''',
            description="Well-structured function with validation",
            expected="CLEAN"
        ),
        ExampleCode(
            name="Off-by-one Error",
            code='''def get_last_items(items, n):
    result = []
    for i in range(n + 1):  # Bug: should be range(n)
        result.append(items[i])
    return result''',
            description="Off-by-one error in loop range",
            expected="BUGGY"
        ),
        ExampleCode(
            name="Missing Null Check",
            code='''def process_data(data):
    result = data.strip().lower()
    parts = result.split(',')
    return parts[0]''',
            description="No validation for None input",
            expected="BUGGY"
        ),
        ExampleCode(
            name="Mutable Default",
            code='''def append_to_list(item, items=[]):
    items.append(item)
    return items''',
            description="Mutable default argument (common Python bug)",
            expected="BUGGY"
        ),
        ExampleCode(
            name="Binary Search",
            code='''def binary_search(arr, target):
    """Binary search for target in sorted array."""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1''',
            description="Classic binary search implementation",
            expected="CLEAN"
        ),
        ExampleCode(
            name="Deep Nesting",
            code='''def complex_logic(a, b, c, d):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    for i in range(a):
                        for j in range(b):
                            if i == j:
                                return i * j
    return 0''',
            description="Deeply nested logic (code smell)",
            expected="BUGGY"
        )
    ]


# Mount static files (if directory exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
