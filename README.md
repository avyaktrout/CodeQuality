# Bug Prediction Neural Network

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A machine learning system that predicts whether Python code contains bugs using neural networks. Features data collection from GitHub, AST-based feature extraction, hybrid neural network architecture, and REST API deployment.

## Results

| Metric | Score |
|--------|-------|
| **Accuracy** | 73.2% |
| **Precision** | 82.2% |
| **Recall** | 59.2% |
| **F1 Score** | 68.8% |
| **ROC-AUC** | 0.81 |

*Evaluated on 750 test functions (balanced dataset of buggy/clean code)*

## Architecture

```
                              CODE INPUT
                                  |
                    +-------------+-------------+
                    |                           |
            [Feature Extraction]         [Tokenization]
                    |                           |
            32 Numerical Features      200 Token Sequence
                    |                           |
              +-----+-----+              +------+------+
              |           |              |             |
          Dense(64)   BatchNorm     Embedding(128)  Padding
              |           |              |             |
            ReLU      Dropout(0.3)    LSTM(64)        Mask
              |           |              |             |
          Dense(32)       +         Global Pool        +
              |                          |
              +--------CONCAT------------+
                         |
                     [96 dims]
                         |
                   +-----+-----+
                   |           |
               Dense(64)   BatchNorm
                   |           |
                 ReLU      Dropout(0.3)
                   |           |
               Dense(32)       +
                   |
               Dense(1)
                   |
               Sigmoid
                   |
           Bug Probability [0-1]
```

## Features

- **32 Code Features**: Cyclomatic complexity, nesting depth, loop counts, etc.
- **Token Sequences**: 5000-word vocabulary, 200-token context
- **LSTM Encoder**: Captures sequential patterns in code
- **Hybrid Fusion**: Combines numerical and sequential features

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/CodeQuality.git
cd CodeQuality

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Interactive demo
python demo.py

# Test on specific code
python -m src.evaluate --code "def f(x=[]): x.append(1); return x"

# Analyze a file
python -m src.evaluate --file mycode.py
```

### Start API Server

```bash
# Start the server
uvicorn src.api:app --reload

# Open http://localhost:8000 for web interface
# Open http://localhost:8000/docs for API documentation
```

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t bug-prediction .
docker run -p 8000:8000 bug-prediction
```

## API Usage

### Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"code": "def divide(a, b): return a / b"}'
```

Response:
```json
{
  "has_bug": true,
  "probability": 0.72,
  "confidence": 0.44,
  "prediction": "BUGGY",
  "top_features": [
    {"name": "num_binary_ops", "value": 1},
    {"name": "lines_of_code", "value": 2}
  ]
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

## Project Structure

```
CodeQuality/
├── src/
│   ├── api.py              # FastAPI service
│   ├── model.py            # Neural network architecture
│   ├── preprocessing.py    # Feature extraction
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation & CLI
│   ├── data_collection.py  # GitHub data scraping
│   └── static/             # Web interface
├── data/
│   ├── raw/                # Collected functions
│   └── processed/          # Extracted features
├── models/
│   └── best_model.pth      # Trained model
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_demo.ipynb
├── demo.py                 # Interactive demo
├── config.yaml             # Configuration
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## Training Your Own Model

### 1. Data Collection

Set up GitHub token in `.env`:
```bash
GITHUB_TOKEN=your_token_here
```

Collect functions:
```bash
python -m src.data_collection
```

### 2. Preprocessing

Extract features:
```bash
python -m src.preprocessing
```

### 3. Training

```bash
python -m src.train
```

### 4. Evaluation

```bash
python -m src.evaluate --test
```

## Configuration

Edit `config.yaml` to customize:

```yaml
model:
  model_type: "hybrid"      # or "feature_only"
  use_lstm: true
  embedding_dim: 128

training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 50
  early_stopping_patience: 5
```

## Technical Details

### Feature Engineering

| Category | Features | Description |
|----------|----------|-------------|
| Complexity | 5 | Lines, parameters, cyclomatic complexity, nesting depth |
| Control Flow | 8 | Loops, conditionals, try/except, returns |
| Operations | 10 | Assignments, comparisons, function calls |
| Bug Patterns | 9 | Mutable defaults, bare excepts, eval usage |

### Model Architecture

- **FeatureOnlyModel**: 6K parameters, uses only numerical features
- **HybridModel**: 702K parameters, combines features + LSTM

### Training Details

- Optimizer: Adam (lr=0.001, weight_decay=0.0001)
- Loss: Binary Cross-Entropy
- Early stopping: patience=5 epochs
- Learning rate scheduler: ReduceLROnPlateau

## Notebooks

1. **01_data_exploration.ipynb**: Dataset analysis and visualization
2. **02_model_evaluation.ipynb**: Error analysis, feature importance, ablation study
3. **03_demo.ipynb**: Interactive examples and predictions

## Limitations

- Trained on synthetic bugs (injected patterns), not real-world bugs
- Limited to function-level analysis (not project-level)
- Python-specific (not language-agnostic)
- 73% accuracy leaves room for improvement

## Future Improvements

- [ ] Collect real buggy code from GitHub bug-fix commits
- [ ] Add multi-class bug type prediction
- [ ] Implement attention visualization
- [ ] Deploy to cloud (AWS/GCP/Hugging Face Spaces)
- [ ] Build VS Code extension

## License

MIT License - Free to use for learning and research.

## Acknowledgments

- GitHub API for data access
- PyTorch for neural network framework
- FastAPI for REST API
- scikit-learn for ML utilities

---

**Built as an educational ML project demonstrating the complete pipeline from data collection to deployment.**
