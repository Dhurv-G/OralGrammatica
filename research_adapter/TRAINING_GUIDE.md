# Training Guide for Grammar Scoring Models

This guide explains how to train the DeBERTa/RoBERTa fine-tuned model and the stacker model for grammar scoring.

## Overview

The system uses a two-stage approach:
1. **Fine-tune DeBERTa/RoBERTa** to predict grammar scores from text only
2. **Train a stacker model** that combines multiple features to predict final scores

## Prerequisites

- Training data with human-annotated grammar scores (0-100)
- CSV or JSON format with `text` and `score` columns
- Sufficient GPU memory for fine-tuning (or use CPU with smaller batch size)

## Step 1: Fine-tune DeBERTa/RoBERTa

### Prepare Training Data

Create a CSV or JSON file with your training data:

**CSV format:**
```csv
text,score
"The birch canoe slid on the smooth planks.",95.0
"I am going to the store yesterday.",65.0
...
```

**JSON format:**
```json
[
  {"text": "The birch canoe slid on the smooth planks.", "score": 95.0},
  {"text": "I am going to the store yesterday.", "score": 65.0}
]
```

### Run Fine-tuning

```bash
python -m research_adapter.train_deberta \
    --data training_data.csv \
    --text-col text \
    --score-col score \
    --model microsoft/deberta-v3-base \
    --output ./models/deberta_finetuned \
    --epochs 3 \
    --batch-size 8 \
    --lr 2e-5
```

**Options:**
- `--data`: Path to training data (CSV or JSON)
- `--text-col`: Column name for text (default: "text")
- `--score-col`: Column name for score (default: "score")
- `--model`: Base model name (default: "microsoft/deberta-v3-base")
- `--roberta`: Use RoBERTa instead of DeBERTa
- `--output`: Output directory for saved model
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 2e-5)

## Step 2: Train Stacker Model

The stacker combines:
- `score_rule`: Rule-based score
- `score_gec`: GEC score
- `score_text_only`: Text-only transformer score
- GEC edit metrics
- Category counts
- Complexity features

### Run Stacker Training

```bash
python -m research_adapter.train_stacker \
    --data training_data.csv \
    --text-col text \
    --score-col score \
    --output ./models/stacker \
    --val-split 0.2
```

**Options:**
- `--data`: Path to training data (CSV or JSON)
- `--text-col`: Column name for text (default: "text")
- `--score-col`: Column name for score (default: "score")
- `--output`: Output directory for saved model
- `--val-split`: Validation split ratio (default: 0.2)

## Step 3: Configure Model Paths

Update `config.py` to point to your trained models:

```python
STACKER_CONFIG = {
    'model_path': './models/stacker',  # Path to trained stacker
    'text_only_model_path': './models/deberta_finetuned',  # Path to fine-tuned DeBERTa
    ...
}
```

## Fallback Mode

If no trained models are available, the system runs in **fallback mode**:
- Uses weighted average of `score_rule`, `score_gec`, and `score_text_only`
- Weights: rule=0.4, gec=0.4, text_only=0.2
- No isotonic calibration applied

## Model Artifacts

### DeBERTa Fine-tuned Model
- `config.json`: Model configuration
- `pytorch_model.bin`: Model weights
- `tokenizer_config.json`: Tokenizer configuration
- `vocab.txt`: Vocabulary

### Stacker Model
- `stacker_model.pkl`: Trained stacker model
- `scaler.pkl`: Feature scaler
- `calibrator.pkl`: Isotonic calibrator
- `metadata.pkl`: Model metadata

## Evaluation

After training, the stacker will output:
- Train R² score
- Validation R² score
- Model artifacts saved for inference

## Usage in Production

The trained models are automatically loaded in `runner.py`:
- If models exist: Uses trained stacker with calibration
- If models don't exist: Falls back to weighted average

No code changes needed - just ensure model paths are correct in `config.py`.

