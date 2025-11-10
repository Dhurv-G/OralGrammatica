"""
Training script for fine-tuning DeBERTa/RoBERTa
"""

import argparse
import json
import pandas as pd
from pathlib import Path

from .finetune_deberta import DeBERTaFineTuner
from .config import FINETUNE_CONFIG


def load_data_from_csv(csv_path: str, text_column: str = "text", 
                      score_column: str = "score"):
    """Load training data from CSV"""
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    scores = df[score_column].tolist()
    return texts, scores


def load_data_from_json(json_path: str):
    """Load training data from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    scores = [item['score'] for item in data]
    
    return texts, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa/RoBERTa for grammar scoring")
    parser.add_argument("--data", type=str, required=True, help="Path to training data (CSV or JSON)")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--score-col", type=str, default="score", help="Score column name")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v3-base", help="Base model name")
    parser.add_argument("--roberta", action="store_true", help="Use RoBERTa instead of DeBERTa")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # Load data
    if args.data.endswith('.csv'):
        texts, scores = load_data_from_csv(args.data, args.text_col, args.score_col)
    elif args.data.endswith('.json'):
        texts, scores = load_data_from_json(args.data)
    else:
        raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    print(f"📊 Loaded {len(texts)} training samples")
    
    # Initialize fine-tuner
    fine_tuner = DeBERTaFineTuner(
        model_name=args.model,
        use_roberta=args.roberta
    )
    
    # Set output directory
    output_dir = args.output or FINETUNE_CONFIG['output_dir']
    
    # Train
    fine_tuner.train(
        texts=texts,
        scores=scores,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"✅ Fine-tuning complete! Model saved to {output_dir}")

