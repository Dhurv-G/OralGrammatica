"""
Training script for stacker model
Can be run with or without labels (fallback mode)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
import os
import json

from .stacker_model import StackerModel
from .runner import score_rule
from .features_onfly import FeatureExtractor
from .ensemble_infer import EnsembleInference
from .text_only_scorer import TextOnlyScorer


def prepare_training_data(texts: List[str], labels: Optional[List[float]] = None) -> tuple:
    """
    Prepare training data for stacker
    
    Args:
        texts: List of input texts
        labels: Optional list of human-annotated scores (0-100)
        
    Returns:
        Tuple of (feature_matrix, labels) or (None, None) if no labels
    """
    print("📊 Preparing training data...")
    
    # Initialize components
    feature_extractor = FeatureExtractor()
    ensemble = EnsembleInference()
    text_scorer = TextOnlyScorer()
    
    all_features = []
    all_labels = []
    
    for i, text in enumerate(texts):
        if (i + 1) % 10 == 0:
            print(f"   Processing {i+1}/{len(texts)}...")
        
        # Get all analysis results
        analysis_result = ensemble.analyze_text(text)
        rule_results = score_rule(text)
        
        # Add rule-based results
        analysis_result['rule_score'] = rule_results['rule_score']
        analysis_result['category_counts'] = rule_results['category_counts']
        analysis_result['complexity_stats'] = rule_results['complexity_stats']
        analysis_result['languagetool_matches'] = rule_results['languagetool_matches']
        
        # Add GEC results
        gec_results = rule_results.get('gec_results', {})
        analysis_result['gec_results'] = gec_results
        analysis_result['score_gec'] = gec_results.get('score_gec', 100.0)
        analysis_result['edit_metrics'] = gec_results.get('edit_metrics', {})
        
        # Add text-only score
        analysis_result['score_text_only'] = text_scorer.score_text_only(text)
        
        # Extract features
        stacker = StackerModel()
        features = stacker.extract_features(analysis_result)
        all_features.append(features)
        
        # Add label if available
        if labels is not None:
            all_labels.append(labels[i])
    
    X = np.array(all_features)
    y = np.array(all_labels) if labels else None
    
    print(f"✅ Prepared {len(X)} samples with {X.shape[1]} features")
    
    return X, y


def train_stacker(texts: List[str], 
                 labels: Optional[List[float]] = None,
                 output_dir: str = "./models/stacker",
                 validation_split: float = 0.2):
    """
    Train stacker model
    
    Args:
        texts: List of input texts
        labels: Optional list of human-annotated scores (0-100)
        output_dir: Directory to save model
        validation_split: Fraction for validation
    """
    print("🎯 Training Stacker Model")
    print("=" * 60)
    
    # Prepare data
    X, y = prepare_training_data(texts, labels)
    
    if y is None:
        print("⚠️  No labels provided. Stacker will run in fallback mode.")
        print("   Using weighted average of score_rule, score_gec, and score_text_only")
        return None
    
    # Train model
    stacker = StackerModel(use_calibration=True)
    stacker.train(X, y, validation_split=validation_split)
    
    # Save model
    os.makedirs(output_dir, exist_ok=True)
    stacker.save_model(output_dir)
    
    print(f"✅ Stacker model trained and saved to {output_dir}")
    
    return stacker


def load_data_from_csv(csv_path: str, text_column: str = "text", 
                      score_column: str = "score") -> tuple:
    """
    Load training data from CSV file
    
    Args:
        csv_path: Path to CSV file
        text_column: Name of text column
        score_column: Name of score column
        
    Returns:
        Tuple of (texts, scores)
    """
    df = pd.read_csv(csv_path)
    texts = df[text_column].tolist()
    scores = df[score_column].tolist()
    return texts, scores


def load_data_from_json(json_path: str) -> tuple:
    """
    Load training data from JSON file
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Tuple of (texts, scores)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    texts = [item['text'] for item in data]
    scores = [item['score'] for item in data]
    
    return texts, scores


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train stacker model")
    parser.add_argument("--data", type=str, help="Path to training data (CSV or JSON)")
    parser.add_argument("--text-col", type=str, default="text", help="Text column name")
    parser.add_argument("--score-col", type=str, default="score", help="Score column name")
    parser.add_argument("--output", type=str, default="./models/stacker", help="Output directory")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split")
    
    args = parser.parse_args()
    
    if args.data:
        # Load data
        if args.data.endswith('.csv'):
            texts, scores = load_data_from_csv(args.data, args.text_col, args.score_col)
        elif args.data.endswith('.json'):
            texts, scores = load_data_from_json(args.data)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
        
        # Train
        train_stacker(texts, scores, args.output, args.val_split)
    else:
        print("No training data provided. Stacker will use fallback mode.")
        print("To train, provide --data argument with CSV or JSON file.")

