"""
Text-only scoring using transformer models (DeBERTa/RoBERTa)
"""

import torch
from typing import Dict, Any, Optional
import numpy as np

try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification,
        DebertaV2Tokenizer,
        DebertaV2ForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Text-only scoring will be disabled.")


class TextOnlyScorer:
    """Score grammar using text-only transformer models"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", use_finetuned: bool = False, model_path: Optional[str] = None):
        """
        Initialize text-only scorer
        
        Args:
            model_name: Base model name (default: microsoft/deberta-v3-base)
            use_finetuned: Whether to use a fine-tuned model
            model_path: Path to fine-tuned model if available
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.use_finetuned = use_finetuned
        self.model_path = model_path
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            # Load fine-tuned model if available
            if self.use_finetuned and self.model_path:
                try:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_path,
                        num_labels=1  # Regression for score prediction
                    )
                    self.model.eval()
                    print(f"✅ Loaded fine-tuned model from {self.model_path}")
                    return
                except Exception as e:
                    print(f"Warning: Could not load fine-tuned model: {e}")
            
            # Fallback to base model
            if "deberta" in self.model_name.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=1
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    num_labels=1
                )
            
            self.model.eval()
            print(f"✅ Loaded base model: {self.model_name}")
        except Exception as e:
            print(f"Warning: Could not initialize text-only scorer: {e}")
            self.model = None
            self.tokenizer = None
    
    def score_text_only(self, text: str) -> float:
        """
        Predict grammar score from text only
        
        Args:
            text: Input text to score
            
        Returns:
            Predicted score (0-100)
        """
        if not self.model or not self.tokenizer:
            # Fallback: return neutral score
            return 75.0
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to score (0-100)
                # For regression, logits might be raw scores
                # Apply sigmoid and scale to 0-100
                score = torch.sigmoid(logits).item() * 100
                
                # Clamp to valid range
                score = max(0.0, min(100.0, score))
                
                return score
        except Exception as e:
            print(f"Error in text-only scoring: {e}")
            return 75.0  # Fallback score
    
    def predict_batch(self, texts: list) -> np.ndarray:
        """Predict scores for a batch of texts"""
        if not self.model or not self.tokenizer:
            return np.array([75.0] * len(texts))
        
        try:
            # Tokenize batch
            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Convert to scores
                scores = torch.sigmoid(logits).squeeze().numpy() * 100
                scores = np.clip(scores, 0, 100)
                
                return scores
        except Exception as e:
            print(f"Error in batch prediction: {e}")
            return np.array([75.0] * len(texts))

