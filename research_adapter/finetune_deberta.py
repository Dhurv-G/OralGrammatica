"""
Fine-tune DeBERTa/RoBERTa for grammar score prediction
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from pathlib import Path
import os

try:
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding
    )
    from transformers import DebertaV2Tokenizer, DebertaV2ForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers not available. Fine-tuning will be disabled.")


class GrammarScoreDataset(Dataset):
    """Dataset for grammar score prediction"""
    
    def __init__(self, texts: List[str], scores: List[float], tokenizer, max_length: int = 512):
        self.texts = texts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        score = float(self.scores[idx])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(score, dtype=torch.float32)
        }


class DeBERTaFineTuner:
    """Fine-tune DeBERTa/RoBERTa for grammar score prediction"""
    
    def __init__(self, model_name: str = "microsoft/deberta-v3-base", use_roberta: bool = False):
        """
        Initialize fine-tuner
        
        Args:
            model_name: Model name (default: microsoft/deberta-v3-base)
            use_roberta: Whether to use RoBERTa instead
        """
        if use_roberta:
            self.model_name = "roberta-base"
        else:
            self.model_name = model_name
        
        self.tokenizer = None
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize model and tokenizer"""
        if not TRANSFORMERS_AVAILABLE:
            return
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=1,  # Regression task
                problem_type="regression"
            )
            print(f"✅ Initialized model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            self.model = None
            self.tokenizer = None
    
    def prepare_data(self, texts: List[str], scores: List[float], 
                    train_split: float = 0.8) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets
        
        Args:
            texts: List of input texts
            scores: List of target scores (0-100)
            train_split: Fraction for training
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer not initialized")
        
        # Split data
        split_idx = int(len(texts) * train_split)
        train_texts = texts[:split_idx]
        train_scores = scores[:split_idx]
        val_texts = texts[split_idx:]
        val_scores = scores[split_idx:]
        
        train_dataset = GrammarScoreDataset(train_texts, train_scores, self.tokenizer)
        val_dataset = GrammarScoreDataset(val_texts, val_scores, self.tokenizer)
        
        return train_dataset, val_dataset
    
    def train(self, texts: List[str], scores: List[float], 
             output_dir: str = "./models/deberta_finetuned",
             num_epochs: int = 3,
             batch_size: int = 8,
             learning_rate: float = 2e-5,
             train_split: float = 0.8):
        """
        Fine-tune the model
        
        Args:
            texts: List of input texts
            scores: List of target scores (0-100)
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            train_split: Fraction for training
        """
        if not TRANSFORMERS_AVAILABLE or not self.model:
            print("Error: Model not initialized. Cannot train.")
            return
        
        # Prepare data
        train_dataset, val_dataset = self.prepare_data(texts, scores, train_split)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=3,
        )
        
        # Data collator
        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # Custom compute_metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            # Calculate MSE and MAE
            mse = np.mean((predictions - labels) ** 2)
            mae = np.mean(np.abs(predictions - labels))
            return {
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        print("🚀 Starting fine-tuning...")
        trainer.train()
        
        # Save model
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"✅ Model fine-tuned and saved to {output_dir}")
        
        return output_dir

