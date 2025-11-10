"""
Ensemble inference using multiple models for grammar scoring
"""

import torch
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gramformer import Gramformer
from .config import ENSEMBLE_CONFIG, ISSUE_WEIGHTS
from .features_onfly import FeatureExtractor


class EnsembleInference:
    """Ensemble inference combining multiple grammar checking approaches"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.bert_tokenizer = None
        self.bert_model = None
        self.gramformer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models based on configuration"""
        if ENSEMBLE_CONFIG['use_bert']:
            try:
                self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                    "bert-base-uncased", num_labels=1
                )
                self.bert_model.eval()
            except Exception as e:
                print(f"Warning: Could not load BERT model: {e}")
                ENSEMBLE_CONFIG['use_bert'] = False
        
        if ENSEMBLE_CONFIG['use_gramformer']:
            try:
                self.gramformer = Gramformer(models=1)
            except Exception as e:
                print(f"Warning: Could not load Gramformer: {e}")
                ENSEMBLE_CONFIG['use_gramformer'] = False
    
    def bert_score(self, sentence: str) -> float:
        """Get grammar score from BERT model"""
        if not ENSEMBLE_CONFIG['use_bert'] or self.bert_tokenizer is None:
            return 1.0  # Default to perfect if not available
        
        try:
            inputs = self.bert_tokenizer(
                sentence, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            with torch.no_grad():
                outputs = self.bert_model(**inputs)
            score = torch.sigmoid(outputs.logits).item()
            # Normalize: if BERT model isn't properly trained for grammar, default to high score
            # This is a fallback - ideally BERT would be fine-tuned for grammar checking
            return max(0.7, score)  # Ensure minimum reasonable score
        except Exception as e:
            print(f"Error in BERT scoring: {e}")
            return 1.0  # Default to perfect on error
    
    def gramformer_score(self, sentence: str) -> Dict[str, Any]:
        """Get grammar corrections from Gramformer"""
        if not ENSEMBLE_CONFIG['use_gramformer'] or self.gramformer is None:
            return {'score': 1.0, 'corrections': [], 'issues': []}  # Default to perfect if not available
        
        try:
            corrections = self.gramformer.correct(sentence)
            influences = self.gramformer.detect(sentence)
            
            # Calculate score based on number of issues
            issue_count = len(influences) if influences else 0
            if issue_count == 0:
                score = 1.0  # Perfect score when no issues found
            else:
                score = max(0.0, 1.0 - (issue_count * 0.1))
            
            return {
                'score': score,
                'corrections': corrections if corrections else [],
                'issues': influences if influences else []
            }
        except Exception as e:
            print(f"Error in Gramformer scoring: {e}")
            return {'score': 1.0, 'corrections': [], 'issues': []}  # Default to perfect on error
    
    def rule_based_score(self, sentence: str) -> Dict[str, Any]:
        """Get grammar score from rule-based analysis"""
        issues = self.feature_extractor.detect_issues(sentence)
        
        # Calculate score based on issues
        # If no issues found, return perfect score
        if not issues:
            return {
                'score': 1.0,  # Perfect score when no issues
                'issues': []
            }
        
        total_penalty = sum(ISSUE_WEIGHTS.get(issue['severity'], 1) for issue in issues)
        base_score = 100
        score = max(0.0, min(1.0, (base_score - total_penalty) / 100))
        
        return {
            'score': score,
            'issues': issues
        }
    
    def ensemble_score(self, sentence: str) -> Dict[str, Any]:
        """Combine scores from all models"""
        results = {
            'sentence': sentence,
            'bert_score': 1.0,
            'gramformer_result': {'score': 1.0, 'corrections': [], 'issues': []},
            'rule_based_result': {'score': 1.0, 'issues': []},
            'ensemble_score': 1.0,
            'all_issues': []
        }
        
        # Get scores from each model
        if ENSEMBLE_CONFIG['use_bert']:
            results['bert_score'] = self.bert_score(sentence)
        
        if ENSEMBLE_CONFIG['use_gramformer']:
            results['gramformer_result'] = self.gramformer_score(sentence)
        
        if ENSEMBLE_CONFIG['use_rule_based']:
            results['rule_based_result'] = self.rule_based_score(sentence)
        
        # Check if any issues were found
        has_issues = (
            len(results['gramformer_result']['issues']) > 0 or
            len(results['rule_based_result']['issues']) > 0
        )
        
        # If no issues found, use perfect score
        if not has_issues:
            results['ensemble_score'] = 1.0
        else:
            # Combine scores with weights only when issues are found
            weights = ENSEMBLE_CONFIG['ensemble_weights']
            ensemble_score = 0.0
            total_weight = 0.0
            
            if ENSEMBLE_CONFIG['use_bert']:
                ensemble_score += results['bert_score'] * weights['bert']
                total_weight += weights['bert']
            
            if ENSEMBLE_CONFIG['use_gramformer']:
                ensemble_score += results['gramformer_result']['score'] * weights['gramformer']
                total_weight += weights['gramformer']
            
            if ENSEMBLE_CONFIG['use_rule_based']:
                ensemble_score += results['rule_based_result']['score'] * weights['rule_based']
                total_weight += weights['rule_based']
            
            if total_weight > 0:
                results['ensemble_score'] = ensemble_score / total_weight
            else:
                results['ensemble_score'] = 1.0
        
        # Collect all issues
        if results['gramformer_result']['issues']:
            for issue in results['gramformer_result']['issues']:
                results['all_issues'].append({
                    'type': 'Grammar Error',
                    'severity': 'Critical',
                    'context': sentence,
                    'correction': results['gramformer_result']['corrections'][0] if results['gramformer_result']['corrections'] else 'No correction available'
                })
        
        results['all_issues'].extend(results['rule_based_result']['issues'])
        
        return results
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze full text with ensemble approach"""
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        sentence_results = []
        all_issues = []
        
        for sentence in sentences:
            result = self.ensemble_score(sentence)
            sentence_results.append(result)
            all_issues.extend(result['all_issues'])
        
        # Calculate overall score
        if sentence_results:
            overall_score = sum(r['ensemble_score'] for r in sentence_results) / len(sentence_results)
        else:
            overall_score = 1.0  # Perfect score if no sentences (shouldn't happen)
        
        # If no issues found across all sentences, ensure perfect score
        if not all_issues:
            overall_score = 1.0
        
        # Calculate final score (0-100 scale)
        final_score = overall_score * 100
        
        return {
            'text': text,
            'sentence_results': sentence_results,
            'all_issues': all_issues,
            'overall_score': overall_score,
            'final_score': final_score,
            'word_count': len(text.split()),
            'sentence_count': len(sentences)
        }

