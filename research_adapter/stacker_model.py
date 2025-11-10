"""
Stacker model that combines multiple features to predict final grammar score
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import pickle
import os
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Stacker will use fallback mode.")


class StackerModel:
    """
    Stacker model that combines:
    - score_rule
    - score_gec
    - score_text_only
    - GEC edit metrics
    - category counts
    - complexity features
    - (optional) ASR meta
    """
    
    def __init__(self, model_path: Optional[str] = None, use_calibration: bool = True):
        """
        Initialize stacker model
        
        Args:
            model_path: Path to saved model artifacts
            use_calibration: Whether to use isotonic calibration
        """
        self.model_path = model_path
        self.use_calibration = use_calibration
        self.model = None
        self.scaler = None
        self.calibrator = None
        self.feature_names = []
        self.is_trained = False
        
        # Check if model exists (check for model file, not just directory)
        if model_path:
            model_file = os.path.join(model_path, 'stacker_model.pkl')
            if os.path.exists(model_file):
                self.load_model(model_path)
            else:
                # Initialize with default fallback mode
                self._initialize_fallback()
        else:
            # Initialize with default fallback mode
            self._initialize_fallback()
    
    def _initialize_fallback(self):
        """Initialize in fallback mode (no trained model)"""
        # Don't create untrained models - just mark as not trained
        # The _fallback_predict method will handle prediction
        self.is_trained = False
    
    def extract_features(self, analysis_result: Dict[str, Any]) -> np.ndarray:
        """
        Extract features from analysis result for stacker
        
        Args:
            analysis_result: Dictionary with all analysis results
            
        Returns:
            Feature vector as numpy array
        """
        features = []
        feature_names = []
        
        # 1. Score features
        score_rule = analysis_result.get('rule_score', 75.0)
        score_gec = analysis_result.get('score_gec', 75.0)
        score_text_only = analysis_result.get('score_text_only', 75.0)
        
        features.extend([score_rule, score_gec, score_text_only])
        feature_names.extend(['score_rule', 'score_gec', 'score_text_only'])
        
        # 2. GEC edit metrics
        edit_metrics = analysis_result.get('edit_metrics', {})
        features.append(edit_metrics.get('num_edits', 0))
        features.append(edit_metrics.get('edit_rate', 0.0))
        features.append(edit_metrics.get('precision', 1.0))
        features.append(edit_metrics.get('recall', 1.0))
        features.append(edit_metrics.get('f1', 1.0))
        features.append(edit_metrics.get('edit_distance', 0))
        feature_names.extend(['gec_num_edits', 'gec_edit_rate', 'gec_precision', 
                             'gec_recall', 'gec_f1', 'gec_edit_distance'])
        
        # 3. Category counts (top categories)
        category_counts = analysis_result.get('category_counts', {})
        top_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        category_dict = dict(top_categories)
        
        # Common categories
        common_categories = [
            'Subject-Verb Agreement', 'Tense Consistency', 'Article Usage',
            'Preposition Usage', 'Redundancy', 'Sentence Length', 'Word Choice',
            'Apostrophe Usage', 'Comma Splice', 'Grammar'
        ]
        
        for cat in common_categories:
            features.append(category_counts.get(cat, 0))
            feature_names.append(f'category_{cat.replace(" ", "_").lower()}')
        
        # Total category count
        total_categories = sum(category_counts.values())
        features.append(total_categories)
        feature_names.append('total_category_count')
        
        # 4. Complexity features
        complexity_stats = analysis_result.get('complexity_stats', {})
        features.append(complexity_stats.get('complexity_score', 0.0))
        features.append(complexity_stats.get('avg_sentence_length', 0.0))
        features.append(complexity_stats.get('avg_word_length', 0.0))
        features.append(complexity_stats.get('dependency_depth', 0.0))
        features.append(complexity_stats.get('subordinate_clauses', 0))
        features.append(complexity_stats.get('noun_phrases', 0))
        feature_names.extend(['complexity_score', 'avg_sentence_length', 'avg_word_length',
                             'dependency_depth', 'subordinate_clauses', 'noun_phrases'])
        
        # 5. Basic text features
        word_count = analysis_result.get('word_count', 0)
        sentence_count = analysis_result.get('sentence_count', 0)
        issue_count = len(analysis_result.get('all_issues', []))
        
        features.extend([word_count, sentence_count, issue_count])
        feature_names.extend(['word_count', 'sentence_count', 'issue_count'])
        
        # 6. Ratio features
        if word_count > 0:
            features.append(issue_count / word_count)  # Issues per word
            features.append(sentence_count / word_count if word_count > 0 else 0)  # Sentences per word
        else:
            features.extend([0.0, 0.0])
        feature_names.extend(['issues_per_word', 'sentences_per_word'])
        
        # 7. (Optional) ASR meta - placeholder for future
        # asr_meta = analysis_result.get('asr_meta', {})
        # features.append(asr_meta.get('confidence', 1.0))
        # feature_names.append('asr_confidence')
        
        self.feature_names = feature_names
        return np.array(features, dtype=np.float32)
    
    def predict(self, analysis_result: Dict[str, Any]) -> float:
        """
        Predict final score using stacker
        
        Args:
            analysis_result: Dictionary with all analysis results
            
        Returns:
            Final predicted score (0-100)
        """
        if not self.is_trained:
            # Fallback mode: use weighted average
            return self._fallback_predict(analysis_result)
        
        # Extract features
        features = self.extract_features(analysis_result).reshape(1, -1)
        
        # Scale features
        if self.scaler:
            features = self.scaler.transform(features)
        
        # Predict
        raw_score = self.model.predict(features)[0]
        
        # Apply calibration
        if self.use_calibration and self.calibrator:
            raw_score = self.calibrator.transform([raw_score])[0]
        
        # Clamp to valid range
        score = max(0.0, min(100.0, raw_score))
        
        return score
    
    def _fallback_predict(self, analysis_result: Dict[str, Any]) -> float:
        """Fallback prediction when model is not trained"""
        # Use weighted average of available scores
        score_rule = analysis_result.get('rule_score', 75.0)
        score_gec = analysis_result.get('score_gec', 75.0)
        score_text_only = analysis_result.get('score_text_only', 75.0)
        
        # Weights for fallback
        weights = {
            'rule': 0.4,
            'gec': 0.4,
            'text_only': 0.2
        }
        
        # Use available scores
        available_scores = []
        available_weights = []
        
        if 'rule_score' in analysis_result:
            available_scores.append(score_rule)
            available_weights.append(weights['rule'])
        if 'score_gec' in analysis_result:
            available_scores.append(score_gec)
            available_weights.append(weights['gec'])
        if 'score_text_only' in analysis_result:
            available_scores.append(score_text_only)
            available_weights.append(weights['text_only'])
        
        if not available_scores:
            return 75.0  # Default fallback
        
        # Normalize weights
        total_weight = sum(available_weights)
        if total_weight > 0:
            available_weights = [w / total_weight for w in available_weights]
        
        # Weighted average
        final_score = sum(s * w for s, w in zip(available_scores, available_weights))
        
        return max(0.0, min(100.0, final_score))
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2):
        """
        Train the stacker model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target scores (n_samples,)
            validation_split: Fraction of data to use for validation
        """
        if not SKLEARN_AVAILABLE:
            print("Error: scikit-learn not available. Cannot train stacker.")
            return
        
        # Split data
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Train calibrator
        if self.use_calibration:
            # Get predictions on validation set
            y_pred_val = self.model.predict(X_val_scaled)
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_pred_val, y_val)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        val_score = self.model.score(X_val_scaled, y_val)
        
        print(f"✅ Stacker trained:")
        print(f"   Train R²: {train_score:.4f}")
        print(f"   Val R²: {val_score:.4f}")
        
        self.is_trained = True
    
    def save_model(self, save_path: str):
        """Save model artifacts"""
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        if self.model:
            joblib.dump(self.model, os.path.join(save_path, 'stacker_model.pkl'))
        
        # Save scaler
        if self.scaler:
            joblib.dump(self.scaler, os.path.join(save_path, 'scaler.pkl'))
        
        # Save calibrator
        if self.calibrator:
            joblib.dump(self.calibrator, os.path.join(save_path, 'calibrator.pkl'))
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
            'use_calibration': self.use_calibration
        }
        
        with open(os.path.join(save_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✅ Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """Load model artifacts"""
        if not SKLEARN_AVAILABLE:
            print("Warning: scikit-learn not available. Cannot load model.")
            return
        
        # Load model
        model_file = os.path.join(load_path, 'stacker_model.pkl')
        if os.path.exists(model_file):
            self.model = joblib.load(model_file)
        
        # Load scaler
        scaler_file = os.path.join(load_path, 'scaler.pkl')
        if os.path.exists(scaler_file):
            self.scaler = joblib.load(scaler_file)
        
        # Load calibrator
        calibrator_file = os.path.join(load_path, 'calibrator.pkl')
        if os.path.exists(calibrator_file):
            self.calibrator = joblib.load(calibrator_file)
        
        # Load metadata
        metadata_file = os.path.join(load_path, 'metadata.pkl')
        if os.path.exists(metadata_file):
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                self.feature_names = metadata.get('feature_names', [])
                self.is_trained = metadata.get('is_trained', False)
                self.use_calibration = metadata.get('use_calibration', True)
        
        if self.is_trained:
            print(f"✅ Model loaded from {load_path}")

