"""
Configuration settings for the research adapter
"""

# Model configuration
MODEL_CONFIG = {
    'whisper_model': 'base.en',
    'gramformer_models': 1,
    'bert_model': 'bert-base-uncased',
    'use_gpu': False
}

# Scoring weights
ISSUE_WEIGHTS = {
    'Critical': 8,
    'Major': 6,
    'Minor': 2,
    'Suggestion': 1
}

# Feature extraction settings
FEATURE_CONFIG = {
    'max_sentence_length': 30,
    'min_sentence_length': 3,
    'speech_rate_threshold': 180,
    'pause_ratio_threshold': 0.05
}

# Ensemble settings
ENSEMBLE_CONFIG = {
    'use_bert': True,
    'use_gramformer': True,
    'use_rule_based': True,
    'ensemble_weights': {
        'bert': 0.4,
        'gramformer': 0.4,
        'rule_based': 0.2
    }
}

# GEC (Grammatical Error Correction) settings
GEC_CONFIG = {
    'model_name': 'prithivida/grammar_error_correcter_v1',  # T5 model for GEC
    'beta': 0.7,  # Weight for precision in score_gec calculation
    'gamma': 0.3,  # Weight for recall in score_gec calculation
    'use_gec': True
}

# Stacker model settings
STACKER_CONFIG = {
    'model_path': './models/stacker',  # Path to trained stacker model
    'use_calibration': True,  # Use isotonic calibration
    'fallback_mode': True,  # Use fallback when no trained model
    'text_only_model': 'microsoft/deberta-v3-base',  # Model for text-only scoring
    'text_only_model_path': None,  # Path to fine-tuned text-only model
}

# Fine-tuning settings
FINETUNE_CONFIG = {
    'base_model': 'microsoft/deberta-v3-base',  # or 'roberta-base'
    'use_roberta': False,
    'output_dir': './models/deberta_finetuned',
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 2e-5,
}

# Report generation settings
REPORT_CONFIG = {
    'include_detailed_analysis': True,
    'include_suggestions': True,
    'include_strengths': True,
    'include_gec_view': True,  # Include original/corrected split view
    'color_scheme': {
        'critical': '#dc3545',
        'major': '#fd7e14',
        'minor': '#ffc107',
        'suggestion': '#17a2b8',
        'excellent': '#28a745',
        'good': '#20c997',
        'average': '#ffc107',
        'needs_work': '#dc3545'
    }
}

