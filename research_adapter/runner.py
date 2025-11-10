"""
Main runner for grammar scoring from transcript
"""

from typing import Dict, Any
from .ensemble_infer import EnsembleInference
from .html_report import HTMLReportGenerator
from .rule_based_checker import RuleBasedChecker
from .features_onfly import FeatureExtractor
from .text_only_scorer import TextOnlyScorer
from .stacker_model import StackerModel
from .config import ISSUE_WEIGHTS, STACKER_CONFIG


def score_rule(transcript: str) -> Dict[str, Any]:
    """
    Compute rule-based score from transcript
    
    Args:
        transcript: The transcript text to analyze
        
    Returns:
        Dictionary with rule-based analysis results including score and category counts
    """
    # Initialize rule-based checker
    rule_checker = RuleBasedChecker()
    feature_extractor = FeatureExtractor()
    
    # Get rule-based issues
    rule_results = rule_checker.check_text(transcript)
    
    # Get LanguageTool matches
    lt_results = feature_extractor.get_languagetool_matches(transcript)
    
    # Get complexity stats
    complexity_stats = feature_extractor.get_complexity_stats(transcript)
    
    # Run GEC (T5 correction) and compute edit metrics & score_gec
    gec_results = feature_extractor.run_gec(transcript)
    
    # Combine all issues (with defensive checks)
    rule_issues = rule_results.get('issues', [])
    lt_issues = lt_results.get('issues', [])
    all_issues = rule_issues + lt_issues
    
    # Calculate rule-based score
    total_penalty = 0
    for issue in all_issues:
        severity = issue.get('severity', 'Minor')
        penalty = ISSUE_WEIGHTS.get(severity, 1)
        total_penalty += penalty
    
    base_score = 100
    word_count = len(transcript.split())
    
    # Normalize penalty by word count
    if word_count > 0:
        normalized_penalty = total_penalty * (40 / max(word_count, 40))
    else:
        normalized_penalty = total_penalty
    
    rule_score = max(0, min(100, base_score - normalized_penalty))
    
    # Combine category counts (with defensive checks)
    category_counts = rule_results.get('category_counts', {}).copy()
    lt_categories = lt_results.get('categories', {})
    for category, count in lt_categories.items():
        category_counts[category] = category_counts.get(category, 0) + count
    
    return {
        'rule_score': rule_score,
        'total_issues': len(all_issues),
        'category_counts': category_counts,
        'issues': all_issues,
        'complexity_stats': complexity_stats,
        'languagetool_matches': lt_results.get('match_count', 0),
        'rule_based_issues': len(rule_issues),
        'gec_results': gec_results
    }


def score_grammar_from_transcript(transcript: str) -> str:
    """
    Score grammar from a transcript string and return HTML report
    
    Args:
        transcript: The transcript text to analyze
        
    Returns:
        HTML string containing the grammar analysis report
    """
    # Initialize components
    ensemble = EnsembleInference()
    report_generator = HTMLReportGenerator()
    
    # Analyze the transcript with ensemble
    analysis_result = ensemble.analyze_text(transcript)
    
    # Get rule-based score and category counts
    rule_results = score_rule(transcript)
    
    # Add rule-based results to analysis result
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
    text_scorer = TextOnlyScorer(
        model_name=STACKER_CONFIG.get('text_only_model', 'microsoft/deberta-v3-base'),
        use_finetuned=STACKER_CONFIG.get('text_only_model_path') is not None,
        model_path=STACKER_CONFIG.get('text_only_model_path')
    )
    analysis_result['score_text_only'] = text_scorer.score_text_only(transcript)
    
    # Combine all issues (ensemble + rule-based + LanguageTool)
    analysis_result['all_issues'].extend(rule_results['issues'])
    
    # Use stacker to predict final score
    stacker = StackerModel(
        model_path=STACKER_CONFIG.get('model_path'),
        use_calibration=STACKER_CONFIG.get('use_calibration', True)
    )
    
    # Predict final score using stacker
    final_score = stacker.predict(analysis_result)
    analysis_result['final_score'] = final_score
    
    # Generate HTML report
    html_report = report_generator.generate_html(analysis_result, transcript)
    
    return html_report

