"""
On-the-fly feature extraction for grammar analysis
Integrates LanguageTool and spaCy for comprehensive analysis
"""

import re
from collections import defaultdict
from typing import List, Dict, Any

try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False
    print("Warning: language-tool-python not available. LanguageTool features will be disabled.")

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    print("Warning: spaCy not available. Some features will be disabled.")

from .gec_module import GECModule
from .config import GEC_CONFIG


class FeatureExtractor:
    """Extract linguistic features from text on the fly"""
    
    def __init__(self):
        # Initialize LanguageTool
        self.language_tool = None
        if LANGUAGETOOL_AVAILABLE:
            try:
                self.language_tool = language_tool_python.LanguageTool('en-US')
            except Exception as e:
                print(f"Warning: Could not initialize LanguageTool: {e}")
        
        # Initialize spaCy
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("Warning: spaCy model 'en_core_web_sm' not found.")
        
        # Initialize GEC module
        self.gec_module = None
        if GEC_CONFIG.get('use_gec', True):
            try:
                self.gec_module = GECModule(model_name=GEC_CONFIG.get('model_name', 'prithivida/grammar_error_correcter_v1'))
            except Exception as e:
                print(f"Warning: Could not initialize GEC module: {e}")
        
        self.contractions = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "couldn't": "could not", "didn't": "did not", "doesn't": "does not",
            "don't": "do not", "hadn't": "had not", "hasn't": "has not",
            "haven't": "have not", "he'd": "he would", "he'll": "he will",
            "he's": "he is", "I'd": "I would", "I'll": "I will", "I'm": "I am",
            "I've": "I have", "isn't": "is not", "it's": "it is", "let's": "let us",
            "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "shouldn't": "should not", "that's": "that is", "there's": "there is",
            "they'd": "they would", "they'll": "they will", "they're": "they are",
            "they've": "they have", "we'd": "we would", "we're": "we are",
            "we've": "we have", "weren't": "were not", "what'll": "what will",
            "what're": "what are", "what's": "what is", "what've": "what have",
            "where's": "where is", "who'd": "who would", "who'll": "who will",
            "who're": "who are", "who's": "who is", "who've": "who have",
            "won't": "will not", "wouldn't": "would not", "you'd": "you would",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
    
    def extract_sentence_features(self, sentence: str) -> Dict[str, Any]:
        """Extract features from a single sentence"""
        words = sentence.split()
        word_count = len(words)
        
        features = {
            'word_count': word_count,
            'char_count': len(sentence),
            'avg_word_length': sum(len(w) for w in words) / word_count if word_count > 0 else 0,
            'punctuation_count': len(re.findall(r'[,.!?;:]', sentence)),
            'capital_letters': len(re.findall(r'[A-Z]', sentence)),
            'has_question': '?' in sentence,
            'has_exclamation': '!' in sentence,
            'has_comma': ',' in sentence,
            'contraction_count': sum(1 for word in words if word.lower() in self.contractions),
            'digit_count': len(re.findall(r'\d', sentence)),
            'special_char_count': len(re.findall(r'[^a-zA-Z0-9\s.,!?;:]', sentence))
        }
        
        return features
    
    def extract_grammar_features(self, sentence: str) -> Dict[str, Any]:
        """Extract grammar-specific features"""
        features = {
            'tense_markers': {
                'past': len(re.findall(r'\b(was|were|had|did|went|came|saw|made|said|took|got|thought)\b', sentence, re.I)),
                'present': len(re.findall(r'\b(is|are|am|do|does|go|come|see|make|say|take|get|think)\b', sentence, re.I)),
                'future': len(re.findall(r'\b(will|shall|going to|about to)\b', sentence, re.I))
            },
            'modal_verbs': len(re.findall(r'\b(should|could|would|must|might|may|can)\b', sentence, re.I)),
            'articles': len(re.findall(r'\b(a|an|the)\b', sentence, re.I)),
            'pronouns': len(re.findall(r'\b(I|you|he|she|it|we|they|me|him|her|us|them)\b', sentence, re.I)),
            'prepositions': len(re.findall(r'\b(in|on|at|to|for|of|with|by|from|about|into|onto)\b', sentence, re.I)),
            'conjunctions': len(re.findall(r'\b(and|or|but|so|because|if|when|while|although)\b', sentence, re.I)),
            'subject_verb_patterns': len(re.findall(r'\b(they|we|you|I)\s+(is|was)\b|\b(he|she|it)\s+(are|were)\b', sentence, re.I))
        }
        
        return features
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract features from full text"""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        text_features = {
            'sentence_count': len(sentences),
            'total_words': len(text.split()),
            'avg_sentence_length': sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0,
            'longest_sentence': max((len(s.split()) for s in sentences), default=0),
            'shortest_sentence': min((len(s.split()) for s in sentences), default=0),
            'sentence_features': [self.extract_sentence_features(s) for s in sentences],
            'grammar_features': [self.extract_grammar_features(s) for s in sentences]
        }
        
        return text_features
    
    def detect_issues(self, sentence: str) -> List[Dict[str, Any]]:
        """Detect grammar issues in a sentence"""
        issues = []
        words = sentence.split()
        
        # Check sentence length
        if len(words) > 30:
            issues.append({
                'type': 'Sentence Length',
                'severity': 'Minor',
                'context': sentence,
                'correction': "💡 This is quite a long sentence! Breaking it into smaller parts can make your message clearer."
            })
        
        # Check for subject-verb agreement
        sv_patterns = [
            (r'\b(they|we|you|I)\s+(is|was)\b', "Remember: 'they', 'we', 'you', and 'I' go with 'are' or 'were'"),
            (r'\b(he|she|it)\s+(are|were)\b', "Remember: 'he', 'she', and 'it' go with 'is' or 'was'"),
        ]
        
        for pattern, msg in sv_patterns:
            if re.search(pattern, sentence, re.I):
                issues.append({
                    'type': 'Subject-Verb Agreement',
                    'severity': 'Major',
                    'context': sentence,
                    'correction': f"🤔 {msg}"
                })
        
        # Check tense consistency
        past_markers = r'\b(was|were|had|did|went)\b'
        present_markers = r'\b(is|are|am|do|does)\b'
        
        has_past = bool(re.search(past_markers, sentence, re.I))
        has_present = bool(re.search(present_markers, sentence, re.I))
        
        if has_past and has_present:
            issues.append({
                'type': 'Tense Consistency',
                'severity': 'Major',
                'context': sentence,
                'correction': "🤔 I noticed you're mixing tenses here. Usually, it's clearer to stick with one tense."
            })
        
        return issues
    
    def get_languagetool_matches(self, text: str) -> Dict[str, Any]:
        """Get grammar matches from LanguageTool"""
        if not self.language_tool:
            return {
                'matches': [],
                'match_count': 0,
                'categories': defaultdict(int),
                'rule_ids': []
            }
        
        try:
            matches = self.language_tool.check(text)
            
            # Categorize matches
            categories = defaultdict(int)
            rule_ids = []
            
            for match in matches:
                category = match.category if hasattr(match, 'category') else 'Other'
                categories[category] += 1
                rule_ids.append(match.ruleId if hasattr(match, 'ruleId') else 'unknown')
            
            return {
                'matches': matches,
                'match_count': len(matches),
                'categories': dict(categories),
                'rule_ids': rule_ids,
                'issues': self._convert_languagetool_to_issues(matches, text)
            }
        except Exception as e:
            print(f"Error in LanguageTool checking: {e}")
            return {
                'matches': [],
                'match_count': 0,
                'categories': defaultdict(int),
                'rule_ids': [],
                'issues': []
            }
    
    def _convert_languagetool_to_issues(self, matches, text: str) -> List[Dict[str, Any]]:
        """Convert LanguageTool matches to issue format"""
        issues = []
        
        for match in matches:
            # Determine severity based on rule type
            severity = 'Minor'
            if hasattr(match, 'ruleId'):
                rule_id = match.ruleId.lower()
                if any(keyword in rule_id for keyword in ['grammar', 'spelling', 'typos']):
                    severity = 'Major'
                elif any(keyword in rule_id for keyword in ['style', 'punct']):
                    severity = 'Suggestion'
            
            # Get context
            start = max(0, match.offset - 20)
            end = min(len(text), match.offset + match.errorLength + 20)
            context = text[start:end]
            
            # Get suggestions
            suggestions = match.replacements[:3] if hasattr(match, 'replacements') else []
            correction_msg = f"💡 Consider: {', '.join(suggestions)}" if suggestions else "💡 Check this part of the text"
            
            issues.append({
                'type': match.category if hasattr(match, 'category') else 'Grammar',
                'severity': severity,
                'context': context,
                'correction': correction_msg,
                'rule_id': match.ruleId if hasattr(match, 'ruleId') else 'unknown',
                'source': 'LanguageTool'
            })
        
        return issues
    
    def get_complexity_stats(self, text: str) -> Dict[str, Any]:
        """Get complexity statistics using spaCy"""
        if not self.nlp:
            return {
                'complexity_score': 0.0,
                'avg_sentence_length': 0,
                'avg_word_length': 0,
                'syntactic_complexity': 0,
                'pos_distribution': {},
                'dependency_depth': 0
            }
        
        try:
            doc = self.nlp(text)
            sentences = list(doc.sents)
            
            if not sentences:
                return self._default_complexity_stats()
            
            # Calculate metrics
            total_words = sum(len([t for t in sent if not t.is_punct]) for sent in sentences)
            avg_sentence_length = total_words / len(sentences) if sentences else 0
            
            avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / total_words if total_words > 0 else 0
            
            # Dependency depth
            max_depths = []
            for sent in sentences:
                root = [token for token in sent if token.dep_ == "ROOT"]
                if root:
                    depth = self._calculate_dependency_depth(root[0])
                    max_depths.append(depth)
            avg_dependency_depth = sum(max_depths) / len(max_depths) if max_depths else 0
            
            # POS distribution
            pos_distribution = defaultdict(int)
            for token in doc:
                if not token.is_punct:
                    pos_distribution[token.pos_] += 1
            
            # Syntactic complexity (based on subordinate clauses, noun phrases, etc.)
            subord_conj = len([token for token in doc if token.dep_ == "mark"])
            noun_phrases = len(list(doc.noun_chunks))
            syntactic_complexity = (subord_conj + noun_phrases) / len(sentences) if sentences else 0
            
            # Overall complexity score (0-1)
            complexity_score = min(1.0, (
                (avg_sentence_length / 30) * 0.3 +
                (avg_dependency_depth / 10) * 0.3 +
                (syntactic_complexity / 5) * 0.4
            ))
            
            return {
                'complexity_score': complexity_score,
                'avg_sentence_length': avg_sentence_length,
                'avg_word_length': avg_word_length,
                'syntactic_complexity': syntactic_complexity,
                'pos_distribution': dict(pos_distribution),
                'dependency_depth': avg_dependency_depth,
                'subordinate_clauses': subord_conj,
                'noun_phrases': noun_phrases
            }
        except Exception as e:
            print(f"Error in complexity analysis: {e}")
            return self._default_complexity_stats()
    
    def _calculate_dependency_depth(self, token, visited=None) -> int:
        """Calculate dependency tree depth"""
        if visited is None:
            visited = set()
        if token in visited:
            return 0
        visited.add(token)
        
        children = list(token.children)
        if not children:
            return 1
        
        return 1 + max((self._calculate_dependency_depth(child, visited) for child in children), default=0)
    
    def _default_complexity_stats(self) -> Dict[str, Any]:
        """Return default complexity stats when spaCy is unavailable"""
        return {
            'complexity_score': 0.0,
            'avg_sentence_length': 0,
            'avg_word_length': 0,
            'syntactic_complexity': 0,
            'pos_distribution': {},
            'dependency_depth': 0,
            'subordinate_clauses': 0,
            'noun_phrases': 0
        }
    
    def run_gec(self, text: str) -> Dict[str, Any]:
        """
        Run T5 correction and compute edit metrics & score_gec
        
        Args:
            text: Input text to correct
            
        Returns:
            Dictionary with corrected text, edit metrics, and score_gec
        """
        if not self.gec_module:
            return {
                'original': text,
                'corrected': text,
                'edit_metrics': {
                    'num_edits': 0,
                    'edit_distance': 0,
                    'edit_rate': 0.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'has_changes': False
                },
                'score_gec': 100.0
            }
        
        try:
            # Correct the text
            corrected = self.gec_module.correct(text)
            
            # Compute edit metrics
            edit_metrics = self.gec_module.compute_edit_metrics(text, corrected)
            
            # Compute score_gec using beta and gamma weights
            beta = GEC_CONFIG.get('beta', 0.7)
            gamma = GEC_CONFIG.get('gamma', 0.3)
            
            precision = edit_metrics['precision']
            recall = edit_metrics['recall']
            
            # Score based on precision and recall, weighted by beta and gamma
            # Higher precision and recall = higher score
            score_gec = (beta * precision + gamma * recall) * 100
            
            # If no changes were made, give perfect score
            if not edit_metrics['has_changes']:
                score_gec = 100.0
            
            return {
                'original': text,
                'corrected': corrected,
                'edit_metrics': edit_metrics,
                'score_gec': score_gec
            }
        except Exception as e:
            print(f"Error in GEC processing: {e}")
            return {
                'original': text,
                'corrected': text,
                'edit_metrics': {
                    'num_edits': 0,
                    'edit_distance': 0,
                    'edit_rate': 0.0,
                    'precision': 1.0,
                    'recall': 1.0,
                    'f1': 1.0,
                    'has_changes': False
                },
                'score_gec': 100.0
            }

