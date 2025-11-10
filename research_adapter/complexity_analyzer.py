"""
Complexity analysis using spaCy for linguistic complexity metrics
"""

import spacy
from typing import Dict, Any, List
from collections import defaultdict


class ComplexityAnalyzer:
    """Analyze text complexity using spaCy"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Warning: spaCy model 'en_core_web_sm' not found. Some complexity features will be unavailable.")
            self.nlp = None
    
    def analyze_sentence_complexity(self, sentence: str) -> Dict[str, Any]:
        """Analyze complexity of a single sentence"""
        if not self.nlp:
            return self._default_complexity()
        
        doc = self.nlp(sentence)
        
        # Basic metrics
        word_count = len([token for token in doc if not token.is_punct])
        char_count = len(sentence)
        
        # Syntactic complexity
        avg_word_length = sum(len(token.text) for token in doc if not token.is_punct) / word_count if word_count > 0 else 0
        
        # Dependency depth
        max_depth = self._get_max_dependency_depth(doc)
        
        # Part-of-speech distribution
        pos_counts = defaultdict(int)
        for token in doc:
            if not token.is_punct:
                pos_counts[token.pos_] += 1
        
        # Noun phrases
        noun_phrases = len(list(doc.noun_chunks))
        
        # Verb complexity
        verbs = [token for token in doc if token.pos_ == "VERB"]
        verb_count = len(verbs)
        
        # Subordinate clauses (indicated by subordinating conjunctions)
        subord_conj = len([token for token in doc if token.dep_ == "mark"])
        
        return {
            'word_count': word_count,
            'char_count': char_count,
            'avg_word_length': avg_word_length,
            'max_dependency_depth': max_depth,
            'noun_phrases': noun_phrases,
            'verb_count': verb_count,
            'subordinate_clauses': subord_conj,
            'pos_distribution': dict(pos_counts),
            'complexity_score': self._calculate_complexity_score(
                word_count, max_depth, subord_conj, noun_phrases
            )
        }
    
    def _get_max_dependency_depth(self, doc) -> int:
        """Calculate maximum dependency tree depth"""
        if not doc:
            return 0
        
        def get_depth(token, visited=None):
            if visited is None:
                visited = set()
            if token in visited:
                return 0
            visited.add(token)
            
            if not list(token.children):
                return 1
            
            return 1 + max((get_depth(child, visited) for child in token.children), default=0)
        
        root = [token for token in doc if token.dep_ == "ROOT"]
        if root:
            return get_depth(root[0])
        return 0
    
    def _calculate_complexity_score(self, word_count: int, max_depth: int, 
                                   subord_clauses: int, noun_phrases: int) -> float:
        """Calculate overall complexity score (0-1, higher = more complex)"""
        # Normalize factors
        word_factor = min(1.0, word_count / 30)  # Normalize to 30 words
        depth_factor = min(1.0, max_depth / 10)  # Normalize to depth 10
        clause_factor = min(1.0, subord_clauses / 5)  # Normalize to 5 clauses
        np_factor = min(1.0, noun_phrases / 10)  # Normalize to 10 noun phrases
        
        # Weighted average
        complexity = (
            word_factor * 0.3 +
            depth_factor * 0.3 +
            clause_factor * 0.2 +
            np_factor * 0.2
        )
        
        return min(1.0, complexity)
    
    def _default_complexity(self) -> Dict[str, Any]:
        """Return default complexity metrics when spaCy is unavailable"""
        return {
            'word_count': 0,
            'char_count': 0,
            'avg_word_length': 0,
            'max_dependency_depth': 0,
            'noun_phrases': 0,
            'verb_count': 0,
            'subordinate_clauses': 0,
            'pos_distribution': {},
            'complexity_score': 0.0
        }
    
    def analyze_text_complexity(self, text: str) -> Dict[str, Any]:
        """Analyze complexity of full text"""
        if not self.nlp:
            return {'sentences': [], 'overall_complexity': 0.0}
        
        doc = self.nlp(text)
        sentences = list(doc.sents)
        
        sentence_complexities = []
        for sent in sentences:
            complexity = self.analyze_sentence_complexity(sent.text)
            sentence_complexities.append(complexity)
        
        # Calculate overall complexity
        if sentence_complexities:
            overall_complexity = sum(
                s['complexity_score'] for s in sentence_complexities
            ) / len(sentence_complexities)
        else:
            overall_complexity = 0.0
        
        return {
            'sentences': sentence_complexities,
            'overall_complexity': overall_complexity,
            'sentence_count': len(sentences),
            'total_words': sum(s['word_count'] for s in sentence_complexities)
        }

