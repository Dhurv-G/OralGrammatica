"""
Part II: Rule-based grammar checker with comprehensive rule sets
"""

import re
from typing import List, Dict, Any
from collections import defaultdict


class RuleBasedChecker:
    """Comprehensive rule-based grammar checking"""
    
    def __init__(self):
        self.rules = self._initialize_rules()
        self.category_counts = defaultdict(int)
    
    def _initialize_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize all grammar checking rules"""
        return {
            'subject_verb_agreement': [
                {
                    'pattern': r'\b(they|we|you|I)\s+(is|was)\b',
                    'category': 'Subject-Verb Agreement',
                    'severity': 'Major',
                    'message': "Remember: 'they', 'we', 'you', and 'I' go with 'are' or 'were'"
                },
                {
                    'pattern': r'\b(he|she|it)\s+(are|were)\b',
                    'category': 'Subject-Verb Agreement',
                    'severity': 'Major',
                    'message': "Remember: 'he', 'she', and 'it' go with 'is' or 'was'"
                },
                {
                    'pattern': r'\b(each|every|nobody|somebody|anybody|everyone|someone|anyone)\s+(are|were|have)\b',
                    'category': 'Subject-Verb Agreement',
                    'severity': 'Major',
                    'message': "Words like 'everyone' and 'somebody' are singular, so use 'is', 'was', or 'has'"
                }
            ],
            'tense_consistency': [
                {
                    'pattern': r'\b(was|were|had|did)\b.*\b(is|are|am|do|does)\b',
                    'category': 'Tense Consistency',
                    'severity': 'Major',
                    'message': "I noticed you're mixing past and present tenses. Usually, it's clearer to stick with one tense."
                },
                {
                    'pattern': r'\b(is|are|am)\b.*\b(was|were|had|did)\b',
                    'category': 'Tense Consistency',
                    'severity': 'Major',
                    'message': "I noticed you're mixing present and past tenses. Usually, it's clearer to stick with one tense."
                }
            ],
            'article_usage': [
                {
                    'pattern': r'\b(is|are|am)\s+([A-Z][a-z]+)\b',
                    'category': 'Article Usage',
                    'severity': 'Minor',
                    'message': "Consider adding 'a' or 'the' before the noun for clarity"
                },
                {
                    'pattern': r'\b(go|went|goes)\s+to\s+([A-Z][a-z]+)\b',
                    'category': 'Article Usage',
                    'severity': 'Minor',
                    'message': "Consider adding 'the' before the location name"
                }
            ],
            'preposition_errors': [
                {
                    'pattern': r'\bin\s+the\s+(morning|afternoon|evening)\s+of\b',
                    'category': 'Preposition Usage',
                    'severity': 'Minor',
                    'message': "Use 'on' instead of 'in' for specific dates or times"
                },
                {
                    'pattern': r'\bdepend\s+of\b',
                    'category': 'Preposition Usage',
                    'severity': 'Major',
                    'message': "Use 'depend on' instead of 'depend of'"
                }
            ],
            'double_negatives': [
                {
                    'pattern': r'\b(not|n\'t)\s+\w+\s+(no|nothing|nobody|nowhere)\b',
                    'category': 'Double Negative',
                    'severity': 'Major',
                    'message': "Avoid double negatives. Use a positive statement instead."
                }
            ],
            'redundancy': [
                {
                    'pattern': r'\b(past|previous)\s+history\b',
                    'category': 'Redundancy',
                    'severity': 'Suggestion',
                    'message': '"History" already means the past, so you can just say "history"'
                },
                {
                    'pattern': r'\b(future)\s+plans\b',
                    'category': 'Redundancy',
                    'severity': 'Suggestion',
                    'message': '"Plans" are always about the future, so you can just say "plans"'
                },
                {
                    'pattern': r'\b(repeat)\s+again\b',
                    'category': 'Redundancy',
                    'severity': 'Suggestion',
                    'message': '"Repeat" already means "do again", so you can just say "repeat"'
                }
            ],
            'sentence_structure': [
                {
                    'pattern': r'^[A-Z][^.!?]{100,}',
                    'category': 'Sentence Length',
                    'severity': 'Minor',
                    'message': "This is quite a long sentence! Breaking it into smaller parts can make your message clearer."
                },
                {
                    'pattern': r',\s*,\s*',
                    'category': 'Punctuation',
                    'severity': 'Major',
                    'message': "You have consecutive commas. Check your punctuation."
                }
            ],
            'word_choice': [
                {
                    'pattern': r'\b(very)\s+unique\b',
                    'category': 'Word Choice',
                    'severity': 'Suggestion',
                    'message': '"Unique" means one-of-a-kind, so you can just say "unique"'
                },
                {
                    'pattern': r'\b(irregardless)\b',
                    'category': 'Word Choice',
                    'severity': 'Major',
                    'message': "Use 'regardless' instead of 'irregardless'"
                }
            ],
            'apostrophe_usage': [
                {
                    'pattern': r'\bits\s+([a-z]+)\b',
                    'category': 'Apostrophe Usage',
                    'severity': 'Major',
                    'message': "Check if you meant 'it's' (it is) or 'its' (possessive)"
                },
                {
                    'pattern': r'\byour\s+([a-z]+)\b',
                    'category': 'Apostrophe Usage',
                    'severity': 'Major',
                    'message': "Check if you meant 'you're' (you are) or 'your' (possessive)"
                }
            ],
            'comma_splice': [
                {
                    'pattern': r'[a-z]\s*,\s*[a-z]',
                    'category': 'Comma Splice',
                    'severity': 'Major',
                    'message': "This might be a comma splice. Consider using a period, semicolon, or conjunction."
                }
            ]
        }
    
    def check_sentence(self, sentence: str) -> List[Dict[str, Any]]:
        """Check a single sentence against all rules"""
        issues = []
        
        for category, rule_list in self.rules.items():
            for rule in rule_list:
                matches = re.finditer(rule['pattern'], sentence, re.IGNORECASE)
                for match in matches:
                    # Avoid duplicate issues
                    context = sentence[max(0, match.start()-20):min(len(sentence), match.end()+20)]
                    
                    issue = {
                        'type': rule['category'],
                        'severity': rule['severity'],
                        'context': context,
                        'correction': f"💡 {rule['message']}",
                        'category': category,
                        'match_text': match.group(0)
                    }
                    
                    # Check if similar issue already exists
                    if not any(i['match_text'] == issue['match_text'] and 
                              i['type'] == issue['type'] for i in issues):
                        issues.append(issue)
                        self.category_counts[rule['category']] += 1
        
        return issues
    
    def check_text(self, text: str) -> Dict[str, Any]:
        """Check entire text"""
        import re
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        all_issues = []
        for sentence in sentences:
            issues = self.check_sentence(sentence)
            all_issues.extend(issues)
        
        return {
            'issues': all_issues,
            'category_counts': dict(self.category_counts),
            'total_issues': len(all_issues)
        }
    
    def reset_counts(self):
        """Reset category counts"""
        self.category_counts.clear()

