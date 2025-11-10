"""
Grammatical Error Correction (GEC) module using T5 model
"""

import re
from typing import Dict, Any, List, Tuple
from collections import defaultdict

try:
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    import torch
    T5_AVAILABLE = True
except ImportError:
    T5_AVAILABLE = False
    print("Warning: transformers not available. T5 GEC features will be disabled.")


class GECModule:
    """Grammatical Error Correction using T5 model"""
    
    def __init__(self, model_name: str = "grammarly/coedit-large"):
        """
        Initialize GEC module with T5 model
        
        Args:
            model_name: HuggingFace model name for GEC (default: grammarly/coedit-large)
                       Alternative: "prithivida/grammar_error_correcter_v1" or "t5-base"
        """
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize T5 model for GEC"""
        if not T5_AVAILABLE:
            return
        
        try:
            # Try grammarly/coedit-large first (best for GEC)
            # Fallback to prithivida/grammar_error_correcter_v1 if not available
            try:
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
                self.model.eval()
                print(f"✅ GEC model loaded: {self.model_name}")
            except Exception as e:
                print(f"Warning: Could not load {self.model_name}, trying fallback...")
                fallback_model = "prithivida/grammar_error_correcter_v1"
                self.tokenizer = T5Tokenizer.from_pretrained(fallback_model)
                self.model = T5ForConditionalGeneration.from_pretrained(fallback_model)
                self.model.eval()
                self.model_name = fallback_model
                print(f"✅ GEC model loaded: {fallback_model}")
        except Exception as e:
            print(f"Warning: Could not initialize GEC model: {e}")
            self.model = None
            self.tokenizer = None
    
    def correct(self, text: str, max_length: int = 512) -> str:
        """
        Correct grammatical errors in text using T5 model
        Processes sentence by sentence for better accuracy
        
        Args:
            text: Input text to correct
            max_length: Maximum length for generation
            
        Returns:
            Corrected text
        """
        if not self.model or not self.tokenizer:
            return text
        
        try:
            # Split into sentences for better processing
            import re
            sentences = re.split(r'([.!?]+)', text)
            
            # Group sentences with their punctuation
            sentence_pairs = []
            for i in range(0, len(sentences)-1, 2):
                if i+1 < len(sentences):
                    sentence_pairs.append((sentences[i].strip(), sentences[i+1]))
            
            corrected_sentences = []
            
            for sentence, punctuation in sentence_pairs:
                if not sentence:
                    continue
                
                # Skip very short sentences (likely false positives from splitting)
                if len(sentence.split()) < 2:
                    corrected_sentences.append(sentence + punctuation)
                    continue
                
                try:
                    # Prepare input - use simpler format to avoid model artifacts
                    # The prithivida model works best with just "grammar: sentence"
                    if self.model_name.startswith("grammarly") or "coedit" in self.model_name.lower():
                        # Coedit model uses specific format
                        input_text = f"Fix grammatical errors in this sentence: {sentence}"
                    else:
                        # Standard T5 format - keep it simple
                        input_text = f"grammar: {sentence}"
                    
                    # Tokenize
                    inputs = self.tokenizer.encode(
                        input_text,
                        return_tensors="pt",
                        max_length=128,  # Shorter for individual sentences
                        truncation=True
                    )
                    
                    # Generate correction
                    with torch.no_grad():
                        outputs = self.model.generate(
                            inputs,
                            max_length=128,
                            num_beams=4,
                            early_stopping=True,
                            repetition_penalty=1.2,
                            do_sample=False
                        )
                    
                    # Decode
                    corrected_sent = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    # Clean up the output (remove prefixes and fix formatting)
                    corrected_sent = corrected_sent.strip()
                    
                    # Remove common prefixes
                    prefixes_to_remove = [
                        "grammar:",
                        "Grammar:",
                        "GRAMMAR:",
                        "Fix grammatical errors in this sentence:",
                        "fix grammatical errors in this sentence:",
                    ]
                    
                    for prefix in prefixes_to_remove:
                        if corrected_sent.lower().startswith(prefix.lower()):
                            corrected_sent = corrected_sent[len(prefix):].strip()
                    
                    # Remove any remaining "Grammar:" that might be in the middle
                    corrected_sent = corrected_sent.replace("Grammar:", "").replace("grammar:", "").strip()
                    
                    # Fix double periods and extra punctuation
                    corrected_sent = re.sub(r'\.{2,}', '.', corrected_sent)  # Remove double+ periods
                    corrected_sent = re.sub(r'\s+', ' ', corrected_sent)  # Normalize whitespace
                    
                    # If the corrected sentence is empty or seems corrupted, use original
                    if not corrected_sent or len(corrected_sent) < len(sentence) * 0.3:
                        corrected_sent = sentence
                    
                    # Validate correction: be conservative to avoid accepting bad corrections
                    # Compare word overlap and length
                    orig_words = set(sentence.lower().split())
                    corr_words = set(corrected_sent.lower().split())
                    
                    # Check 1: Word overlap should be high (at least 70%)
                    if orig_words and corr_words:
                        overlap = len(orig_words & corr_words) / len(orig_words | corr_words)
                        if overlap < 0.7:
                            corrected_sent = sentence
                    
                    # Check 2: Length shouldn't change dramatically (more than 50% increase)
                    if len(corrected_sent) > len(sentence) * 1.5:
                        corrected_sent = sentence
                    
                    # Check 3: If correction adds many new words not in original, it's likely wrong
                    new_words = corr_words - orig_words
                    if len(new_words) > len(orig_words) * 0.3:  # More than 30% new words
                        corrected_sent = sentence
                    
                    # Check 4: If the correction contains suspicious phrases, reject it
                    suspicious_phrases = [
                        "written grammar", "grammar error", "grammatical error",
                        "fix grammar", "correct grammar"
                    ]
                    corrected_lower = corrected_sent.lower()
                    if any(phrase in corrected_lower for phrase in suspicious_phrases):
                        corrected_sent = sentence
                    
                    # Preserve original punctuation (don't add extra)
                    # Remove trailing punctuation from corrected sentence if it already has punctuation
                    corrected_sent = re.sub(r'[.!?]+$', '', corrected_sent)
                    
                    # Final cleanup: ensure no "Grammar:" or similar artifacts remain
                    corrected_sent = re.sub(r'\bGrammar:\s*', '', corrected_sent, flags=re.IGNORECASE)
                    corrected_sent = re.sub(r'\bgrammar:\s*', '', corrected_sent, flags=re.IGNORECASE)
                    
                    corrected_sentences.append(corrected_sent.strip() + punctuation)
                except Exception as e:
                    # If correction fails for a sentence, keep original
                    print(f"Warning: GEC failed for sentence '{sentence}': {e}")
                    corrected_sentences.append(sentence + punctuation)
            
            # Join all corrected sentences
            corrected = " ".join(corrected_sentences)
            return corrected.strip()
        except Exception as e:
            print(f"Error in GEC correction: {e}")
            return text
    
    def correct_sentences(self, sentences: List[str]) -> List[str]:
        """Correct multiple sentences"""
        return [self.correct(sent) for sent in sentences]
    
    def compute_edit_metrics(self, original: str, corrected: str) -> Dict[str, Any]:
        """
        Compute edit-based metrics between original and corrected text
        Compares sentence by sentence for more accurate metrics
        
        Args:
            original: Original text
            corrected: Corrected text
            
        Returns:
            Dictionary with edit metrics
        """
        if original.strip() == corrected.strip():
            return {
                'num_edits': 0,
                'edit_distance': 0,
                'edit_rate': 0.0,
                'precision': 1.0,
                'recall': 1.0,
                'f1': 1.0,
                'has_changes': False,
                'original_length': len(original.split()),
                'corrected_length': len(corrected.split())
            }
        
        # Split into sentences for comparison
        import re
        orig_sentences = [s.strip() for s in re.split(r'[.!?]+', original) if s.strip()]
        corr_sentences = [s.strip() for s in re.split(r'[.!?]+', corrected) if s.strip()]
        
        # Align sentences (handle case where counts differ)
        max_sentences = max(len(orig_sentences), len(corr_sentences))
        total_edits = 0
        total_edit_distance = 0
        total_words = 0
        
        for i in range(max_sentences):
            orig_sent = orig_sentences[i] if i < len(orig_sentences) else ""
            corr_sent = corr_sentences[i] if i < len(corr_sentences) else ""
            
            if not orig_sent and not corr_sent:
                continue
            
            # Tokenize into words (case-insensitive comparison)
            orig_words = orig_sent.lower().split()
            corr_words = corr_sent.lower().split()
            
            total_words += max(len(orig_words), len(corr_words))
            
            # Compute edit distance for this sentence
            sent_edit_distance = self._word_edit_distance(orig_words, corr_words)
            total_edit_distance += sent_edit_distance
            
            # Count edits for this sentence
            sent_edits = self._count_edits(orig_words, corr_words)
            total_edits += sent_edits
        
        # Edit rate (edits per word)
        edit_rate = total_edits / total_words if total_words > 0 else 0.0
        
        # Precision and Recall calculation
        # If no edits, perfect scores
        if total_edits == 0:
            precision = 1.0
            recall = 1.0
        else:
            # Simplified: precision = 1 - normalized edit rate
            # Higher edit rate means more changes needed = lower precision
            normalized_edit_rate = min(1.0, edit_rate * 2)  # Scale edit rate
            precision = 1.0 - (normalized_edit_rate * 0.5)  # Penalize high edit rates
            
            # Recall: how many errors were caught (simplified)
            # If edit rate is low, recall is high
            recall = 1.0 - (normalized_edit_rate * 0.3)
        
        # F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'num_edits': total_edits,
            'edit_distance': total_edit_distance,
            'edit_rate': edit_rate,
            'precision': max(0.0, min(1.0, precision)),
            'recall': max(0.0, min(1.0, recall)),
            'f1': max(0.0, min(1.0, f1)),
            'has_changes': total_edits > 0,
            'original_length': len(original.split()),
            'corrected_length': len(corrected.split())
        }
    
    def _word_edit_distance(self, words1: List[str], words2: List[str]) -> int:
        """Compute word-level edit distance"""
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1].lower() == words2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # deletion
                        dp[i][j-1],      # insertion
                        dp[i-1][j-1]     # substitution
                    )
        
        return dp[m][n]
    
    def _count_edits(self, words1: List[str], words2: List[str]) -> int:
        """Count number of edits needed"""
        m, n = len(words1), len(words2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if words1[i-1].lower() == words2[j-1].lower():
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Trace back to count actual edits
        i, j = m, n
        edits = 0
        while i > 0 or j > 0:
            if i > 0 and j > 0 and words1[i-1].lower() == words2[j-1].lower():
                i -= 1
                j -= 1
            else:
                edits += 1
                if i > 0 and j > 0:
                    if dp[i-1][j-1] <= dp[i-1][j] and dp[i-1][j-1] <= dp[i][j-1]:
                        i -= 1
                        j -= 1
                    elif dp[i-1][j] < dp[i][j-1]:
                        i -= 1
                    else:
                        j -= 1
                elif i > 0:
                    i -= 1
                else:
                    j -= 1
        
        return edits

