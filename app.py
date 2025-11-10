# Grammar Checker - Your Friendly Writing Assistant
# This tool helps you improve your spoken and written English with gentle, helpful feedback

import whisper
import re
from collections import defaultdict
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from gramformer import Gramformer
import time
import sys
from flask import Flask, render_template_string, request, jsonify
from research_adapter import score_grammar_from_transcript
import os

# Download required NLTK data with user-friendly messages
print("🔧 Setting up your grammar checker...")
try:
    nltk.data.find('tokenizers/punkt')
    print("✅ NLTK data already available")
except LookupError:
    print("📥 Downloading language processing tools...")
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    print("✅ Language tools downloaded successfully")

print("🤖 Loading AI models (this may take a moment)...")
model = whisper.load_model("base.en")  # or "small.en" for better accuracy/speed trade-off
print("✅ Speech recognition model loaded")

class GrammarScorer:
    def __init__(self):
        print("🧠 Initializing grammar analysis engine...")
        # Initialize BERT for grammar checking
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=1)
        print("✅ Grammar analysis engine ready")
        
        # Scoring weights with encouraging messaging
        self.issue_weights = {
            'Critical': 8,       # "Let's fix this together"
            'Major': 6,         # "This needs attention"
            'Minor': 2,         # "Small improvement opportunity"
            'Suggestion': 1     # "Food for thought"
        }
        
        # Common contractions and their expanded forms
        self.contractions = {
            "ain't": "is not",
            "aren't": "are not",
            "can't": "cannot",
            "couldn't": "could not",
            "didn't": "did not",
            "doesn't": "does not",
            "don't": "do not",
            "hadn't": "had not",
            "hasn't": "has not",
            "haven't": "have not",
            "he'd": "he would",
            "he'll": "he will",
            "he's": "he is",
            "I'd": "I would",
            "I'll": "I will",
            "I'm": "I am",
            "I've": "I have",
            "isn't": "is not",
            "it's": "it is",
            "let's": "let us",
            "mightn't": "might not",
            "mustn't": "must not",
            "shan't": "shall not",
            "she'd": "she would",
            "she'll": "she will",
            "she's": "she is",
            "shouldn't": "should not",
            "that's": "that is",
            "there's": "there is",
            "they'd": "they would",
            "they'll": "they will",
            "they're": "they are",
            "they've": "they have",
            "we'd": "we would",
            "we're": "we are",
            "we've": "we have",
            "weren't": "were not",
            "what'll": "what will",
            "what're": "what are",
            "what's": "what is",
            "what've": "what have",
            "where's": "where is",
            "who'd": "who would",
            "who'll": "who will",
            "who're": "who are",
            "who's": "who is",
            "who've": "who have",
            "won't": "will not",
            "wouldn't": "would not",
            "you'd": "you would",
            "you'll": "you will",
            "you're": "you are",
            "you've": "you have"
        }
        
        # Initialize confidence tracking
        self.confidence_scores = []
        
    def check_grammar_bert(self, sentence):
        """Use BERT to get a grammar quality score"""
        inputs = self.bert_tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
        score = torch.sigmoid(outputs.logits).item()
        return score
        
    def check_prosodic_patterns(self, sentence):
        """Analyze speech patterns and rhythm with friendly suggestions"""
        issues = []
        
        # Check for speech rate (words per minute)
        words = sentence.split()
        speech_rate = len(words) / (len(sentence) * 0.006)  # Approximate
        
        if speech_rate > 180:  # Too fast
            issues.append({
                'type': 'Speech Rate',
                'severity': 'Minor',
                'context': sentence,
                'correction': "💡 Speaking tip: Try slowing down a bit for better clarity. Your audience will thank you!"
            })
        
        # Check for pauses using punctuation as proxy
        pause_markers = re.findall(r'[,.!?;:]', sentence)
        if len(pause_markers) < len(words) / 20:  # Not enough pauses
            issues.append({
                'type': 'Pausing',
                'severity': 'Suggestion',
                'context': sentence,
                'correction': "💡 Consider adding natural pauses - they help your audience follow along and emphasize key points"
            })
        
        return issues

    def analyze_sentence_structure(self, sentence):
        """Comprehensive sentence structure analysis with encouraging feedback"""
        issues = []
        words = sentence.split()
        
        # Check sentence length
        if len(words) > 30:
            issues.append({
                'type': 'Sentence Length',
                'severity': 'Minor',
                'context': sentence,
                'correction': "💡 This is quite a long sentence! Breaking it into smaller parts can make your message clearer and easier to follow"
            })
        
        # Check for compound sentence issues
        if ',' in sentence:
            parts = sentence.split(',')
            for part in parts:
                # Check if each part can stand as an independent clause
                if re.search(r'\b(I|he|she|it|we|they|you)\b.*\b(is|am|are|was|were|has|have|had)\b', part.strip()):
                    issues.append({
                        'type': 'Compound Sentence',
                        'severity': 'Minor',
                        'context': sentence,
                        'correction': "💡 Consider using connecting words (like 'and', 'but', 'because') or splitting into separate sentences for better flow"
                    })
                    break
        
        # Check for repeated words
        word_counts = defaultdict(int)
        for word in words:
            word_counts[word.lower()] += 1
        for word, count in word_counts.items():
            if count > 3 and word not in {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'of', 'that', 'this', 'these', 'those', 'with', 'for'}:
                issues.append({
                    'type': 'Word Repetition',
                    'severity': 'Suggestion',
                    'context': sentence,
                    'correction': f"💡 The word '{word}' appears several times. Try using synonyms to add variety and keep your audience engaged!"
                })
        
        return issues

    def check_tense_consistency(self, sentence):
        """Enhanced tense consistency checking with helpful explanations"""
        issues = []
        
        # Define tense patterns
        past_markers = r'\b(was|were|had|did|went|came|saw|made|said|took|got|thought|called|looked|wanted|needed|seemed|felt|became|started)\b'
        present_markers = r'\b(is|are|am|do|does|go|come|see|make|say|take|get|think|call|look|want|need|seem|feel|become|start)\b'
        future_markers = r'\b(will|shall|going to|about to|plan to|intend to|expect to)\b'
        
        # Check for legitimate mixed tense cases
        legitimate_mixed = [
            r'(always|usually|often|sometimes|rarely|never)\s+\w+',  # Habitual actions
            r'(while|as|when)\s+\w+',  # Simultaneous actions
            r'(since|for)\s+\w+',  # Duration
            r'if\s+\w+',  # Conditional statements
        ]
        
        # Detect tenses
        tenses_found = []
        if re.search(past_markers, sentence, re.I): tenses_found.append('past')
        if re.search(present_markers, sentence, re.I): tenses_found.append('present')
        if re.search(future_markers, sentence, re.I): tenses_found.append('future')
        
        # Check mixed tenses
        if len(tenses_found) > 1:
            is_legitimate = any(re.search(pattern, sentence, re.I) for pattern in legitimate_mixed)
            if not is_legitimate:
                issues.append({
                    'type': 'Tense Consistency',
                    'severity': 'Major',
                    'context': sentence,
                    'correction': f"🤔 I noticed you're mixing {', '.join(tenses_found)} tenses here. Usually, it's clearer to stick with one tense unless you're describing different time periods. What do you think?"
                })
        
        # Check modal verb constructions
        should_have_pattern = r'\b(should|could|would|must) have \w+ed\b'
        if re.search(should_have_pattern, sentence):
            # Verify past participle usage
            match = re.search(r'\b(should|could|would|must) have (\w+ed|\w+en|\w+t)\b', sentence)
            if match and not self.is_valid_past_participle(match.group(2)):
                issues.append({
                    'type': 'Modal Verb Usage',
                    'severity': 'Major',
                    'context': sentence,
                    'correction': "🤔 When using 'should have', 'could have', etc., make sure to use the correct past participle form"
                })
        
        return issues

    def check_subject_verb_agreement(self, sentence):
        """Enhanced subject-verb agreement checking with friendly explanations"""
        issues = []
        
        # Enhanced patterns with helpful messages
        patterns = [
            (r'\b(they|we|you|I)\s+(is|was)\b', "Remember: 'they', 'we', 'you', and 'I' go with 'are' or 'were'"),
            (r'\b(he|she|it)\s+(are|were)\b', "Remember: 'he', 'she', and 'it' go with 'is' or 'was'"),
            (r'\b(the|a|an)\s+\w+s\s+is\b', "If the noun ends with 's' (plural), use 'are' instead of 'is'"),
            (r'\b(the|a|an)\s+\w+[^s]\s+are\b', "If the noun doesn't end with 's' (singular), use 'is' instead of 'are'"),
            (r'\b(each|every|nobody|somebody|anybody|everyone|someone|anyone)\s+(are|were|have)\b', "Words like 'everyone' and 'somebody' are singular, so use 'is', 'was', or 'has'"),
            (r'\b(none|neither|either)\s+of\s+the\s+\w+s?\s+(is|are)\b', "With 'none of', 'neither of', or 'either of', the verb usually matches the noun after 'of'"),
            (r'\b(one\s+of\s+the\s+\w+s)\s+are\b', "Use 'is' with 'one of the' since you're talking about one thing"),
            (r'\b(a\s+number\s+of\s+\w+s?)\s+is\b', "Use 'are' with 'a number of' since you're talking about multiple things"),
            (r'\b(the\s+number\s+of\s+\w+s?)\s+are\b', "Use 'is' with 'the number of' since you're talking about the number itself")
        ]
        
        for pattern, error_msg in patterns:
            if re.search(pattern, sentence, re.I):
                issues.append({
                    'type': 'Subject-Verb Agreement',
                    'severity': 'Major',
                    'context': sentence,
                    'correction': f"🤔 {error_msg}"
                })
        
        return issues

    def check_article_usage(self, sentence):
        """Check for proper article usage with helpful tips"""
        issues = []
        
        # Check for missing articles
        missing_article_patterns = [
            (r'\b(is|are|am) doing ([A-Z]+)\b', "💡 Consider adding 'a' or 'the' before the noun for clarity"),
            (r'\b(go|went) to ([A-Z][a-z]+)\b', "💡 Consider adding 'the' before the location name"),
            (r'\b(have|has) ([A-Z][a-z]+)\b', "💡 Consider adding 'a' or 'the' if you're referring to a specific item")
        ]
        
        for pattern, msg in missing_article_patterns:
            if re.search(pattern, sentence):
                issues.append({
                    'type': 'Article Usage',
                    'severity': 'Minor',
                    'context': sentence,
                    'correction': msg
                })
        
        return issues

    def check_redundancy(self, sentence):
        """Check for redundant phrases and expressions with friendly suggestions"""
        issues = []
        
        redundant_patterns = [
            (r'\b(past|previous) history\b', '💡 "History" already means the past, so you can just say "history"'),
            (r'\b(future) plans\b', '💡 "Plans" are always about the future, so you can just say "plans"'),
            (r'\b(basic) essentials\b', '💡 "Essentials" are already basic, so you can just say "essentials"'),
            (r'\b(personal) opinion\b', '💡 "Opinion" is always personal, so you can just say "opinion"'),
            (r'\badvance (planning|preparation)\b', '💡 "Planning" and "preparation" are already about the future, so you can drop "advance"'),
            (r'\b(unexpected) surprise\b', '💡 "Surprise" is always unexpected, so you can just say "surprise"'),
            (r'\b(repeat) again\b', '💡 "Repeat" already means "do again", so you can just say "repeat"'),
            (r'\b(new) innovation\b', '💡 "Innovation" is always new, so you can just say "innovation"'),
            (r'\b(very) unique\b', '💡 "Unique" means one-of-a-kind, so you can just say "unique"'),
            (r'\b(end) result\b', '💡 "Result" is always the end, so you can just say "result"')
        ]
        
        for pattern, suggestion in redundant_patterns:
            if re.search(pattern, sentence, re.I):
                issues.append({
                    'type': 'Redundancy',
                    'severity': 'Suggestion',
                    'context': sentence,
                    'correction': suggestion
                })
        
        return issues

    def calculate_score(self, issues, word_count):
        """Enhanced scoring system with encouraging feedback"""
        base_score = 100
        deductions = 0
        
        # Calculate weighted deductions with diminishing returns
        severity_counts = defaultdict(int)
        for issue in issues:
            severity_counts[issue['severity']] += 1
            
        for severity, count in severity_counts.items():
            deduction = self.issue_weights[severity] * (1 - (0.1 * (count - 1)))
            deductions += deduction * count
        
        # Length factor
        length_factor = min(1.0, word_count / 40)
        
        # Context penalties
        context_penalty = 0
        if any(i['type'] == 'Subject-Verb Agreement' for i in issues):
            context_penalty += 5
        if any(i['type'] == 'Tense Consistency' for i in issues):
            context_penalty += 5
        
        # Final score calculation
        final_score = max(40, base_score - (deductions * length_factor) - context_penalty)
        
        # Grade assignment with encouraging messages
        if final_score >= 85:
            grade = 'A'
            grade_message = "🌟 Excellent work! Your grammar is really strong!"
        elif final_score >= 75:
            grade = 'B'
            grade_message = "👍 Good job! You're doing well with just a few areas to polish."
        elif final_score >= 65:
            grade = 'C'
            grade_message = "📚 You're on the right track! A bit more practice and you'll be great."
        elif final_score >= 55:
            grade = 'D'
            grade_message = "💪 Keep practicing! Every mistake is a learning opportunity."
        else:
            grade = 'F'
            grade_message = "🌱 Don't worry! Everyone starts somewhere. Let's work on this together!"
        
        return final_score, grade, grade_message

    def generate_detailed_report(self, transcript, all_issues, final_score, grade):
        """Generate comprehensive analysis report with encouraging language"""
        report = {
            'summary': {
                'score': final_score,
                'grade': grade,
                'word_count': len(transcript.split())
            },
            'error_distribution': defaultdict(int),
            'suggestions': [],
            'strengths': [],
            'areas_for_improvement': []
        }
        
        # Analyze error distribution
        for issue in all_issues:
            report['error_distribution'][issue['type']] += 1
        
        # Generate specific suggestions with encouraging language
        if report['error_distribution'].get('Filler Words', 0) > 3:
            report['areas_for_improvement'].append("🎯 Try reducing filler words - your message will be more powerful!")
        if report['error_distribution'].get('Tense Consistency', 0) > 0:
            report['areas_for_improvement'].append("⏰ Pay attention to verb tenses - consistency helps your audience follow along")
        if report['error_distribution'].get('Subject-Verb Agreement', 0) > 0:
            report['areas_for_improvement'].append("🤝 Review subject-verb agreement - it's like making sure your words work together")
        
        # Identify strengths with celebration
        if final_score >= 85:
            report['strengths'].append("🌟 Your grammar skills are impressive!")
        if report['error_distribution'].get('Sentence Structure', 0) == 0:
            report['strengths'].append("📝 Great sentence structure - your ideas flow well!")
        if len(all_issues) == 0:
            report['strengths'].append("🎉 Perfect! No issues found - you're doing amazing!")
        
        return report

    def is_valid_past_participle(self, word):
        """Helper method to check if a word is a valid past participle"""
        # This is a simplified check - in a real implementation, you'd want a more comprehensive approach
        return word.endswith(('ed', 'en', 't')) or word in ['been', 'gone', 'seen', 'done', 'written', 'spoken']

# Main execution with friendly user experience
def main():
    print("\n" + "="*60)
    print("🎤 Welcome to Your Friendly Grammar Checker!")
    print("="*60)
    print("I'm here to help you improve your English with gentle, helpful feedback.")
    print("Let's analyze your speech together! 🚀\n")
    
    try:
        # Load audio and transcribe with progress indication
        audio_file = "imcorrectgrammar.wav"
        print(f"🎵 Listening to your audio file: {audio_file}")
        print("🔄 Transcribing your speech (this may take a moment)...")
        
        result = model.transcribe(audio_file)
        transcript = result["text"]
        
        print("✅ Transcription complete!")
        print(f"\n📝 Here's what I heard you say:")
        print(f"   \"{transcript}\"")
        print()

        # Initialize grammar checking with progress updates
        print("🔍 Analyzing your grammar and style...")
        gf = Gramformer(models=1)
        scorer = GrammarScorer()
        all_issues = []

        # Get Gramformer corrections
        try:
            print("🤖 Checking with advanced grammar models...")
            corrections = gf.correct(transcript)
            influences = gf.detect(transcript)
            
            if influences:
                for influence in influences:
                    all_issues.append({
                        'type': 'Grammar',
                        'severity': 'Critical',
                        'context': transcript[max(0, influence['start']-20):influence['end']+20],
                        'correction': corrections[0] if corrections else "No correction available"
                    })
        except Exception as e:
            print(f"⚠️  Note: Some advanced grammar checking features aren't available right now, but I'll still help you with the basics!")

        # Process each sentence with progress indication
        sentences = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        print(f"📊 Analyzing {len(sentences)} sentence(s)...")
        
        for i, sentence in enumerate(sentences, 1):
            try:
                # Run all checks
                all_issues.extend(scorer.analyze_sentence_structure(sentence))
                all_issues.extend(scorer.check_tense_consistency(sentence))
                all_issues.extend(scorer.check_subject_verb_agreement(sentence))
                all_issues.extend(scorer.check_article_usage(sentence))
            except Exception as e:
                print(f"⚠️  Had trouble analyzing one sentence, but continuing with the rest...")

        # Group issues by severity
        severity_groups = defaultdict(list)
        for issue in all_issues:
            severity_groups[issue['severity']].append(issue)

        # Display issues with friendly formatting
        if all_issues:
            print("\n" + "="*60)
            print("📋 Your Personalized Grammar Report")
            print("="*60)
            
            for severity in ['Critical', 'Major', 'Minor', 'Suggestion']:
                issues = severity_groups[severity]
                if issues:
                    severity_icons = {
                        'Critical': '🚨',
                        'Major': '⚠️',
                        'Minor': '💡',
                        'Suggestion': '💭'
                    }
                    print(f"\n{severity_icons[severity]} {severity} Areas to Consider:")
                    print("-" * (len(severity) + 25))
                    for i, issue in enumerate(issues, 1):
                        print(f"\n{i}. {issue['type']}")
                        print(f"   Context: \"...{issue['context']}...\"")
                        print(f"   💬 {issue['correction']}")
        else:
            print("\n🎉 Fantastic news! I didn't find any issues to address.")
            print("Your grammar looks great! 🌟")

        # Calculate final score with encouraging feedback
        word_count = len(transcript.split())
        final_score, grade, grade_message = scorer.calculate_score(all_issues, word_count)

        # Generate and display report
        report = scorer.generate_detailed_report(transcript, all_issues, final_score, grade)

        print("\n" + "="*60)
        print("📊 Your Grammar Assessment Summary")
        print("="*60)
        print(f"📝 Words analyzed: {word_count}")
        print(f"🔍 Total suggestions: {len(all_issues)}")
        print(f"📈 Your score: {final_score:.1f}/100")
        print(f"🏆 Grade: {grade}")
        print(f"\n{grade_message}")

        if report['strengths']:
            print("\n🌟 What you're doing well:")
            for strength in report['strengths']:
                print(f"   • {strength}")

        if report['areas_for_improvement']:
            print("\n🎯 Areas to focus on:")
            for area in report['areas_for_improvement']:
                print(f"   • {area}")

        print("\n" + "="*60)
        print("💡 Remember: Every great writer started somewhere!")
        print("Keep practicing, and you'll keep improving. You've got this! 💪")
        print("="*60)

    except FileNotFoundError:
        print("❌ Oops! I couldn't find the audio file 'imcorrectgrammar.wav'")
        print("💡 Make sure the file is in the same folder as this program.")
    except Exception as e:
        print(f"❌ Something unexpected happened: {str(e)}")
        print("💡 Don't worry - this sometimes happens. Try running the program again!")

# Flask web application
app = Flask(__name__)

# HTML template for the main page
MAIN_PAGE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grammar Checker - Your Friendly Writing Assistant</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }
        
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #6c757d;
            margin-bottom: 30px;
            font-size: 1.1em;
        }
        
        .upload-section {
            background: #f8f9fa;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 2px dashed #dee2e6;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            color: #495057;
            font-weight: 500;
        }
        
        input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
        }
        
        textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #dee2e6;
            border-radius: 8px;
            font-size: 1em;
            font-family: inherit;
            resize: vertical;
            min-height: 120px;
        }
        
        .button-group {
            display: flex;
            gap: 10px;
            margin-top: 20px;
        }
        
        button {
            flex: 1;
            padding: 15px 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-size: 1.1em;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        button:active {
            transform: translateY(0);
        }
        
        button:disabled {
            background: #6c757d;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
            color: #6c757d;
        }
        
        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #dc3545;
        }
        
        .info {
            background: #d1ecf1;
            color: #0c5460;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            border-left: 4px solid #17a2b8;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎤 Grammar Checker</h1>
        <p class="subtitle">Your Friendly Writing Assistant</p>
        
        <div class="upload-section">
            <form id="grammarForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label for="audioFile">📁 Upload Audio File (WAV format):</label>
                    <input type="file" id="audioFile" name="audioFile" accept=".wav,.mp3,.m4a" />
                </div>
                
                <div style="text-align: center; margin: 20px 0; color: #6c757d;">
                    <strong>OR</strong>
                </div>
                
                <div class="form-group">
                    <label for="transcript">📝 Enter Transcript Text:</label>
                    <textarea id="transcript" name="transcript" placeholder="Paste or type your transcript here..."></textarea>
                </div>
                
                <div class="button-group">
                    <button type="submit" id="submitBtn">🔍 Analyze Grammar</button>
                </div>
            </form>
            
            <div id="loading" class="loading" style="display: none;">
                <p>🔄 Processing your request... This may take a moment.</p>
            </div>
            
            <div id="error" class="error" style="display: none;"></div>
            <div id="info" class="info" style="display: none;"></div>
        </div>
        
        <div id="result" style="display: none;"></div>
    </div>
    
    <script>
        document.getElementById('grammarForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const loading = document.getElementById('loading');
            const error = document.getElementById('error');
            const info = document.getElementById('info');
            const result = document.getElementById('result');
            
            // Reset UI
            error.style.display = 'none';
            info.style.display = 'none';
            result.style.display = 'none';
            loading.style.display = 'block';
            submitBtn.disabled = true;
            
            const formData = new FormData();
            const audioFile = document.getElementById('audioFile').files[0];
            const transcript = document.getElementById('transcript').value;
            
            if (audioFile) {
                formData.append('audioFile', audioFile);
            } else if (transcript.trim()) {
                formData.append('transcript', transcript);
            } else {
                error.textContent = 'Please either upload an audio file or enter a transcript.';
                error.style.display = 'block';
                loading.style.display = 'none';
                submitBtn.disabled = false;
                return;
            }
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Server error: ' + response.statusText);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                if (data.html) {
                    // Extract body content from the HTML report
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(data.html, 'text/html');
                    const bodyContent = doc.body.innerHTML;
                    
                    result.innerHTML = bodyContent;
                    result.style.display = 'block';
                    result.scrollIntoView({ behavior: 'smooth' });
                }
                
            } catch (err) {
                error.textContent = 'Error: ' + err.message;
                error.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                submitBtn.disabled = false;
            }
        });
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """Main page with upload form"""
    return render_template_string(MAIN_PAGE_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze grammar from audio file or transcript"""
    try:
        transcript = None
        
        # Check if audio file was uploaded
        if 'audioFile' in request.files:
            audio_file = request.files['audioFile']
            if audio_file.filename:
                # Save uploaded file temporarily
                temp_path = f"temp_{audio_file.filename}"
                audio_file.save(temp_path)
                
                try:
                    # Transcribe audio
                    print(f"🔄 Transcribing audio file: {audio_file.filename}")
                    result = model.transcribe(temp_path)
                    transcript = result["text"]
                    print(f"✅ Transcription complete: {transcript}")
                finally:
                    # Clean up temporary file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        # Check if transcript was provided directly
        if not transcript and 'transcript' in request.form:
            transcript = request.form['transcript']
        
        if not transcript or not transcript.strip():
            return jsonify({'error': 'No transcript available. Please provide an audio file or transcript text.'}), 400
        
        # Get HTML report from research adapter
        print(f"🔍 Analyzing grammar for transcript: {transcript[:50]}...")
        html_report = score_grammar_from_transcript(transcript.strip())
        print("✅ Analysis complete")
        
        return jsonify({'html': html_report, 'transcript': transcript})
    
    except Exception as e:
        print(f"❌ Error in analyze route: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    import sys
    # Check if running as web app or CLI
    if len(sys.argv) > 1 and sys.argv[1] == '--web':
        print("🌐 Starting web server...")
        print("📱 Open http://localhost:5000 in your browser")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Run CLI version
        main()
