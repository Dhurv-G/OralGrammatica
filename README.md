# 🎤 Friendly Grammar Checker

A humanized, AI-powered grammar checker that analyzes your spoken English with gentle, encouraging feedback. Perfect for language learners, public speakers, and anyone looking to improve their English communication skills!

## ✨ Features

- **🎵 Speech-to-Text Analysis**: Transcribes your audio files and analyzes the grammar
- **🤖 AI-Powered Grammar Checking**: Uses advanced language models for comprehensive analysis
- **💬 Friendly Feedback**: Encouraging, supportive language instead of harsh corrections
- **📊 Detailed Scoring**: Personalized grade with specific areas for improvement
- **🎯 Multiple Analysis Types**:
  - Subject-verb agreement
  - Tense consistency
  - Sentence structure
  - Article usage
  - Redundancy detection
  - Speech patterns and rhythm

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- An audio file in WAV format (e.g., `incorrectgrammar.wav`)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone https://github.com/Dhurv-G/OralGrammatica.git
   cd GrammarChecker
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your audio file**
   - Put your WAV audio file in the same directory as `app.py`
   - Name it `incorrectgrammar.wav` or update the filename in the code

4. **Run the grammar checker**
   ```bash
   python app.py
   ```

## 📋 Requirements

The main dependencies include:

- **whisper**: OpenAI's speech recognition model
- **transformers**: Hugging Face's language models
- **torch**: PyTorch for deep learning
- **nltk**: Natural Language Toolkit
- **gramformer**: Advanced grammar correction
- **re**: Regular expressions for text processing

See `requirements.txt` for the complete list with specific versions.

## 🎯 How It Works

1. **Audio Processing**: The app transcribes your WAV file using OpenAI's Whisper model
2. **Grammar Analysis**: Multiple AI models analyze your text for various grammar issues
3. **Friendly Feedback**: Issues are categorized and presented with encouraging suggestions
4. **Scoring**: You receive a personalized grade (A-F) with specific improvement areas
5. **Detailed Report**: Comprehensive analysis with strengths and areas for growth

## 📊 Sample Output

```
🎤 Welcome to Your Friendly Grammar Checker!
============================================================
I'm here to help you improve your English with gentle, helpful feedback.
Let's analyze your speech together! 🚀

🎵 Listening to your audio file: incorrectgrammar.wav
🔄 Transcribing your speech (this may take a moment)...
✅ Transcription complete!

📝 Here's what I heard you say:
   "I am going to the store yesterday"

🔍 Analyzing your grammar and style...
📊 Analyzing 1 sentence(s)...

============================================================
📋 Your Personalized Grammar Report
============================================================

⚠️ Major Areas to Consider:
-------------------------

1. Tense Consistency
   Context: "...I am going to the store yesterday..."
   💬 🤔 I noticed you're mixing present and past tenses here. Usually, it's clearer to stick with one tense unless you're describing different time periods. What do you think?

============================================================
📊 Your Grammar Assessment Summary
============================================================
📝 Words analyzed: 7
🔍 Total suggestions: 1
📈 Your score: 85.2/100
🏆 Grade: A

🌟 Excellent work! Your grammar is really strong!

🌟 What you're doing well:
   • 🌟 Your grammar skills are impressive!

🎯 Areas to focus on:
   • ⏰ Pay attention to verb tenses - consistency helps your audience follow along

============================================================
💡 Remember: Every great writer started somewhere!
Keep practicing, and you'll keep improving. You've got this! 💪
============================================================
```

## 🔧 Customization

### Changing the Audio File

Edit line 368 in `app.py`:
```python
audio_file = "your_audio_file.wav"  # Change this to your file name
```

### Adjusting Analysis Sensitivity

Modify the scoring weights in the `GrammarScorer` class:
```python
self.issue_weights = {
    'Critical': 8,    # Most important issues
    'Major': 6,       # Important issues  
    'Minor': 2,       # Minor improvements
    'Suggestion': 1   # Helpful tips
}
```

## 🐛 Troubleshooting

### Common Issues

1. **"Couldn't find the audio file"**
   - Make sure your WAV file is in the same directory as `app.py`
   - Check the filename matches exactly (case-sensitive)

2. **"Some advanced grammar checking features aren't available"**
   - This is normal if Gramformer models aren't fully loaded
   - The basic grammar checking will still work

3. **Slow performance**
   - First run downloads AI models (this is normal)
   - Subsequent runs will be faster
   - Consider using "small.en" instead of "base.en" for faster processing

### System Requirements

- **RAM**: At least 4GB recommended
- **Storage**: ~2GB for AI models (downloaded automatically)
- **Audio**: WAV format, any sample rate

## 🤝 Contributing

Feel free to contribute improvements:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📝 License

This project is open source. Feel free to use and modify for your needs.

---