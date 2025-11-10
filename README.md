# OralGrammatica — AI-Powered Voice Grammar Analysis

Transform spoken English with gentle, intelligent grammar guidance. Built on local Whisper ASR and classic NLP to provide encouraging, personalized feedback for language learners, presenters, and public speakers.

> ✨ Real-time-ready transcription • Comprehensive analysis • Humanized feedback • Personalized scoring

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Sample Output](#sample-output)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Features

- **Speech-to-Text (ASR):** Transcribes WAV audio using OpenAI’s Whisper (local inference).
- **AI-assisted Grammar Analysis:** Checks tense consistency, subject–verb agreement, sentence structure, article use, redundancy, and rhythm.
- **Humanized Feedback:** Supportive, non-judgmental phrasing and gentle suggestions.
- **Personalized Scoring:** Weighted rubric rolled up into a grade/score with strengths and next steps.
- **Drop-in WAV Support:** Works with your own audio files out of the box.

---

## Architecture

```
Audio (.wav)
│
├─► Whisper ASR (local) → transcription
│
├─► NLP / Heuristics / Grammar models
│      • tense consistency
│      • S/V agreement
│      • structure & articles
│      • redundancy & rhythm
│
└─► Friendly report + weighted scoring
```

**Notes**
- Whisper runs locally via the `whisper` python package.
- Classic NLP utilities (e.g., `nltk`, regex) and a grammar-correction model (`gramformer`) complement rule-based checks.

---

## Repository Structure

```
.
├── app.py                   # Entry point (CLI)
├── requirements.txt         # Python dependencies
├── README.md                # You are here
├── correct_grammar.wav      # Sample audio (good grammar)
├── incorrect_grammar.wav    # Sample audio (issues to detect)
└── research_adapter/        # (WIP) notes / experiments

```

---

## Getting Started

### Prerequisites
- **Python**: 3.8+ recommended
- **OS**: Linux/Mac/Windows
- **Audio**: WAV files (mono/stereo; common sample rates work)
- **Optional**: GPU with CUDA for faster Whisper inference

### 1) Clone

```bash
git clone https://github.com/Dhurv-G/OralGrammatica.git
cd OralGrammatica
````

### 2) Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix/Mac:
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Quick run with the included sample

```bash
python app.py
```

By default, `app.py` is set up to read a WAV file from the project directory (e.g., `incorrect_grammar.wav`).
To analyze your own file, either **replace the filename** in the code (see [Configuration](#configuration)) or rename your file to match the expected default.

### Analyze a specific file (simple edit)

Open `app.py` and set:

```python
audio_file = "your_audio_file.wav"  # point this to your file
```

Then run:

```bash
python app.py
```

---

## Sample Output

```
Welcome to Your Friendly Grammar Checker!
============================================================
I'm here to help you improve your English with gentle, helpful feedback.

Listening to your audio file: incorrect_grammar.wav
Transcribing your speech...
✅ Transcription complete!

Here's what I heard you say:
  "I am going to the store yesterday"
Analyzing your grammar and style...
Analyzing 1 sentence(s)...

============================================================
Your Personalized Grammar Report
============================================================

⚠️ Major Areas to Consider:
---------------------------
1. Tense Consistency
   Context: "...I am going to the store yesterday..."
   I noticed you're mixing present and past tenses here. Usually, it's clearer to
   stick with one tense unless you're describing different time periods.

============================================================
Your Grammar Assessment Summary
============================================================
Words analyzed: 7
Total suggestions: 1
Your score: 85.2/100
Grade: A

Excellent work! Your grammar is really strong!

What you're doing well:
  • Your grammar skills are impressive!

Areas to focus on:
  • ⏰ Pay attention to verb tenses - consistency helps your audience follow along
============================================================
Remember: Every great writer started somewhere!
Keep practicing, and you'll keep improving. You've got this!
============================================================
```

---

## Configuration

Open `app.py` and look for the top-level settings/constants:

* **Audio file path**

  ```python
  audio_file = "your_audio_file.wav"
  ```

* **Whisper model size**
  Whisper models trade accuracy for speed (`tiny`, `base`, `small`, `medium`, `large`).
  If inference is slow on your machine, consider a smaller English-only model such as `small.en` instead of `base.en`.

* **Scoring weights**
  Adjust the relative importance of issue categories:

  ```python
  self.issue_weights = {
      "Critical": 8,  # Most important issues
      "Major": 6,     # Important issues
      "Minor": 2,     # Minor improvements
      "Suggestion": 1 # Helpful tips
  }
  ```

> Tip: The first run may download model assets; subsequent runs are faster.

---

## Troubleshooting

1. **“Couldn’t find the audio file”**

   * Ensure the WAV sits in the project root next to `app.py`.
   * Confirm the filename (case-sensitive) matches `audio_file`.

2. **“Some advanced grammar features aren’t available”**

   * This can happen if large language resources aren’t fully loaded.
   * Core checks still run; try again or verify model installs.

3. **Slow performance**

   * Use a smaller Whisper model (e.g., `small.en`).
   * Prefer a GPU-enabled environment if available.

4. **Dependency conflicts**

   * Upgrade `pip` and reinstall:

     ```bash
     pip install --upgrade pip
     pip install -r requirements.txt --upgrade --force-reinstall
     ```

---

## Roadmap

* [ ] **Mic / Real-time mode** (VAD + streaming ASR)
* [ ] **Simple web UI** (Streamlit/Gradio)
* [ ] **Docker image** for reproducible runs
* [ ] **Config file** (YAML/ENV) for model/weights/filepaths
* [ ] **Unit tests** and CI checks
* [ ] **Evaluation harness** on public ESL datasets
* [ ] **Export** reports to JSON/Markdown

---

## Contributing

Contributions are welcome!

1. Fork the repo
2. Create a feature branch
3. Make your changes + add tests (where applicable)
4. Open a PR describing the change and rationale

---

## Acknowledgments

* **Whisper** (OpenAI) — local ASR via the `whisper` python package
* **Hugging Face Transformers** — model utilities
* **PyTorch**, **NLTK**, **Gramformer** — grammar/NLP components

---

## 📝 License

This project is open source. Feel free to use and modify for your needs.

---
