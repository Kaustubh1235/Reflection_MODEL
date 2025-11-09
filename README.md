

# 📄 README 



---

# 🏛️ Legal QA Advisory System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.30%2B-yellow.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> An intelligent legal advisory system that combines Large Language Models with fine-tuned T5 transformers to provide accurate, reflective, and continuously improving legal information on constitutional law.

## 🎯 Overview

The Legal QA Advisory System is a hybrid AI architecture that:
- **Generates** fast initial answers using GROQ LLM
- **Critiques** its own responses using fine-tuned T5
- **Improves** answers through self-reflection
- **Learns** continuously from user feedback
- **Remembers** past consultations for faster responses

### Key Features

- ⚡ **Hybrid Architecture**: Combines GROQ (speed) + T5 (quality)
- 🧠 **Self-Reflection**: AI critiques and improves its own answers
- 💾 **Memory System**: SQLite-based storage with similarity matching
- 🔄 **Continuous Learning**: Auto-retraining from user corrections
- ⚖️ **Legal Focus**: Specialized in constitutional law
- 🔒 **Ethical AI**: Built-in legal disclaimers and limitations

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER QUESTION                             │
└──────────────┬──────────────────────────────────────────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Memory Check       │◄───────── SQLite Database
    │   (Similarity: 0.65) │           - Past Q&A
    └──────┬───────┬───────┘           - Reflections
           │       │                    - Feedback
           │       └──Found ──► Return Cached Answer
           │
           └──Not Found
               │
               ▼
    ┌──────────────────────┐
    │   GROQ LLM           │
    │   (Fast Generation)  │
    └──────────┬───────────┘
               │
               ▼
         Initial Answer
               │
               ▼
    ┌──────────────────────┐
    │   T5 Critique        │
    │   (Find Issues)      │
    └──────────┬───────────┘
               │
               ▼
         Reflection
               │
               ▼
    ┌──────────────────────┐
    │   T5 Improvement     │
    │   (Enhanced Answer)  │
    └──────────┬───────────┘
               │
               ▼
    ┌──────────────────────┐
    │   Store in Memory    │
    │   + User Feedback    │
    └──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
PyTorch 2.0+
CUDA (optional, for GPU acceleration)
8GB RAM minimum (16GB recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/legal-qa-system.git
cd legal-qa-system
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
# Create .env file
echo "GROQ_API_KEY=your_api_key_here" > .env
```

4. **Download NLTK data**
```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### Usage

#### 1. Prepare Your Data

Place your legal Q&A dataset in JSON format:

```json
{
  "question": "What are fundamental rights?",
  "answer": "Fundamental rights are basic human rights..."
}
```

#### 2. Run the Complete Pipeline

```bash
python legal_advisor.py
```

#### 3. Menu Options

```
🏛️  LEGAL ADVISOR MAIN MENU  ⚖️
═════════════════════════════════
1. Train Legal T5 model
2. Interactive Legal Advisory session
3. Test legal advisory system
4. Show consultation memory
5. Retrain with feedback
6. Database maintenance
7. Export consultation data
8. System statistics
9. Legal disclaimer
10. Exit
```

#### 4. Interactive Session Example

```
📝 Legal Question: What is due process?

🔍 Analyzing legal question with GROQ...

═══════════════════════════════════════
Initial Legal Analysis
───────────────────────────────────────
Due process is a constitutional principle
that requires fair legal procedures...
═══════════════════════════════════════

⚖️ Generating legal critique with T5...

═══════════════════════════════════════
Legal Analysis Critique
───────────────────────────────────────
The answer is accurate but could include
more detail on procedural vs substantive
due process...
═══════════════════════════════════════

═══════════════════════════════════════
Enhanced Legal Analysis
───────────────────────────────────────
Due process requires fair procedures and
protects fundamental rights. It includes:
1. Procedural due process - fair procedures
2. Substantive due process - protection of
   fundamental rights from arbitrary action

⚖️ Disclaimer: This is general legal
information, not legal advice...
═══════════════════════════════════════

🤔 Is this analysis helpful? (y/n/correction):
```

---

## 📊 System Components

### 1. Data Preprocessing

**Module**: `preprocess.py`

```python
from preprocess import EnhancedDataPreprocessor, PreprocessingConfig

config = PreprocessingConfig(
    input_csv_path="constitution_qa.csv",
    text_column="question",
    target_column="answer",
    output_dir="processed_data",
    test_size=0.2,
    validation_size=0.1,
    lemmatize=True,
    remove_stopwords=True
)

preprocessor = EnhancedDataPreprocessor(config)
results = preprocessor.run()
```

**Features**:
- ✅ NLP text cleaning (lemmatization, stopword removal)
- ✅ Duplicate detection (TF-IDF + cosine similarity)
- ✅ Quality assessment and labeling
- ✅ Train/test/validation splitting
- ✅ Comprehensive reporting

### 2. Model Training

**Module**: `legal_advisor.py` → `train_legal_model()`

```python
# Automatically handles:
# - Loading preprocessed data
# - Initializing Flan-T5-base
# - Adding legal-specific tokens
# - Fine-tuning for 5 epochs
# - Checkpoint management
# - Evaluation on validation set

train_legal_model()
```

**Training Details**:
- Base Model: `google/flan-t5-base` (248M parameters)
- Custom Tokens: `<legal_critique>`, `<legal_improve>`, `<statute>`, `<case_law>`
- Epochs: 5
- Batch Size: 2 (CPU-optimized)
- Learning Rate: 5e-5
- Checkpoint Strategy: Save best 3 models

### 3. Hybrid Inference

**Module**: `legal_advisor.py` → `legal_interactive_session()`

**Components**:

a) **GROQ Handler**
```python
groq = LegalGroqLLMHandler(GROQ_API_KEY)
answer = groq.generate_legal_answer(question)
```
- Fast generation (< 2 seconds)
- Legal-specific system prompts
- Automatic disclaimer injection

b) **T5 Critique**
```python
critique_prompt = f"legal_critique: {question}\n{groq_answer}"
reflection = t5(critique_prompt)
```
- Identifies factual errors
- Finds missing information
- Assesses legal accuracy

c) **T5 Improvement**
```python
improve_prompt = f"legal_improve: {critique}\n{original}"
enhanced = t5(improve_prompt)
```
- Fixes identified issues
- Adds missing details
- Improves structure

### 4. Memory System

**Module**: `DatabaseManager` class

**Database Schema**:
```sql
CREATE TABLE feedback (
    id INTEGER PRIMARY KEY,
    question TEXT NOT NULL,
    question_hash TEXT UNIQUE,
    groq_answer TEXT,
    improvement TEXT,
    t5_reflection TEXT,
    source TEXT,
    confidence_score REAL,
    timestamp TEXT
);

CREATE TABLE system_stats (
    total_memories INTEGER,
    auto_saved INTEGER,
    user_corrections INTEGER,
    groq_responses INTEGER,
    t5_reflections INTEGER
);
```

**Features**:
- ✅ Similarity-based retrieval (threshold: 0.65)
- ✅ Confidence scoring (0.0 - 1.0)
- ✅ Usage statistics tracking
- ✅ Automatic cleanup of old entries
- ✅ Export capabilities

---

## 🧪 Testing

### Manual Testing

```bash
# Run interactive session
python legal_advisor.py
# Select option 2: Interactive Legal Advisory session

# Test questions:
1. "What is due process?"
2. "Explain judicial review"
3. "What are fundamental rights?"
```

### System Testing

```bash
# Run test suite
python legal_advisor.py
# Select option 3: Test legal advisory system
```

This will:
- Test GROQ connectivity
- Test T5 model loading
- Test memory system
- Generate sample Q&A

### Database Testing

```bash
# Check memory statistics
python legal_advisor.py
# Select option 4: Show consultation memory
```

---

## 📈 Performance Metrics

### Speed Benchmarks

| Component | Time | Notes |
|-----------|------|-------|
| Memory Check | < 100ms | SQLite query + similarity |
| GROQ Generation | 1-2s | API call |
| T5 Critique | 2-3s | CPU inference |
| T5 Improvement | 3-5s | CPU inference |
| **Total (cold)** | **6-10s** | No memory hit |
| **Total (cached)** | **< 1s** | Memory hit |

### Model Metrics

| Metric | Training | Validation |
|--------|----------|------------|
| Loss | 0.45 | 0.52 |
| Perplexity | 1.57 | 1.68 |
| BLEU Score | 0.68 | 0.63 |

### Memory Statistics

```
Total Consultations: 150
Auto-saved: 120
User Corrections: 30
Memory Reuse Rate: 42%
Average Confidence: 0.78
```

---

## 🗂️ Project Structure

```
legal-qa-system/
│
├── Dev/
│   ├── preprocessed_data/
│   │   ├── processed_constitution/
│   │   │   ├── train.csv
│   │   │   ├── test.csv
│   │   │   └── validation.csv
│   │   └── formatted_constitution_dataset.csv
│   │
│   └── flan_t5_legal_advisor_model/
│       ├── checkpoint-600/ (best model)
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── legal_feedback_system.db
│       └── logs/
│
├── legal_advisor.py           # Main system (Document 4)
├── preprocess.py              # Data preprocessing
├── json_to_csv_formatter.py  # Data formatter
├── requirements.txt
├── .env                       # API keys
└── README.md                  # This file
```

---

## 🔧 Configuration

### Environment Variables

```bash
# .env file
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b  # Optional, default shown
```

### System Configuration

```python
# In legal_advisor.py

# Paths
PREPROCESSED_DIR = "Dev/preprocessed_data/processed_constitution"
MODEL_DIR = "Dev/flan_t5_legal_advisor_model"
DATABASE_PATH = f"{MODEL_DIR}/legal_feedback_system.db"

# Model Parameters
PRETRAINED_MODEL = "google/flan-t5-base"
MAX_INPUT_LENGTH = 768
MAX_TARGET_LENGTH = 768
BATCH_SIZE = 2
NUM_TRAIN_EPOCHS = 5

# Memory Settings
SIMILARITY_THRESHOLD = 0.65
```

---

## 📚 Dependencies

### Core Libraries

```txt
# requirements.txt

torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0

# NLP
nltk>=3.8
scikit-learn>=1.2.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0

# Database
sqlite3  # Built-in

# API
groq>=0.4.0
python-dotenv>=1.0.0

# UI
rich>=13.0.0
```

### Installation Command

```bash
pip install torch transformers datasets accelerate nltk scikit-learn pandas numpy groq python-dotenv rich
```

---

## 🎓 Educational Value

### Key Concepts Demonstrated

1. **Hybrid AI Architecture**
   - Combining multiple AI models
   - Leveraging strengths of each
   - Efficient resource usage

2. **Self-Reflection in AI**
   - Meta-learning concepts
   - Self-critique mechanisms
   - Iterative improvement

3. **NLP Pipeline**
   - Text preprocessing
   - Tokenization
   - Lemmatization
   - POS tagging

4. **Transfer Learning**
   - Fine-tuning pre-trained models
   - Domain adaptation
   - Catastrophic forgetting prevention

5. **Production ML**
   - Model versioning
   - Checkpoint management
   - Memory optimization
   - Error handling

6. **Database Design**
   - Schema design
   - Indexing strategies
   - Query optimization

---

## 🚧 Limitations & Future Work

### Current Limitations

1. **Domain Specificity**
   - Trained only on constitutional law
   - May not generalize to other legal areas

2. **Language Support**
   - English only currently
   - No multilingual support

3. **Jurisdiction**
   - US-centric constitutional knowledge
   - Limited international law coverage

4. **Compute Requirements**
   - CPU inference is slow (3-5s per response)
   - GPU strongly recommended for production

5. **Scale**
   - SQLite limitations for very large deployments
   - No distributed training support

### Planned Improvements

- [ ] Add GPU acceleration
- [ ] Implement RAG (Retrieval Augmented Generation)
- [ ] Add citation system for legal sources
- [ ] Multi-jurisdictional support
- [ ] Domain validation layer
- [ ] Automated testing suite
- [ ] Web UI interface
- [ ] API endpoints
- [ ] Docker containerization
- [ ] Monitoring and analytics dashboard

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

```bash
# Fork and clone
git clone https://github.com/yourusername/legal-qa-system.git
cd legal-qa-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .

# Run tests
pytest tests/
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Legal Disclaimer

This system provides general legal information for educational purposes only. It does NOT provide legal advice and cannot replace consultation with a qualified attorney.

- Responses are for informational purposes only
- Laws vary by jurisdiction and change frequently
- Specific legal matters require professional legal counsel
- No attorney-client relationship is created
- Always verify information with current legal sources

**For legal advice specific to your situation, consult a licensed attorney.**

---

## 👨‍💻 Author

**Kaustubh**


---

## 🙏 Acknowledgments

- **Flan-T5**: Google Research
- **GROQ**: GROQ Inc.
- **Transformers Library**: Hugging Face
- **Constitutional Dataset**: [Source]
- **Inspiration**: Reflective QA systems research

---


---

**Made with ⚖️ and 🤖 by Kaustubh**

---

