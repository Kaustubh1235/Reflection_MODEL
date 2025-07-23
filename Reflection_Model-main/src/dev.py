
import os
import json
import glob
import pandas as pd
import torch
import torch, time, re
from transformers import pipeline
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    pipeline,
)
import difflib
from typing import Dict, List, Optional, Tuple
import hashlib
from groq import Groq
import time
from dotenv import load_dotenv
load_dotenv()


PREPROCESSED_DIR   = "./preprocessed_data"
CLEAN_CSV_PATH     = os.path.join(PREPROCESSED_DIR, "cleaned_qa_data.csv")
TRAIN_SPLIT_PATH   = os.path.join(PREPROCESSED_DIR, "train_split.csv")
TEST_SPLIT_PATH    = os.path.join(PREPROCESSED_DIR, "test_split.csv")

MODEL_DIR          = "./flan_t5_reflective_model"
FEEDBACK_JSON      = os.path.join(MODEL_DIR, "feedback.json")
MEMORY_STATS_JSON  = os.path.join(MODEL_DIR, "memory_stats.json")
os.makedirs(MODEL_DIR, exist_ok=True)

PRETRAINED_MODEL   = "google/flan-t5-base"
MAX_INPUT_LENGTH   = 512
MAX_TARGET_LENGTH  = 512
BATCH_SIZE         = 2
NUM_TRAIN_EPOCHS   = 3
RETRAIN_EPOCHS     = 2
REFLECTION_RETRY_LIMIT = 2
SIMILARITY_THRESHOLD = 0.6  

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama3-8b-8192"  
GROQ_MAX_TOKENS = 512
DEVICE = "cpu"

class GroqLLMHandler:
    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        if not api_key:
            raise ValueError("GROQ API key not found. Please set GROQ_API_KEY environment variable.")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        
    def generate_answer(self, question: str, max_tokens: int = GROQ_MAX_TOKENS) -> str:
        """Generate initial answer using GROQ LLM"""
        try:
            system_prompt = """You are a helpful and knowledgeable assistant. Provide accurate, concise, and informative answers to questions. Focus on being helpful while being direct and to the point."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Question: {question}"}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  GROQ API Error: {e}")
            return f"Sorry, I couldn't generate an answer due to an API error: {str(e)}"
    
    def check_connection(self) -> bool:
        """Test GROQ API connection"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            print(f" GROQ Connection Error: {e}")
            return False

class EnhancedMemorySystem:
    def __init__(self, feedback_json_path: str, stats_json_path: str, similarity_threshold: float = 0.7):
        self.feedback_path = feedback_json_path
        self.stats_path = stats_json_path
        self.similarity_threshold = similarity_threshold
        self.stats = self.load_stats()
    
    def load_feedback(self) -> List[Dict]:
        """Load feedback from JSON file"""
        if os.path.exists(self.feedback_path):
            return json.load(open(self.feedback_path, "r", encoding="utf-8"))
        return []
    
    def save_feedback(self, feedback_list: List[Dict]):
        """Save feedback to JSON file"""
        json.dump(feedback_list, open(self.feedback_path, "w", encoding="utf-8"),
                  indent=2, ensure_ascii=False)
        self.update_stats(len(feedback_list))
    
    def load_stats(self) -> Dict:
        """Load memory statistics"""
        if os.path.exists(self.stats_path):
            return json.load(open(self.stats_path, "r", encoding="utf-8"))
        return {
            "total_memories": 0,
            "auto_saved": 0,
            "user_corrections": 0,
            "reuse_count": 0,
            "groq_responses": 0,
            "t5_reflections": 0,
            "last_updated": pd.Timestamp.now().isoformat()
        }
    
    def update_stats(self, total_memories: int):
        """Update memory statistics"""
        self.stats["total_memories"] = total_memories
        self.stats["last_updated"] = pd.Timestamp.now().isoformat()
        json.dump(self.stats, open(self.stats_path, "w", encoding="utf-8"), indent=2)
    
    def calculate_similarity(self, question1: str, question2: str) -> float:
        """Calculate similarity between two questions using multiple methods"""
        q1_clean = question1.strip().lower()
        q2_clean = question2.strip().lower()
        
        if q1_clean == q2_clean:
            return 1.0
        
        if q1_clean in q2_clean or q2_clean in q1_clean:
            return 0.9
        
        sequence_similarity = difflib.SequenceMatcher(None, q1_clean, q2_clean).ratio()
        
        words1 = set(q1_clean.split())
        words2 = set(q2_clean.split())
        if len(words1) == 0 or len(words2) == 0:
            word_similarity = 0.0
        else:
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            word_similarity = intersection / union
        
        combined_similarity = (sequence_similarity * 0.6) + (word_similarity * 0.4)
        return combined_similarity
    
    def find_similar_question(self, question: str, feedback_list: List[Dict]) -> Optional[Dict]:
        """Find the most similar question in feedback with similarity above threshold"""
        if not feedback_list:
            return None
        
        best_match = None
        best_similarity = 0.0
        
        for feedback in feedback_list:
            similarity = self.calculate_similarity(question, feedback["question"])
            if similarity > best_similarity and similarity >= self.similarity_threshold:
                best_similarity = similarity
                best_match = feedback
        
        if best_match:
            self.stats["reuse_count"] += 1
            self.update_stats(self.stats["total_memories"])
            print(f" Found similar question (similarity: {best_similarity:.2f})")
            
        return best_match
    
    def auto_save_reflection(self, question, groq_answer, improved_answer, reflection):
        entry = {
        "question": question,
        "groq_answer": groq_answer,
        "improvement": improved_answer,
        "t5_reflection": reflection,
        "source": "auto-reflection",
        "confidence_score": self.estimate_confidence(reflection),
        "improvement_type": "groq-t5-hybrid",
        "timestamp": pd.Timestamp.now().isoformat(),
        "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
        }
        self.stats["auto_saved"] += 1
        return entry

    
    def save_user_correction(self, question: str, groq_answer: str, 
                           user_correction: str, reflection: str) -> Dict:
        """Save user-provided correction"""
        feedback_entry = {
            "question": question,
            "groq_answer": groq_answer,
            "improvement": user_correction,
            "t5_reflection": reflection,
            "source": "user-correction",
            "confidence_score": 1.0,  
            "improvement_type": "user-feedback",
            "timestamp": pd.Timestamp.now().isoformat(),
            "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
        }
        
        self.stats["user_corrections"] += 1
        print(" Saved user correction to memory")
        return feedback_entry
    
    def estimate_confidence(self, reflection: str) -> float:
        """Estimate confidence based on reflection content"""
        reflection_lower = reflection.lower()
        
        high_conf_words = ["accurate", "correct", "complete", "comprehensive", "detailed"]
        low_conf_words = ["unsure", "might", "possibly", "incomplete", "missing", "unclear"]
        
        high_count = sum(1 for word in high_conf_words if word in reflection_lower)
        low_count = sum(1 for word in low_conf_words if word in reflection_lower)
        
        if high_count > low_count:
            return min(0.8 + (high_count * 0.05), 1.0)
        elif low_count > high_count:
            return max(0.3 - (low_count * 0.05), 0.1)
        else:
            return 0.6  
    
    def get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        return self.stats.copy()
    
    def increment_groq_usage(self):
        """Track GROQ API usage"""
        self.stats["groq_responses"] += 1
        self.update_stats(self.stats["total_memories"])
    
    def increment_t5_usage(self):
        """Track T5 reflection usage"""
        self.stats["t5_reflections"] += 1
        self.update_stats(self.stats["total_memories"])

def load_and_split_dataset(test_size=0.1, seed=42):
    if os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(TEST_SPLIT_PATH):
        train_df = pd.read_csv(TRAIN_SPLIT_PATH)
        test_df  = pd.read_csv(TEST_SPLIT_PATH)
    else:
        df = pd.read_csv(CLEAN_CSV_PATH)
        assert "question_clean" in df and "answer_clean" in df, \
            "CSV must have question_clean & answer_clean"
        train_df, test_df = train_test_split(
            df[["question_clean", "answer_clean"]],
            test_size=test_size, random_state=seed, shuffle=True
        )
        train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
        test_df.to_csv(TEST_SPLIT_PATH, index=False)
    return train_df, test_df

class QADataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length, max_target_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = str(self.df.loc[idx, "question_clean"]).strip()
        answer = str(self.df.loc[idx, "answer_clean"]).strip()
        
        input_text = f"answer: {question}"
    
        target_text = answer
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def get_latest_checkpoint(model_dir):
    """Returns path to latest checkpoint folder, else fallback to model_dir"""
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    if not checkpoints:
        return model_dir
    return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

def get_model_and_tokenizer():
    """Load T5 model and tokenizer with proper error handling from latest checkpoint"""
    try:
        model_path = get_latest_checkpoint(MODEL_DIR)
        print(f" Loading model from: {model_path}")

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        model = model.to(DEVICE)
        print(f" Model loaded on {DEVICE}")
        return model, tokenizer

    except Exception as e:
        print(f" Critical error loading model: {e}")
        raise e

def train_model(train_df, test_df, output_dir, epochs):
    """Train T5 model for reflection and improvement"""
    print(f" Training T5 model on CPU...")
    
    model, tokenizer = get_model_and_tokenizer()
    
    train_dataset = QADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        seed=42,
        dataloader_pin_memory=False,  
        no_cuda=True,  
        fp16=False,  
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f" Starting training on {DEVICE}")
    print(f" Training examples: {len(train_dataset)}")
    print(f" Validation examples: {len(eval_dataset)}")
    
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(" T5 model training completed!")

def retrain_on_feedback(model_dir, memory_system):
    """Enhanced retraining with memory system"""
    feedback_list = memory_system.load_feedback()
    if not feedback_list:
        print("No feedback found in memory.")
        return
    
    feedback_data = []
    for entry in feedback_list:
        feedback_data.append({
            "question_clean": entry["question"],
            "answer_clean": entry["improvement"]
        })
    
    fb_df = pd.DataFrame(feedback_data)
    train_df, test_df = load_and_split_dataset()
    
    combined_df = pd.concat([train_df, fb_df], ignore_index=True)
    
    print(f" Retraining T5 model on {len(fb_df)} feedback items + {len(train_df)} original = {len(combined_df)} examples")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    train_dataset = QADataset(combined_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        # evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=RETRAIN_EPOCHS,
        logging_steps=10,
        learning_rate=3e-5,
        warmup_steps=50,
        weight_decay=0.01,
        no_cuda=True,
        fp16=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(" T5 model retraining completed!")

def analyze_answer_quality(self, question: str, answer: str, reflection: str) -> Dict:
    """Enhanced analysis of answer quality with specific criteria"""
    analysis = {
        "completeness": 0.5,
        "accuracy_confidence": 0.5,
        "specificity": 0.5,
        "clarity": 0.5,
        "issues_found": [],
        "improvement_suggestions": []
    }
    
    answer_lower = answer.lower()
    reflection_lower = reflection.lower()
    question_lower = question.lower()
    
    if len(answer.split()) < 10:
        analysis["completeness"] = 0.3
        analysis["issues_found"].append("too_brief")
    elif len(answer.split()) > 50:
        analysis["completeness"] = 0.8
    
    vague_phrases = ["might be", "could be", "possibly", "maybe", "not sure", "i think"]
    if any(phrase in answer_lower for phrase in vague_phrases):
        analysis["accuracy_confidence"] = 0.3
        analysis["issues_found"].append("vague_language")
    
    generic_responses = ["it depends", "various factors", "many ways", "different approaches"]
    if any(phrase in answer_lower for phrase in generic_responses):
        analysis["specificity"] = 0.3
        analysis["issues_found"].append("too_generic")
    
    negative_indicators = ["incomplete", "missing", "lacks", "unclear", "confusing", "wrong", "inaccurate"]
    positive_indicators = ["comprehensive", "accurate", "clear", "complete", "detailed", "good"]
    
    neg_count = sum(1 for word in negative_indicators if word in reflection_lower)
    pos_count = sum(1 for word in positive_indicators if word in reflection_lower)
    
    if neg_count > pos_count:
        analysis["accuracy_confidence"] = max(0.2, analysis["accuracy_confidence"] - 0.3)
        analysis["issues_found"].extend(["reflection_negative"])
    
    return analysis

def create_detailed_critique_prompt(self, question: str, answer: str, attempt_num: int = 1) -> str:
    """Force T5 to pinpoint exactly what to improve."""

    snippet = " ".join(answer.split()[:150]) + ("…" if len(answer.split())>150 else "")
    return (
        "critique: You are an expert. Identify any clarity, accuracy or completeness issues.\n"
        f"Question: {question}\n"
        f"Answer: {snippet}\n\n"
        "List issues and what is missing. Be concise."
    )

def create_targeted_improvement_prompt(self, question: str, current_answer: str, reflection: str, attempt_num: int = 1) -> str:
    """Force T5 to rewrite, not repeat."""
    return (
        "improve: Using the critique below, produce a substantially rewritten, clearer, more complete answer. Do NOT copy phrasing.\n"
        f"Question: {question}\n"
        f"Critique: {reflection}\n"
        f"Original Answer: {current_answer}\n\n"
        "Improved Answer:"
    )

def interactive_session(model_dir):
    
    console = Console()

    def format_bullets(text):
        text = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def display_answer(title: str, answer: str):
        formatted = format_bullets(answer)
        console.print(Panel.fit(Markdown(f"### {title}\n\n{formatted}"), border_style="cyan", title="Reflective Answer"))

    def safe_t5(prompt, **kwargs):
        try:
            return t5(prompt, **kwargs)[0]["generated_text"].strip()
        except Exception:
            short = " ".join(prompt.split()[:100])
            return t5(short, **kwargs)[0]["generated_text"].strip()

    try:
        groq = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq.check_connection():
            print(" GROQ API failed.")
            return
    except Exception as e:
        print("", e)
        return
    print(" GROQ ready")

    model, tokenizer = get_model_and_tokenizer()
    device_id = 0 if torch.cuda.is_available() else -1
    t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
    print(f" T5 on {'GPU' if device_id >= 0 else 'CPU'}")

    memory = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
    feedback_list = memory.load_feedback()

    print("\n=== Interactive Hybrid Reflective QA ===")
    print("Commands: help | stats | exit")

    while True:
        q = input("\nQuestion: ").strip()
        if not q: continue
        if q.lower() in ("exit", "quit"): break
        if q.lower() == "help":
            print(" help  • show this\n stats • memory stats\n exit  • quit session")
            continue
        if q.lower() == "stats":
            for k, v in memory.get_memory_stats().items():
                print(f" {k}: {v}")
            continue

        print("Generating initial answer with GROQ…")
        t0 = time.time()
        base = groq.generate_answer(q)
        memory.increment_groq_usage()
        console.print(Panel.fit(Markdown(f"### LLM Answer\n\n{format_bullets(base)}"), title="GROQ", border_style="magenta"))

        existing = memory.find_similar_question(q, feedback_list)
        if existing:
            sim = memory.calculate_similarity(q, existing["question"])
            console.print(Panel.fit(Markdown(f"### Memory Recall (sim={sim:.2f})\n\n{format_bullets(existing['improvement'])}"), border_style="green"))
            continue

        critique_prompt = (
            "critique: List factual errors or omissions in bullet points.\n\n"
            f"Q: {q}\nA: {base}"
        )
        print("Reflecting on LLM output…")
        reflection = safe_t5(
            critique_prompt,
            max_new_tokens=128,
            num_beams=2,
            do_sample=False
        )
        memory.increment_t5_usage()
        display_answer("T5 Reflection (Critique)", reflection)

        improve_prompt = (
            "improve: Use the critique below to rewrite the answer fully. "
            "Fix all errors, add missing info, and do NOT copy the original phrasing.\n\n"
            f"Q: {q}\n"
            f"Critique:\n{reflection}\n\n"
            f"Original Answer: {base}\n\n"
            f"Corrected Answer:"
        )
        corrected = safe_t5(
            improve_prompt,
            max_new_tokens=200,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            num_beams=1
        )
        display_answer("T5 Improved Answer", corrected)

        choice = input(" Accept this answer? (Y/n): ").strip().lower()
        if choice in ("", "y", "yes"):
            final = corrected
            entry = memory.auto_save_reflection(q, base, final, reflection)
            print(" Saved to memory.")
        else:
            user_fix = input(" Your correction (or blank to keep T5 version):\n").strip()
            if user_fix:
                final = user_fix
                entry = memory.save_user_correction(q, base, final, reflection)
                print(" User fix saved.")
            else:
                final = corrected
                entry = memory.auto_save_reflection(q, base, final, reflection)
                print(" Saved default improvement.")

        feedback_list.append(entry)
        memory.save_feedback(feedback_list)

    print(" Exiting. See you again!")



def test_hybrid_system(model_dir, test_df):
    """Test the hybrid GROQ-T5 system"""
    print("\n Testing Hybrid GROQ-T5 System")
    print("─" * 50)
    
    try:
        groq_handler = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq_handler.check_connection():
            print(" GROQ API not available for testing")
            return
    except Exception as e:
        print(f" GROQ initialization failed: {e}")
        return
    
    model, tokenizer = get_model_and_tokenizer()
    t5_pipeline = pipeline(
        "text2text-generation", 
        model=model, 
        tokenizer=tokenizer,
        device=-1
    )

    for i, (_, row) in enumerate(test_df.sample(min(3,len(test_df))).iterrows(), 1):
        question, gold = row["question_clean"], row["answer_clean"]
        
        print(f"\n[Test {i}]")
        print(f" Question: {question}")
        
        groq_answer = groq_handler.generate_answer(question)
        print(f" GROQ Answer: {groq_answer}")
        
        critique_prompt = f"critique: Question: {question} Answer: {groq_answer}"
        reflection = t5_pipeline(critique_prompt, max_new_tokens=128, num_beams=2)[0]["generated_text"].strip()
        print(f" T5 Reflection: {reflection}")
        
        print(f" Expected: {gold}")
        print("─" * 30)

def show_memory_overview():
    """Show hybrid system memory overview"""
    if os.path.exists(FEEDBACK_JSON):
        memory_system = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
        feedback_list = memory_system.load_feedback()
        stats = memory_system.get_memory_stats()
        
        print("\n Hybrid GROQ-T5 Memory System Overview")
        print("─" * 50)
        print(f" Total Memories: {len(feedback_list)}")
        print(f" GROQ API Calls: {stats.get('groq_responses', 0)}")
        print(f" T5 Reflections: {stats.get('t5_reflections', 0)}")
        print(f" Auto-saved: {stats.get('auto_saved', 0)}")
        print(f" User corrections: {stats.get('user_corrections', 0)}")
        print(f" Memory reuses: {stats.get('reuse_count', 0)}")
        print(f" Last updated: {stats.get('last_updated', 'Never')[:19]}")
        
        if feedback_list:
            print("\n Recent Hybrid Memories:")
            for fb in feedback_list[-3:]:  
                print(f"  • {fb['question'][:50]}... ({fb['source']})")
    else:
        print(" No memory data found")

def check_requirements():
    """Check if all requirements are met"""
    print("\n Checking System Requirements...")
    print(f" Device: {DEVICE} (CPU-only mode)")
    
    if not GROQ_API_KEY:
        print(" GROQ_API_KEY environment variable not set")
        print("   Please set it with: export GROQ_API_KEY=your_api_key")
        return False
    else:
        print(" GROQ API key found")
    
    try:
        groq_handler = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if groq_handler.check_connection():
            print(" GROQ API connection successful")
        else:
            print(" GROQ API connection failed")
            return False
    except Exception as e:
        print(f" GROQ API error: {e}")
        return False
    
    if not os.path.exists(CLEAN_CSV_PATH):
        print(f" Missing required file: {CLEAN_CSV_PATH}")
        return False
    else:
        print(" Training data file found")
    
    return True

def create_sample_data():
    """Create sample data for testing if main data file is missing"""
    print("\n Creating sample data for testing...")
    
    sample_data = [
        {"question_clean": "What is artificial intelligence?", 
         "answer_clean": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."},
        {"question_clean": "How does machine learning work?", 
         "answer_clean": "Machine learning works by training algorithms on data to recognize patterns and make predictions or decisions without being explicitly programmed for each specific task. It uses statistical techniques to improve performance through experience."},
        {"question_clean": "What is deep learning?", 
         "answer_clean": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data, particularly useful for tasks like image recognition and natural language processing."},
        {"question_clean": "What are neural networks?", 
         "answer_clean": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections, learning to recognize patterns and make predictions through training."},
        {"question_clean": "What is natural language processing?", 
         "answer_clean": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language in a meaningful way, including tasks like translation, sentiment analysis, and text summarization."}
    ]
    
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    df = pd.DataFrame(sample_data)
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f" Sample data created at {CLEAN_CSV_PATH}")

class FixedQADataset(Dataset):
    def __init__(self, df, tokenizer, max_input_length, max_target_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        question = str(self.df.loc[idx, "question_clean"]).strip()
        answer = str(self.df.loc[idx, "answer_clean"]).strip()
        
        input_text = f"Question: {question}"
        target_text = answer
        
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = input_encoding.input_ids.squeeze()
        attention_mask = input_encoding.attention_mask.squeeze()
        labels = target_encoding.input_ids.squeeze()
        
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def fixed_train_model(train_df, test_df, output_dir, epochs):
    """Fixed training function with proper loss calculation"""
    print(f" Training T5 model on CPU with fixed dataset...")
    
    model, tokenizer = get_model_and_tokenizer()
    
    train_dataset = FixedQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        # evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=epochs,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=5e-5,
        warmup_steps=100,
        weight_decay=0.01,
        seed=42,
        use_cpu=True,  
        dataloader_pin_memory=False,
        fp16=False,
        gradient_checkpointing=False,  
        remove_unused_columns=False, 
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(f" Starting training on CPU")
    print(f" Training examples: {len(train_dataset)}")
    print(f" Validation examples: {len(eval_dataset)}")
    
    print(" Checking first training example...")
    sample = train_dataset[0]
    print(f" Input IDs shape: {sample['input_ids'].shape}")
    print(f" Labels shape: {sample['labels'].shape}")
    print(f" Sample input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f" Sample target: {tokenizer.decode([id for id in sample['labels'] if id != -100], skip_special_tokens=True)}")
    
    trainer.train()
    
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(" T5 model training completed!")

def fixed_retrain_on_feedback(model_dir, memory_system):
    """Fixed retraining with proper dataset"""
    feedback_list = memory_system.load_feedback()
    if not feedback_list:
        print("No feedback found in memory.")
        return
    
    feedback_data = []
    for entry in feedback_list:
        feedback_data.append({
            "question_clean": entry["question"],
            "answer_clean": entry["improvement"]
        })
    
    fb_df = pd.DataFrame(feedback_data)
    train_df, test_df = load_and_split_dataset()
    
    combined_df = pd.concat([train_df, fb_df], ignore_index=True)
    
    print(f" Retraining T5 model on {len(fb_df)} feedback items + {len(train_df)} original = {len(combined_df)} examples")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    train_dataset = FixedQADataset(combined_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    eval_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    training_args = TrainingArguments(
        output_dir=model_dir,
        overwrite_output_dir=False,
        do_train=True,
        do_eval=True,
        # evaluation_strategy="steps",
        eval_steps=50,
        save_steps=100,
        save_total_limit=2,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=RETRAIN_EPOCHS,
        logging_steps=10,
        learning_rate=3e-5,
        warmup_steps=50,
        weight_decay=0.01,
        use_cpu=True,
        fp16=False,
        remove_unused_columns=False,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(" T5 model retraining completed!")

def main():
    """Main function to run the hybrid system"""
    print("=" * 60)
    print("    HYBRID GROQ-T5 REFLECTIVE QA SYSTEM")
    print("=" * 60)
    
    if not check_requirements():
        print("\n Creating sample data for testing...")
        create_sample_data()
    
    while True:
        print("\n Select an option:")
        print("1. Train T5 model from scratch")
        print("2. Interactive QA session (GROQ + T5)")
        print("3. Test hybrid system")
        print("4. Show memory overview")
        print("5. Retrain with feedback")
        print("6. Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            print("\n Starting T5 model training...")
            train_df, test_df = load_and_split_dataset()
            print(f"Training data: {len(train_df)} examples")
            print(f"Test data: {len(test_df)} examples")
            
            fixed_train_model(train_df, test_df, MODEL_DIR, NUM_TRAIN_EPOCHS)
            
        elif choice == "2":
            
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
            os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue

            interactive_session(MODEL_DIR)
            
        elif choice == "3":
            
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
                        os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue

            _, test_df = load_and_split_dataset()
            test_hybrid_system(MODEL_DIR, test_df)

            
        elif choice == "4":
            show_memory_overview()
            
        elif choice == "5":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
                        os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue

            memory_system = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
            fixed_retrain_on_feedback(MODEL_DIR, memory_system)
            
        elif choice == "6":
            print(" Goodbye!")
            break
            
        else:
            print(" Invalid choice. Please enter 1-6.")

if __name__ == "__main__":
    main()