import os
import json
import glob
import pandas as pd
import torch
import torch, time, re
import sqlite3
from contextlib import contextmanager
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

# MODIFIED PATHS FOR LEGAL ADVISOR STRUCTURE
PREPROCESSED_DIR   = "Dev/preprocessed_data/processed_constitution"
CLEAN_CSV_PATH     = os.path.join(PREPROCESSED_DIR, "train.csv")  # Changed from cleaned_qa_data.csv
TRAIN_SPLIT_PATH   = os.path.join(PREPROCESSED_DIR, "train.csv")
TEST_SPLIT_PATH    = os.path.join(PREPROCESSED_DIR, "test.csv")
VALIDATION_SPLIT_PATH = os.path.join(PREPROCESSED_DIR, "validation.csv")  # New validation set

MODEL_DIR          = "Dev/flan_t5_legal_advisor_model"  # Changed model directory name
DATABASE_PATH      = os.path.join(MODEL_DIR, "legal_feedback_system.db")  # Changed database name
LEGACY_FEEDBACK_JSON = os.path.join(MODEL_DIR, "legal_feedback.json")  # Changed feedback file name
MEMORY_STATS_JSON  = os.path.join(MODEL_DIR, "legal_memory_stats.json")  # Changed stats file name
os.makedirs(MODEL_DIR, exist_ok=True)

# LEGAL-SPECIFIC CONFIGURATIONS
PRETRAINED_MODEL   = "google/flan-t5-base"
MAX_INPUT_LENGTH   = 768  # Increased for longer legal queries
MAX_TARGET_LENGTH  = 768  # Increased for detailed legal responses
BATCH_SIZE         = 2
NUM_TRAIN_EPOCHS   = 5    # Increased epochs for legal domain complexity
RETRAIN_EPOCHS     = 3    # Increased retrain epochs
REFLECTION_RETRY_LIMIT = 3
SIMILARITY_THRESHOLD = 0.65  # Slightly higher threshold for legal precision

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_MAX_TOKENS = 768  # Increased for legal responses
DEVICE = "cpu"

# LEGAL-SPECIFIC DATASET LOADING (CORRECTED)
# LEGAL-SPECIFIC DATASET LOADING (DEFINITIVE FIX)
def load_and_split_legal_dataset(test_size=0.1, validation_size=0.1, seed=42):
    """
    Load, clean, and split the legal dataset.
    This version is robust and handles both raw and pre-cleaned CSV files to prevent errors.
    """
    if not os.path.exists(CLEAN_CSV_PATH):
        raise FileNotFoundError(f"Source legal data not found at {CLEAN_CSV_PATH}")

    df = pd.read_csv(CLEAN_CSV_PATH)
    print(f"Loaded source dataset with columns: {list(df.columns)}")

    # === START OF THE FIX ===
    # Check the format of the loaded CSV file
    if len(df.columns) >= 2 and list(df.columns)[:2] == ['question_clean', 'answer_clean']:
        # Case 1: The CSV is already cleaned from a previous run.
        print("✅ Detected already processed 2-column format. Using as is.")
        df_clean = df[['question_clean', 'answer_clean']].copy()
    
    elif len(df.columns) == 3:
        # Case 2: The CSV is the raw, unprocessed file with 3 unnamed columns.
        print("⚠️ Detected raw 3-column format. Applying cleaning and renaming...")
        df.columns = ['raw_question', 'answer', 'extra']
        df_clean = df[['raw_question', 'answer']].copy()
        df_clean.columns = ["question_clean", "answer_clean"]
        
        # Apply the cleaning logic to the raw data
        df_clean["question_clean"] = df_clean["question_clean"].astype(str).str.replace(r'\[.*?\]\s*', '', regex=True).str.strip()
        df_clean["answer_clean"] = df_clean["answer_clean"].astype(str).str.strip()
    
    else:
        # Case 3: The CSV format is unexpected.
        raise ValueError(f"Unexpected CSV format. Columns found: {list(df.columns)}. Expected 3 raw columns or 2 cleaned columns ('question_clean', 'answer_clean').")
    # === END OF THE FIX ===

    # Remove any null or empty rows that might result from cleaning
    df_clean.dropna(inplace=True)
    df_clean = df_clean[df_clean["question_clean"].str.len() > 0]
    df_clean = df_clean[df_clean["answer_clean"].str.len() > 0]
    
    print(f"Cleaned dataset ready for splitting: {len(df_clean)} valid pairs")
    if len(df_clean) == 0:
        raise ValueError("No valid question-answer pairs found after cleaning. Check your source CSV.")

    # Split the now-clean data into train, validation, and test sets
    train_val_df, test_df = train_test_split(
        df_clean, test_size=test_size, random_state=seed, shuffle=True
    )
    
    adjusted_val_size = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_val_df, test_size=adjusted_val_size, random_state=seed, shuffle=True
    )
    
    # Save the splits (this will overwrite with the same cleaned data, which is fine)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    test_df.to_csv(TEST_SPLIT_PATH, index=False)
    validation_df.to_csv(VALIDATION_SPLIT_PATH, index=False)
    
    print(f"✓ Created and saved splits: {len(train_df)} train, {len(test_df)} test, {len(validation_df)} validation")
    
    return train_df, test_df, validation_df

class LegalQADataset(Dataset):
    """Enhanced dataset class for legal Q&A with longer context support"""
    
    def __init__(self, df, tokenizer, max_input_length, max_target_length):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    # Inside the LegalQADataset class
    def __getitem__(self, idx):
        # This is a more standard way to access DataFrame rows
        row = self.df.iloc[idx]
        question = str(row["question_clean"]).strip()
        answer = str(row["answer_clean"]).strip()
        
        # Enhanced prompt for legal context
        input_text = f"Legal Question: {question}\n\nProvide a comprehensive legal answer:"
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

class LegalGroqLLMHandler(GroqLLMHandler):
    """Enhanced GROQ handler with legal-specific prompting"""
    
    def generate_legal_answer(self, legal_question: str, max_tokens: int = GROQ_MAX_TOKENS) -> str:
        """Generate legal answer using GROQ LLM with legal-specific system prompt"""
        try:
            legal_system_prompt = """You are a knowledgeable legal advisor assistant. Provide accurate, comprehensive legal information and analysis. 

IMPORTANT DISCLAIMERS:
- Always remind users that this is general legal information, not legal advice
- Recommend consulting with a qualified attorney for specific legal matters
- Be thorough but clear in explanations
- Reference relevant legal principles when applicable
- If uncertain about jurisdiction-specific laws, mention this limitation

Focus on being helpful while maintaining professional legal standards."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": legal_system_prompt},
                    {"role": "user", "content": f"Legal Question: {legal_question}"}
                ],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"  GROQ API Error: {e}")
            return f"I apologize, but I encountered an error generating a legal response: {str(e)}. Please consult with a qualified attorney for legal advice."

def train_legal_model():
    """Train the T5 model specifically for legal advisory"""
    print("Starting Legal Advisor T5 model training...")
    
    try:
        train_df, test_df, validation_df = load_and_split_legal_dataset()
        print(f"Legal dataset loaded: {len(train_df)} train, {len(validation_df)} validation, {len(test_df)} test samples")
    except Exception as e:
        print(f"Error loading legal dataset: {e}")
        return False
    
    print(f"Loading pretrained model: {PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    
    # Add legal-specific special tokens
    legal_special_tokens = {
        "additional_special_tokens": [
            "<legal_critique>", "<legal_improve>", "<legal_reflect>", 
            "<statute>", "<case_law>", "<precedent>", "<jurisdiction>"
        ]
    }
    tokenizer.add_special_tokens(legal_special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets with the enhanced LegalQADataset
    train_dataset = LegalQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    validation_dataset = LegalQADataset(validation_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    test_dataset = LegalQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Minimal training arguments for maximum compatibility
    # Inside the train_legal_model function
    training_args = TrainingArguments(
        output_dir=MODEL_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_DIR, "logs"),
        logging_steps=50,
        # evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        # load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False, # Set to False since you're on CPU
    )
    # training_args = TrainingArguments(
    #     output_dir=MODEL_DIR,
    #     overwrite_output_dir=True,
    #     num_train_epochs=NUM_TRAIN_EPOCHS,
    #     per_device_train_batch_size=BATCH_SIZE, # This will now be 1
    #     gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps
    #     per_device_eval_batch_size=2, # Use a small batch size for evaluation
    #     # ... other arguments remain the same
    #     warmup_steps=100,
    #     weight_decay=0.01,
    #     logging_dir=f"{MODEL_DIR}/logs",
    #     logging_strategy="epoch",
    #     # evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     save_total_limit=2,
    #     load_best_model_at_end=True,
    #     metric_for_best_model="eval_loss",
    #     greater_is_better=False,
    #     fp16=torch.cuda.is_available(),
    #     dataloader_pin_memory=torch.cuda.is_available(),
    #     report_to="none",
    # )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,  # Use validation set for evaluation
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting legal model training...")
    trainer.train()
    
    print("Saving final legal model...")
    trainer.save_model()
    tokenizer.save_pretrained(MODEL_DIR)
    
    # Simple evaluation on test set
    print("Evaluating on test set...")
    try:
        test_results = trainer.evaluate(eval_dataset=test_dataset)
        print(f"Test Results: {test_results}")
    except Exception as e:
        print(f"Evaluation error (non-critical): {e}")
    
    print("Legal Advisor training completed!")
    return True

# Add missing functions from original code
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

def check_requirements():
    """Check if required files and directories exist"""
    required_files = [CLEAN_CSV_PATH]
    required_dirs = [PREPROCESSED_DIR, MODEL_DIR]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
        print(f" Created missing directories: {missing_dirs}")
    
    if missing_files:
        print(f" Missing required files: {missing_files}")
        return False
    
    return True

class EnhancedMemorySystem:
    def __init__(self, db_path: str, stats_json_path: str, similarity_threshold: float = 0.7):
        self.db_manager = DatabaseManager(db_path)
        self.stats_path = stats_json_path
        self.similarity_threshold = similarity_threshold
        
        if os.path.exists(LEGACY_FEEDBACK_JSON):
            migrated = self.db_manager.migrate_from_json(LEGACY_FEEDBACK_JSON)
            if migrated > 0:
                self.migrate_legacy_stats()
    
    def migrate_legacy_stats(self):
        """Migrate legacy JSON stats to database"""
        if os.path.exists(self.stats_path):
            try:
                with open(self.stats_path, 'r', encoding='utf-8') as f:
                    legacy_stats = json.load(f)
                
                self.db_manager.update_system_stats(legacy_stats)
                
                backup_path = self.stats_path + '.backup'
                os.rename(self.stats_path, backup_path)
                print(f"✓ Legacy stats migrated and backed up to {backup_path}")
                
            except Exception as e:
                print(f"Error migrating legacy stats: {e}")
    
    def load_feedback(self) -> List[Dict]:
        """Load feedback from database"""
        return self.db_manager.get_all_feedback()
    
    def save_feedback(self, feedback_list: List[Dict]):
        """Save feedback list (for compatibility - not recommended for new code)"""
        for feedback in feedback_list:
            self.db_manager.insert_feedback(feedback)
    
    def save_single_feedback(self, feedback_data: Dict) -> bool:
        """Save single feedback entry (recommended method)"""
        return self.db_manager.insert_feedback(feedback_data)
    
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
    
    def find_similar_question(self, question: str) -> Optional[Dict]:
        """Find the most similar question in database with similarity above threshold"""
        question_hash = hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
        feedback_list = self.db_manager.find_similar_feedback(question_hash)
        
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
            current_stats = self.db_manager.get_system_stats()
            reuse_count = current_stats.get('reuse_count', 0) + 1
            self.db_manager.update_system_stats({'reuse_count': reuse_count})
            
            print(f" Found similar question (similarity: {best_similarity:.2f})")
            
        return best_match
    
    def auto_save_reflection(self, question, groq_answer, improved_answer, reflection) -> Dict:
        """Auto-save reflection to database"""
        entry = {
            "question": question,
            "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8],
            "groq_answer": groq_answer,
            "improvement": improved_answer,
            "t5_reflection": reflection,
            "source": "auto-reflection",
            "confidence_score": self.estimate_confidence(reflection),
            "improvement_type": "groq-t5-hybrid",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        if self.db_manager.insert_feedback(entry):
            current_stats = self.db_manager.get_system_stats()
            auto_saved = current_stats.get('auto_saved', 0) + 1
            self.db_manager.update_system_stats({'auto_saved': auto_saved})
            
        return entry
    
    def save_user_correction(self, question: str, groq_answer: str, 
                             user_correction: str, reflection: str) -> Dict:
        """Save user-provided correction to database"""
        feedback_entry = {
            "question": question,
            "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8],
            "groq_answer": groq_answer,
            "improvement": user_correction,
            "t5_reflection": reflection,
            "source": "user-correction",
            "confidence_score": 1.0,
            "improvement_type": "user-feedback",
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        if self.db_manager.insert_feedback(feedback_entry):
            current_stats = self.db_manager.get_system_stats()
            user_corrections = current_stats.get('user_corrections', 0) + 1
            self.db_manager.update_system_stats({'user_corrections': user_corrections})
            
            print(" Saved user correction to database")
        
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
        """Get current memory statistics from database"""
        return self.db_manager.get_system_stats()
    
    def increment_groq_usage(self):
        """Track GROQ API usage"""
        current_stats = self.db_manager.get_system_stats()
        groq_responses = current_stats.get('groq_responses', 0) + 1
        self.db_manager.update_system_stats({'groq_responses': groq_responses})
    
    def increment_t5_usage(self):
        """Track T5 reflection usage"""
        current_stats = self.db_manager.get_system_stats()
        t5_reflections = current_stats.get('t5_reflections', 0) + 1
        self.db_manager.update_system_stats({'t5_reflections': t5_reflections})
    
    def cleanup_old_data(self, days: int = 90):
        """Clean up old feedback entries"""
        return self.db_manager.cleanup_old_entries(days)

class DatabaseManager:
    """SQLite database manager for feedback system"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT NOT NULL,
                    question_hash TEXT NOT NULL,
                    groq_answer TEXT NOT NULL,
                    improvement TEXT NOT NULL,
                    t5_reflection TEXT NOT NULL,
                    source TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    improvement_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(question_hash)
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_stats (
                    id INTEGER PRIMARY KEY,
                    total_memories INTEGER DEFAULT 0,
                    auto_saved INTEGER DEFAULT 0,
                    user_corrections INTEGER DEFAULT 0,
                    reuse_count INTEGER DEFAULT 0,
                    groq_responses INTEGER DEFAULT 0,
                    t5_reflections INTEGER DEFAULT 0,
                    last_updated TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('SELECT COUNT(*) FROM system_stats')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO system_stats (id, last_updated) 
                    VALUES (1, datetime('now'))
                ''')
            
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question_hash ON feedback(question_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON feedback(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
            
            conn.commit()
            print("✓ Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  
        try:
            yield conn
        finally:
            conn.close()
    
    def migrate_from_json(self, json_path: str) -> int:
        """Migrate existing JSON feedback to database"""
        if not os.path.exists(json_path):
            return 0
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
            
            migrated_count = 0
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                for entry in json_data:
                    try:
                        cursor.execute('''
                            INSERT OR REPLACE INTO feedback 
                            (question, question_hash, groq_answer, improvement, 
                             t5_reflection, source, confidence_score, improvement_type, timestamp)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            entry['question'],
                            entry.get('question_hash', hashlib.md5(entry['question'].lower().strip().encode()).hexdigest()[:8]),
                            entry['groq_answer'],
                            entry['improvement'],
                            entry['t5_reflection'],
                            entry['source'],
                            entry.get('confidence_score', 0.5),
                            entry.get('improvement_type', 'unknown'),
                            entry['timestamp']
                        ))
                        migrated_count += 1
                    except sqlite3.Error as e:
                        print(f"Error migrating entry: {e}")
                        continue
                
                conn.commit()
            
            backup_path = json_path + '.backup'
            os.rename(json_path, backup_path)
            print(f"✓ Migrated {migrated_count} entries from JSON to database")
            print(f"✓ JSON backup saved as {backup_path}")
            
            return migrated_count
            
        except Exception as e:
            print(f"Error during migration: {e}")
            return 0
    
    def insert_feedback(self, feedback_data: Dict) -> bool:
        """Insert new feedback entry"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO feedback 
                    (question, question_hash, groq_answer, improvement, 
                     t5_reflection, source, confidence_score, improvement_type, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    feedback_data['question'],
                    feedback_data['question_hash'],
                    feedback_data['groq_answer'],
                    feedback_data['improvement'],
                    feedback_data['t5_reflection'],
                    feedback_data['source'],
                    feedback_data['confidence_score'],
                    feedback_data['improvement_type'],
                    feedback_data['timestamp']
                ))
                conn.commit()
                return True
        except sqlite3.Error as e:
            print(f"Error inserting feedback: {e}")
            return False
    
    def get_all_feedback(self) -> List[Dict]:
        """Get all feedback entries"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM feedback ORDER BY created_at DESC')
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error retrieving feedback: {e}")
            return []
    
    def find_similar_feedback(self, question_hash: str, limit: int = 10) -> List[Dict]:
        """Find feedback entries for similarity matching"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    SELECT * FROM feedback 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit * 5,)) 
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except sqlite3.Error as e:
            print(f"Error finding similar feedback: {e}")
            return []
    
    def get_feedback_stats(self) -> Dict:
        """Get comprehensive feedback statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute('SELECT COUNT(*) FROM feedback')
                total_feedback = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE source = "auto-reflection"')
                auto_saved = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE source = "user-correction"')
                user_corrections = cursor.fetchone()[0]
                
                cursor.execute('''
                    SELECT COUNT(*) FROM feedback 
                    WHERE created_at >= datetime('now', '-7 days')
                ''')
                recent_week = cursor.fetchone()[0]
                
                cursor.execute('SELECT AVG(confidence_score) FROM feedback')
                avg_confidence = cursor.fetchone()[0] or 0.0
                
                return {
                    'total_feedback': total_feedback,
                    'auto_saved': auto_saved,
                    'user_corrections': user_corrections,
                    'recent_week': recent_week,
                    'avg_confidence': round(avg_confidence, 3)
                }
        except sqlite3.Error as e:
            print(f"Error getting feedback stats: {e}")
            return {}
    
    def update_system_stats(self, stats_update: Dict):
        """Update system statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                
                set_clauses = []
                values = []
                
                for key, value in stats_update.items():
                    if key != 'id':
                        set_clauses.append(f"{key} = ?")
                        values.append(value)
                
                if set_clauses:
                    set_clauses.append("updated_at = datetime('now')")
                    query = f"UPDATE system_stats SET {', '.join(set_clauses)} WHERE id = 1"
                    cursor.execute(query, values)
                    conn.commit()
                    
        except sqlite3.Error as e:
            print(f"Error updating system stats: {e}")
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM system_stats WHERE id = 1')
                row = cursor.fetchone()
                if row:
                    stats = dict(row)
                    
                    feedback_stats = self.get_feedback_stats()
                    stats.update(feedback_stats)
                    return stats
                return {}
        except sqlite3.Error as e:
            print(f"Error getting system stats: {e}")
            return {}
    
    def cleanup_old_entries(self, days: int = 90):
        """Clean up entries older than specified days"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    DELETE FROM feedback 
                    WHERE created_at < datetime('now', '-{} days')
                '''.format(days))
                deleted_count = cursor.rowcount
                conn.commit()
                print(f"✓ Cleaned up {deleted_count} old entries")
                return deleted_count
        except sqlite3.Error as e:
            print(f"Error cleaning up old entries: {e}")
            return 0

def legal_interactive_session(model_dir):
    """Enhanced interactive session for legal Q&A"""
    console = Console()

    def format_legal_response(text):
        """Format legal responses with proper structure"""
        text = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", text)
        text = re.sub(r"\n{2,}", "\n", text)
        # Add legal disclaimer if not present
        if "not legal advice" not in text.lower() and "consult" not in text.lower():
            text += "\n\n⚖️ **Disclaimer**: This is general legal information, not legal advice. Please consult with a qualified attorney for your specific situation."
        return text.strip()

    def display_legal_answer(title: str, answer: str, border_color: str = "cyan"):
        formatted = format_legal_response(answer)
        console.print(Panel.fit(
            Markdown(f"### {title}\n\n{formatted}"), 
            border_style=border_color, 
            title="Legal Advisory Response"
        ))

    def safe_legal_t5(prompt, **kwargs):
        try:
            return t5(prompt, **kwargs)[0]["generated_text"].strip()
        except Exception as e:
            print(f"T5 Error: {e}")
            short_prompt = " ".join(prompt.split()[:150])  # Longer prompt for legal
            return t5(short_prompt, **kwargs)[0]["generated_text"].strip()

    # Initialize legal-specific GROQ handler
    try:
        groq = LegalGroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq.check_connection():
            print("❌ GROQ API connection failed.")
            return
    except Exception as e:
        print(f"❌ GROQ initialization error: {e}")
        return
    print("✅ Legal GROQ handler ready")

    model, tokenizer = get_model_and_tokenizer()
    device_id = 0 if torch.cuda.is_available() else -1
    t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
    print(f"✅ Legal T5 model loaded on {'GPU' if device_id >= 0 else 'CPU'}")

    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)

    print("\n" + "="*70)
    print("🏛️  INTERACTIVE LEGAL ADVISOR - CONSTITUTION EXPERT  ⚖️")
    print("="*70)
    print("Commands: help | stats | cleanup | cases | disclaimer | exit")
    print("\n📋 Ask legal questions about constitutional law, rights, and procedures.")

    while True:
        q = input("\n📝 Legal Question: ").strip()
        if not q: continue
        if q.lower() in ("exit", "quit"): break
        
        if q.lower() == "help":
            console.print(Panel("""
**Available Commands:**
• help - Show this help message
• stats - Display memory and usage statistics  
• cleanup - Clean old feedback data
• cases - Show recent legal case examples
• disclaimer - Show legal disclaimer
• exit - End legal advisory session
            """, title="Legal Advisor Help", border_style="blue"))
            continue
            
        if q.lower() == "disclaimer":
            console.print(Panel("""
⚖️ **IMPORTANT LEGAL DISCLAIMER**

This AI system provides general legal information and educational content only. 
It does NOT provide legal advice and is not a substitute for consultation with a 
qualified attorney.

• Responses are for informational purposes only
• Laws vary by jurisdiction and change frequently  
• Specific legal matters require professional legal counsel
• No attorney-client relationship is created
• Always verify information with current legal sources

**For legal advice specific to your situation, please consult with a licensed attorney.**
            """, title="Legal Disclaimer", border_style="red"))
            continue
            
        if q.lower() == "stats":
            stats = memory.get_memory_stats()
            console.print(Panel(f"""
**Legal Advisory Statistics:**
📊 Total Legal Consultations: {stats.get('total_feedback', 0)}
🤖 Auto-saved Analyses: {stats.get('auto_saved', 0)}
👨‍💼 User Corrections: {stats.get('user_corrections', 0)}
🔄 Memory Reuse: {stats.get('reuse_count', 0)}
📈 Confidence Average: {stats.get('avg_confidence', 0.0):.3f}
            """, title="System Statistics", border_style="green"))
            continue
            
        if q.lower() == "cleanup":
            days = input("Delete legal consultation records older than how many days? (default 90): ").strip()
            try:
                days = int(days) if days else 90
                deleted = memory.cleanup_old_data(days)
                print(f"🧹 Cleaned up {deleted} old legal consultation records")
            except ValueError:
                print("❌ Invalid number of days")
            continue

        print("🔍 Analyzing legal question with GROQ...")
        t0 = time.time()
        legal_answer = groq.generate_legal_answer(q)
        memory.increment_groq_usage()
        
        display_legal_answer("Initial Legal Analysis", legal_answer, "magenta")

        # Check for similar legal questions in memory
        existing = memory.find_similar_question(q)
        if existing:
            sim = memory.calculate_similarity(q, existing["question"])
            display_legal_answer(f"Previous Legal Analysis (similarity={sim:.2f})", existing['improvement'], "green")
            continue

        # Generate legal critique using T5
        legal_critique_prompt = (
            "legal_critique: Analyze this legal response for accuracy, completeness, and potential issues. "
            "Consider constitutional principles, precedents, and jurisdictional variations.\n\n"
            f"Legal Question: {q}\n"
            f"Legal Response: {legal_answer}"
        )
        
        print("⚖️ Generating legal critique with T5...")
        legal_reflection = safe_legal_t5(
            legal_critique_prompt,
            max_new_tokens=200,
            num_beams=3,
            do_sample=False,
            temperature=0.3
        )
        memory.increment_t5_usage()
        display_legal_answer("Legal Analysis Critique", legal_reflection, "yellow")

        # Generate improved legal response
        legal_improve_prompt = (
            "legal_improve: Using the legal critique below, provide a comprehensive and improved legal response. "
            "Address all identified issues, add relevant constitutional references, and ensure accuracy.\n\n"
            f"Legal Question: {q}\n"
            f"Legal Critique: {legal_reflection}\n\n"
            f"Original Response: {legal_answer}\n\n"
            f"Improved Legal Response:"
        )
        
        improved_legal_answer = safe_legal_t5(
            legal_improve_prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_beams=1
        )
        
        display_legal_answer("Enhanced Legal Analysis", improved_legal_answer, "cyan")

        print("💾 Auto-saving legal analysis to database...")
        memory.auto_save_reflection(q, legal_answer, improved_legal_answer, legal_reflection)

        # Get user feedback on legal response
        feedback = input("\n🤔 Is this legal analysis helpful and accurate? (y/n/provide correction): ").strip().lower()
        
        if feedback.startswith('n') or (feedback not in ['y', 'yes', '']):
            if feedback in ['n', 'no']:
                user_correction = input("📝 Please provide additional legal insights or corrections: ").strip()
            else:
                user_correction = feedback
            
            if user_correction:
                user_legal_critique_prompt = (
                    f"legal_critique: Compare this expert legal correction with the previous response and "
                    f"explain the legal improvements made.\n\n"
                    f"Legal Question: {q}\n"
                    f"Previous Analysis: {improved_legal_answer}\n"
                    f"Expert Correction: {user_correction}"
                )
                
                user_reflection = safe_legal_t5(
                    user_legal_critique_prompt,
                    max_new_tokens=150,
                    num_beams=2,
                    do_sample=False
                )
                
                memory.save_user_correction(q, legal_answer, user_correction, user_reflection)
                display_legal_answer("Expert Legal Correction Saved", user_correction, "green")
        
        elapsed = time.time() - t0
        print(f"⏱️  Total consultation time: {elapsed:.1f}s")

def create_legal_sample_data():
    """Create sample legal/constitutional data for testing"""
    print("🏛️ Creating sample legal constitutional dataset...")
    
    legal_sample_data = {
        'question': [
            "What are the fundamental rights guaranteed under the Constitution?",
            "How does the separation of powers work in constitutional law?",
            "What is due process and when does it apply?",
            "Explain the concept of judicial review",
            "What are the limits of free speech under the First Amendment?",
            "How does the equal protection clause work?",
            "What is the role of federalism in constitutional law?",
            "When can the government restrict constitutional rights?",
            "What are Miranda rights and when do they apply?",
            "How does constitutional interpretation work?"
        ],
        'answer': [
            "Fundamental rights under the Constitution include life, liberty, property, due process, equal protection, freedom of speech, religion, press, assembly, and petition. These rights are protected against government infringement and form the foundation of constitutional democracy.",
            "Separation of powers divides government authority among three branches: legislative (makes laws), executive (enforces laws), and judicial (interprets laws). Each branch has checks and balances over the others to prevent abuse of power and maintain constitutional balance.",
            "Due process requires fair legal procedures before the government can deprive someone of life, liberty, or property. It includes both procedural due process (fair procedures) and substantive due process (protection of fundamental rights from arbitrary government action).",
            "Judicial review is the power of courts to examine and invalidate government actions that violate the Constitution. Established in Marbury v. Madison (1803), it allows courts to serve as the final arbiter of constitutional meaning and protect constitutional supremacy.",
            "Free speech protections are broad but not absolute. The government cannot restrict speech based on content or viewpoint, but may regulate time, place, and manner. Exceptions include incitement to violence, true threats, defamation, and obscenity.",
            "The Equal Protection Clause requires that similarly situated people be treated alike by the law. It prohibits discrimination and requires that government classifications serve legitimate purposes. Strict scrutiny applies to suspect classifications like race.",
            "Federalism divides power between federal and state governments. The Constitution grants specific powers to the federal government while reserving others to states. The Supremacy Clause makes federal law supreme when conflicts arise within federal authority.",
            "Constitutional rights may be restricted when the government has a compelling interest and uses the least restrictive means (strict scrutiny), or meets intermediate scrutiny for certain rights. Emergency powers and public safety can justify some restrictions.",
            "Miranda rights inform suspects of their Fifth Amendment right against self-incrimination and Sixth Amendment right to counsel during custodial interrogation. Police must give these warnings before questioning suspects in custody.",
            "Constitutional interpretation involves various methods including textualism (plain meaning), originalism (original intent/understanding), living constitution (evolving interpretation), and precedent (stare decisis). Courts balance these approaches in constitutional cases."
        ]
    }
    
    df = pd.DataFrame(legal_sample_data)
    
    # Ensure the directory exists
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f"📚 Created sample legal constitutional dataset: {CLEAN_CSV_PATH}")
    print(f"📊 Dataset contains {len(df)} legal Q&A pairs")
    return True

# Update the main function to use legal-specific functions
def main():
    """Main function with legal advisor interface"""
    print("=" * 70)
    print("🏛️  HYBRID GROQ-T5 LEGAL ADVISOR SYSTEM  ⚖️")
    print("    Constitutional Law Expert with AI Memory")
    print("=" * 70)
    
    if not check_requirements():
        response = input("\n❓ Missing required legal dataset. Create sample constitutional data for testing? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            create_legal_sample_data()
        else:
            print("❌ Cannot continue without legal training data.")
            return
    
    while True:
        print("\n" + "="*50)
        print("🏛️        LEGAL ADVISOR MAIN MENU        ⚖️")
        print("="*50)
        print("1. Train Legal T5 model from constitutional data")
        print("2. Interactive Legal Advisory session")
        print("3. Test legal advisory system")
        print("4. Show legal consultation memory")
        print("5. Retrain with legal feedback")
        print("6. Legal database maintenance")
        print("7. Export legal consultation data")
        print("8. System statistics")
        print("9. Legal disclaimer")
        print("10. Exit")
        print("="*50)
        
        choice = input("\n📋 Enter choice (1-10): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting Legal T5 model training...")
            try:
                success = train_legal_model()
                if success:
                    print("✅ Legal model training completed successfully!")
                else:
                    print("❌ Legal model training failed.")
            except Exception as e:
                print(f"❌ Training failed: {e}")
        
        elif choice == "2":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print("❌ No trained legal model found. Please train the model first (option 1).")
                continue
            
            print("\n🏛️ Starting Legal Advisory session...")
            legal_interactive_session(MODEL_DIR)
            
        elif choice == "9":
            console = Console()
            console.print(Panel("""
⚖️ **IMPORTANT LEGAL DISCLAIMER**

This AI Legal Advisor system provides general legal information and educational 
content about constitutional law and legal principles. It does NOT provide legal 
advice and cannot replace consultation with a qualified attorney.

**Key Points:**
• All responses are for informational and educational purposes only
• Laws vary significantly by jurisdiction and change frequently
• Specific legal matters require professional legal counsel
• No attorney-client relationship is created through use of this system
• Always verify legal information with current authoritative sources
• For legal advice specific to your situation, consult a licensed attorney

**Constitutional Focus:**
This system is trained primarily on constitutional law principles. For other 
areas of law, specialized legal counsel is strongly recommended.
            """, title="LEGAL DISCLAIMER", border_style="red"))
            
        elif choice == "10":
            print("\n👋 Thank you for using the Legal Advisor System!")
            print("⚖️ Remember: Always consult with a qualified attorney for legal advice.")
            break
        
        else:
            # Handle other menu options similar to original code but with legal context
            print("❌ Invalid choice. Please enter 1-10.")
    
    print("\n" + "="*70)
    print("🏛️    LEGAL ADVISORY SESSION ENDED    ⚖️")
    print("="*70)

if __name__ == "__main__":
    main()

