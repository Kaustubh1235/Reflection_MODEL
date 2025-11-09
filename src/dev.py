
# import os
# import json
# import glob
# import pandas as pd
# import torch
# import torch, time, re
# from transformers import pipeline
# from rich.console import Console
# from rich.markdown import Markdown
# from rich.panel import Panel
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq,
#     pipeline,
# )
# import difflib
# from typing import Dict, List, Optional, Tuple
# import hashlib
# from groq import Groq
# import time
# from dotenv import load_dotenv
# load_dotenv()


# PREPROCESSED_DIR   = "./preprocessed_data"
# CLEAN_CSV_PATH     = os.path.join(PREPROCESSED_DIR, "cleaned_qa_data.csv")
# TRAIN_SPLIT_PATH   = os.path.join(PREPROCESSED_DIR, "train_split.csv")
# TEST_SPLIT_PATH    = os.path.join(PREPROCESSED_DIR, "test_split.csv")

# MODEL_DIR          = "./flan_t5_reflective_model"
# FEEDBACK_JSON      = os.path.join(MODEL_DIR, "feedback.json")
# MEMORY_STATS_JSON  = os.path.join(MODEL_DIR, "memory_stats.json")
# os.makedirs(MODEL_DIR, exist_ok=True)

# PRETRAINED_MODEL   = "google/flan-t5-base"
# MAX_INPUT_LENGTH   = 512
# MAX_TARGET_LENGTH  = 512
# BATCH_SIZE         = 2
# NUM_TRAIN_EPOCHS   = 3
# RETRAIN_EPOCHS     = 2
# REFLECTION_RETRY_LIMIT = 2
# SIMILARITY_THRESHOLD = 0.6  

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GROQ_MODEL = "Llama3-70B-8192"  
# GROQ_MAX_TOKENS = 512
# DEVICE = "cpu"

# class GroqLLMHandler:
#     def __init__(self, api_key: str, model: str = GROQ_MODEL):
#         if not api_key:
#             raise ValueError("GROQ API key not found. Please set GROQ_API_KEY environment variable.")
        
#         self.client = Groq(api_key=api_key)
#         self.model = model
        
#     def generate_answer(self, question: str, max_tokens: int = GROQ_MAX_TOKENS) -> str:
#         """Generate initial answer using GROQ LLM"""
#         try:
#             system_prompt = """You are a helpful and knowledgeable assistant. Provide accurate, concise, and informative answers to questions. Focus on being helpful while being direct and to the point."""
            
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"Question: {question}"}
#                 ],
#                 max_tokens=max_tokens,
#                 temperature=0.7,
#             )
            
#             return response.choices[0].message.content.strip()
            
#         except Exception as e:
#             print(f"  GROQ API Error: {e}")
#             return f"Sorry, I couldn't generate an answer due to an API error: {str(e)}"
    
#     def check_connection(self) -> bool:
#         """Test GROQ API connection"""
#         try:
#             response = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[{"role": "user", "content": "Hello"}],
#                 max_tokens=10,
#             )
#             return True
#         except Exception as e:
#             print(f" GROQ Connection Error: {e}")
#             return False

# class EnhancedMemorySystem:
#     def __init__(self, feedback_json_path: str, stats_json_path: str, similarity_threshold: float = 0.7):
#         self.feedback_path = feedback_json_path
#         self.stats_path = stats_json_path
#         self.similarity_threshold = similarity_threshold
#         self.stats = self.load_stats()
    
#     def load_feedback(self) -> List[Dict]:
#         """Load feedback from JSON file"""
#         if os.path.exists(self.feedback_path):
#             return json.load(open(self.feedback_path, "r", encoding="utf-8"))
#         return []
    
#     def save_feedback(self, feedback_list: List[Dict]):
#         """Save feedback to JSON file"""
#         json.dump(feedback_list, open(self.feedback_path, "w", encoding="utf-8"),
#                   indent=2, ensure_ascii=False)
#         self.update_stats(len(feedback_list))
    
#     def load_stats(self) -> Dict:
#         """Load memory statistics"""
#         if os.path.exists(self.stats_path):
#             return json.load(open(self.stats_path, "r", encoding="utf-8"))
#         return {
#             "total_memories": 0,
#             "auto_saved": 0,
#             "user_corrections": 0,
#             "reuse_count": 0,
#             "groq_responses": 0,
#             "t5_reflections": 0,
#             "last_updated": pd.Timestamp.now().isoformat()
#         }
    
#     def update_stats(self, total_memories: int):
#         """Update memory statistics"""
#         self.stats["total_memories"] = total_memories
#         self.stats["last_updated"] = pd.Timestamp.now().isoformat()
#         json.dump(self.stats, open(self.stats_path, "w", encoding="utf-8"), indent=2)
    
#     def calculate_similarity(self, question1: str, question2: str) -> float:
#         """Calculate similarity between two questions using multiple methods"""
#         q1_clean = question1.strip().lower()
#         q2_clean = question2.strip().lower()
        
#         if q1_clean == q2_clean:
#             return 1.0
        
#         if q1_clean in q2_clean or q2_clean in q1_clean:
#             return 0.9
        
#         sequence_similarity = difflib.SequenceMatcher(None, q1_clean, q2_clean).ratio()
        
#         words1 = set(q1_clean.split())
#         words2 = set(q2_clean.split())
#         if len(words1) == 0 or len(words2) == 0:
#             word_similarity = 0.0
#         else:
#             intersection = len(words1.intersection(words2))
#             union = len(words1.union(words2))
#             word_similarity = intersection / union
        
#         combined_similarity = (sequence_similarity * 0.6) + (word_similarity * 0.4)
#         return combined_similarity
    
#     def find_similar_question(self, question: str, feedback_list: List[Dict]) -> Optional[Dict]:
#         """Find the most similar question in feedback with similarity above threshold"""
#         if not feedback_list:
#             return None
        
#         best_match = None
#         best_similarity = 0.0
        
#         for feedback in feedback_list:
#             similarity = self.calculate_similarity(question, feedback["question"])
#             if similarity > best_similarity and similarity >= self.similarity_threshold:
#                 best_similarity = similarity
#                 best_match = feedback
        
#         if best_match:
#             self.stats["reuse_count"] += 1
#             self.update_stats(self.stats["total_memories"])
#             print(f" Found similar question (similarity: {best_similarity:.2f})")
            
#         return best_match
    
#     def auto_save_reflection(self, question, groq_answer, improved_answer, reflection):
#         entry = {
#         "question": question,
#         "groq_answer": groq_answer,
#         "improvement": improved_answer,
#         "t5_reflection": reflection,
#         "source": "auto-reflection",
#         "confidence_score": self.estimate_confidence(reflection),
#         "improvement_type": "groq-t5-hybrid",
#         "timestamp": pd.Timestamp.now().isoformat(),
#         "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
#         }
#         self.stats["auto_saved"] += 1
#         return entry

    
#     def save_user_correction(self, question: str, groq_answer: str, 
#                            user_correction: str, reflection: str) -> Dict:
#         """Save user-provided correction"""
#         feedback_entry = {
#             "question": question,
#             "groq_answer": groq_answer,
#             "improvement": user_correction,
#             "t5_reflection": reflection,
#             "source": "user-correction",
#             "confidence_score": 1.0,  
#             "improvement_type": "user-feedback",
#             "timestamp": pd.Timestamp.now().isoformat(),
#             "question_hash": hashlib.md5(question.lower().strip().encode()).hexdigest()[:8]
#         }
        
#         self.stats["user_corrections"] += 1
#         print(" Saved user correction to memory")
#         return feedback_entry
    
#     def estimate_confidence(self, reflection: str) -> float:
#         """Estimate confidence based on reflection content"""
#         reflection_lower = reflection.lower()
        
#         high_conf_words = ["accurate", "correct", "complete", "comprehensive", "detailed"]
#         low_conf_words = ["unsure", "might", "possibly", "incomplete", "missing", "unclear"]
        
#         high_count = sum(1 for word in high_conf_words if word in reflection_lower)
#         low_count = sum(1 for word in low_conf_words if word in reflection_lower)
        
#         if high_count > low_count:
#             return min(0.8 + (high_count * 0.05), 1.0)
#         elif low_count > high_count:
#             return max(0.3 - (low_count * 0.05), 0.1)
#         else:
#             return 0.6  
    
#     def get_memory_stats(self) -> Dict:
#         """Get current memory statistics"""
#         return self.stats.copy()
    
#     def increment_groq_usage(self):
#         """Track GROQ API usage"""
#         self.stats["groq_responses"] += 1
#         self.update_stats(self.stats["total_memories"])
    
#     def increment_t5_usage(self):
#         """Track T5 reflection usage"""
#         self.stats["t5_reflections"] += 1
#         self.update_stats(self.stats["total_memories"])

# def load_and_split_dataset(test_size=0.1, seed=42):
#     if os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(TEST_SPLIT_PATH):
#         train_df = pd.read_csv(TRAIN_SPLIT_PATH)
#         test_df  = pd.read_csv(TEST_SPLIT_PATH)
#     else:
#         df = pd.read_csv(CLEAN_CSV_PATH)
#         assert "question_clean" in df and "answer_clean" in df, \
#             "CSV must have question_clean & answer_clean"
#         train_df, test_df = train_test_split(
#             df[["question_clean", "answer_clean"]],
#             test_size=test_size, random_state=seed, shuffle=True
#         )
#         train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
#         test_df.to_csv(TEST_SPLIT_PATH, index=False)
#     return train_df, test_df

# class QADataset(Dataset):
#     def __init__(self, df, tokenizer, max_input_length, max_target_length):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_input_length = max_input_length
#         self.max_target_length = max_target_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         question = str(self.df.loc[idx, "question_clean"]).strip()
#         answer = str(self.df.loc[idx, "answer_clean"]).strip()
        
#         input_text = f"answer: {question}"
    
#         target_text = answer
        
#         input_encoding = self.tokenizer(
#             input_text,
#             max_length=self.max_input_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         target_encoding = self.tokenizer(
#             target_text,
#             max_length=self.max_target_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         input_ids = input_encoding.input_ids.squeeze()
#         attention_mask = input_encoding.attention_mask.squeeze()
#         labels = target_encoding.input_ids.squeeze()
        
#         labels[labels == self.tokenizer.pad_token_id] = -100
        
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

# def get_latest_checkpoint(model_dir):
#     """Returns path to latest checkpoint folder, else fallback to model_dir"""
#     checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
#     if not checkpoints:
#         return model_dir
#     return sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]

# def get_model_and_tokenizer():
#     """Load T5 model and tokenizer with proper error handling from latest checkpoint"""
#     try:
#         model_path = get_latest_checkpoint(MODEL_DIR)
#         print(f" Loading model from: {model_path}")

#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

#         model = model.to(DEVICE)
#         print(f" Model loaded on {DEVICE}")
#         return model, tokenizer

#     except Exception as e:
#         print(f" Critical error loading model: {e}")
#         raise e

# def train_model(train_df, test_df, output_dir, epochs):
#     """Train T5 model for reflection and improvement"""
#     print(f" Training T5 model on CPU...")
    
#     model, tokenizer = get_model_and_tokenizer()
    
#     train_dataset = QADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     eval_dataset = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         do_train=True,
#         do_eval=True,
#         # evaluation_strategy="steps",
#         eval_steps=50,
#         save_steps=100,
#         save_total_limit=2,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=epochs,
#         logging_steps=10,
#         logging_dir=os.path.join(output_dir, "logs"),
#         learning_rate=5e-5,
#         warmup_steps=100,
#         weight_decay=0.01,
#         seed=42,
#         dataloader_pin_memory=False,  
#         no_cuda=True,  
#         fp16=False,  
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     print(f" Starting training on {DEVICE}")
#     print(f" Training examples: {len(train_dataset)}")
#     print(f" Validation examples: {len(eval_dataset)}")
    
#     trainer.train()
    
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(" T5 model training completed!")

# def retrain_on_feedback(model_dir, memory_system):
#     """Enhanced retraining with memory system"""
#     feedback_list = memory_system.load_feedback()
#     if not feedback_list:
#         print("No feedback found in memory.")
#         return
    
#     feedback_data = []
#     for entry in feedback_list:
#         feedback_data.append({
#             "question_clean": entry["question"],
#             "answer_clean": entry["improvement"]
#         })
    
#     fb_df = pd.DataFrame(feedback_data)
#     train_df, test_df = load_and_split_dataset()
    
#     combined_df = pd.concat([train_df, fb_df], ignore_index=True)
    
#     print(f" Retraining T5 model on {len(fb_df)} feedback items + {len(train_df)} original = {len(combined_df)} examples")
    
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_dir).to(DEVICE)
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
#     train_dataset = QADataset(combined_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     eval_dataset = QADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     training_args = TrainingArguments(
#         output_dir=model_dir,
#         overwrite_output_dir=False,
#         do_train=True,
#         do_eval=True,
#         # evaluation_strategy="steps",
#         eval_steps=50,
#         save_steps=100,
#         save_total_limit=2,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=RETRAIN_EPOCHS,
#         logging_steps=10,
#         learning_rate=3e-5,
#         warmup_steps=50,
#         weight_decay=0.01,
#         no_cuda=True,
#         fp16=False,
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     trainer.train()
#     trainer.save_model(model_dir)
#     tokenizer.save_pretrained(model_dir)
#     print(" T5 model retraining completed!")

# def analyze_answer_quality(self, question: str, answer: str, reflection: str) -> Dict:
#     """Enhanced analysis of answer quality with specific criteria"""
#     analysis = {
#         "completeness": 0.5,
#         "accuracy_confidence": 0.5,
#         "specificity": 0.5,
#         "clarity": 0.5,
#         "issues_found": [],
#         "improvement_suggestions": []
#     }
    
#     answer_lower = answer.lower()
#     reflection_lower = reflection.lower()
#     question_lower = question.lower()
    
#     if len(answer.split()) < 10:
#         analysis["completeness"] = 0.3
#         analysis["issues_found"].append("too_brief")
#     elif len(answer.split()) > 50:
#         analysis["completeness"] = 0.8
    
#     vague_phrases = ["might be", "could be", "possibly", "maybe", "not sure", "i think"]
#     if any(phrase in answer_lower for phrase in vague_phrases):
#         analysis["accuracy_confidence"] = 0.3
#         analysis["issues_found"].append("vague_language")
    
#     generic_responses = ["it depends", "various factors", "many ways", "different approaches"]
#     if any(phrase in answer_lower for phrase in generic_responses):
#         analysis["specificity"] = 0.3
#         analysis["issues_found"].append("too_generic")
    
#     negative_indicators = ["incomplete", "missing", "lacks", "unclear", "confusing", "wrong", "inaccurate"]
#     positive_indicators = ["comprehensive", "accurate", "clear", "complete", "detailed", "good"]
    
#     neg_count = sum(1 for word in negative_indicators if word in reflection_lower)
#     pos_count = sum(1 for word in positive_indicators if word in reflection_lower)
    
#     if neg_count > pos_count:
#         analysis["accuracy_confidence"] = max(0.2, analysis["accuracy_confidence"] - 0.3)
#         analysis["issues_found"].extend(["reflection_negative"])
    
#     return analysis

# def create_detailed_critique_prompt(self, question: str, answer: str, attempt_num: int = 1) -> str:
#     """Force T5 to pinpoint exactly what to improve."""

#     snippet = " ".join(answer.split()[:150]) + ("…" if len(answer.split())>150 else "")
#     return (
#         "critique: You are an expert. Identify any clarity, accuracy or completeness issues.\n"
#         f"Question: {question}\n"
#         f"Answer: {snippet}\n\n"
#         "List issues and what is missing. Be concise."
#     )

# def create_targeted_improvement_prompt(self, question: str, current_answer: str, reflection: str, attempt_num: int = 1) -> str:
#     """Force T5 to rewrite, not repeat."""
#     return (
#         "improve: Using the critique below, produce a substantially rewritten, clearer, more complete answer. Do NOT copy phrasing.\n"
#         f"Question: {question}\n"
#         f"Critique: {reflection}\n"
#         f"Original Answer: {current_answer}\n\n"
#         "Improved Answer:"
#     )

# def interactive_session(model_dir):
    
#     console = Console()

#     def format_bullets(text):
#         text = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", text)
#         text = re.sub(r"\n{2,}", "\n", text)
#         return text.strip()

#     def display_answer(title: str, answer: str):
#         formatted = format_bullets(answer)
#         console.print(Panel.fit(Markdown(f"### {title}\n\n{formatted}"), border_style="cyan", title="Reflective Answer"))

#     def safe_t5(prompt, **kwargs):
#         try:
#             return t5(prompt, **kwargs)[0]["generated_text"].strip()
#         except Exception:
#             short = " ".join(prompt.split()[:100])
#             return t5(short, **kwargs)[0]["generated_text"].strip()

#     try:
#         groq = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
#         if not groq.check_connection():
#             print(" GROQ API failed.")
#             return
#     except Exception as e:
#         print("", e)
#         return
#     print(" GROQ ready")

#     model, tokenizer = get_model_and_tokenizer()
#     device_id = 0 if torch.cuda.is_available() else -1
#     t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
#     print(f" T5 on {'GPU' if device_id >= 0 else 'CPU'}")

#     memory = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
#     feedback_list = memory.load_feedback()

#     print("\n=== Interactive Hybrid Reflective QA ===")
#     print("Commands: help | stats | exit")

#     while True:
#         q = input("\nQuestion: ").strip()
#         if not q: continue
#         if q.lower() in ("exit", "quit"): break
#         if q.lower() == "help":
#             print(" help  • show this\n stats • memory stats\n exit  • quit session")
#             continue
#         if q.lower() == "stats":
#             for k, v in memory.get_memory_stats().items():
#                 print(f" {k}: {v}")
#             continue

#         print("Generating initial answer with GROQ…")
#         t0 = time.time()
#         base = groq.generate_answer(q)
#         memory.increment_groq_usage()
#         console.print(Panel.fit(Markdown(f"### LLM Answer\n\n{format_bullets(base)}"), title="GROQ", border_style="magenta"))

#         existing = memory.find_similar_question(q, feedback_list)
#         if existing:
#             sim = memory.calculate_similarity(q, existing["question"])
#             console.print(Panel.fit(Markdown(f"### Memory Recall (sim={sim:.2f})\n\n{format_bullets(existing['improvement'])}"), border_style="green"))
#             continue

#         critique_prompt = (
#             "critique: List factual errors or omissions in bullet points.\n\n"
#             f"Q: {q}\nA: {base}"
#         )
#         print("Reflecting on LLM output…")
#         reflection = safe_t5(
#             critique_prompt,
#             max_new_tokens=128,
#             num_beams=2,
#             do_sample=False
#         )
#         memory.increment_t5_usage()
#         display_answer("T5 Reflection (Critique)", reflection)

#         improve_prompt = (
#             "improve: Use the critique below to rewrite the answer fully. "
#             "Fix all errors, add missing info, and do NOT copy the original phrasing.\n\n"
#             f"Q: {q}\n"
#             f"Critique:\n{reflection}\n\n"
#             f"Original Answer: {base}\n\n"
#             f"Corrected Answer:"
#         )
#         corrected = safe_t5(
#             improve_prompt,
#             max_new_tokens=200,
#             do_sample=True,
#             temperature=0.8,
#             top_p=0.9,
#             num_beams=1
#         )
#         display_answer("T5 Improved Answer", corrected)

#         choice = input(" Accept this answer? (Y/n): ").strip().lower()
#         if choice in ("", "y", "yes"):
#             final = corrected
#             entry = memory.auto_save_reflection(q, base, final, reflection)
#             print(" Saved to memory.")
#         else:
#             user_fix = input(" Your correction (or blank to keep T5 version):\n").strip()
#             if user_fix:
#                 final = user_fix
#                 entry = memory.save_user_correction(q, base, final, reflection)
#                 print(" User fix saved.")
#             else:
#                 final = corrected
#                 entry = memory.auto_save_reflection(q, base, final, reflection)
#                 print(" Saved default improvement.")

#         feedback_list.append(entry)
#         memory.save_feedback(feedback_list)

#     print(" Exiting. See you again!")



# def test_hybrid_system(model_dir, test_df):
#     """Test the hybrid GROQ-T5 system"""
#     print("\n Testing Hybrid GROQ-T5 System")
#     print("─" * 50)
    
#     try:
#         groq_handler = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
#         if not groq_handler.check_connection():
#             print(" GROQ API not available for testing")
#             return
#     except Exception as e:
#         print(f" GROQ initialization failed: {e}")
#         return
    
#     model, tokenizer = get_model_and_tokenizer()
#     t5_pipeline = pipeline(
#         "text2text-generation", 
#         model=model, 
#         tokenizer=tokenizer,
#         device=-1
#     )

#     for i, (_, row) in enumerate(test_df.sample(min(3,len(test_df))).iterrows(), 1):
#         question, gold = row["question_clean"], row["answer_clean"]
        
#         print(f"\n[Test {i}]")
#         print(f" Question: {question}")
        
#         groq_answer = groq_handler.generate_answer(question)
#         print(f" GROQ Answer: {groq_answer}")
        
#         critique_prompt = f"critique: Question: {question} Answer: {groq_answer}"
#         reflection = t5_pipeline(critique_prompt, max_new_tokens=128, num_beams=2)[0]["generated_text"].strip()
#         print(f" T5 Reflection: {reflection}")
        
#         print(f" Expected: {gold}")
#         print("─" * 30)

# def show_memory_overview():
#     """Show hybrid system memory overview"""
#     if os.path.exists(FEEDBACK_JSON):
#         memory_system = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
#         feedback_list = memory_system.load_feedback()
#         stats = memory_system.get_memory_stats()
        
#         print("\n Hybrid GROQ-T5 Memory System Overview")
#         print("─" * 50)
#         print(f" Total Memories: {len(feedback_list)}")
#         print(f" GROQ API Calls: {stats.get('groq_responses', 0)}")
#         print(f" T5 Reflections: {stats.get('t5_reflections', 0)}")
#         print(f" Auto-saved: {stats.get('auto_saved', 0)}")
#         print(f" User corrections: {stats.get('user_corrections', 0)}")
#         print(f" Memory reuses: {stats.get('reuse_count', 0)}")
#         print(f" Last updated: {stats.get('last_updated', 'Never')[:19]}")
        
#         if feedback_list:
#             print("\n Recent Hybrid Memories:")
#             for fb in feedback_list[-3:]:  
#                 print(f"  • {fb['question'][:50]}... ({fb['source']})")
#     else:
#         print(" No memory data found")

# def check_requirements():
#     """Check if all requirements are met"""
#     print("\n Checking System Requirements...")
#     print(f" Device: {DEVICE} (CPU-only mode)")
    
#     if not GROQ_API_KEY:
#         print(" GROQ_API_KEY environment variable not set")
#         print("   Please set it with: export GROQ_API_KEY=your_api_key")
#         return False
#     else:
#         print(" GROQ API key found")
    
#     try:
#         groq_handler = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
#         if groq_handler.check_connection():
#             print(" GROQ API connection successful")
#         else:
#             print(" GROQ API connection failed")
#             return False
#     except Exception as e:
#         print(f" GROQ API error: {e}")
#         return False
    
#     if not os.path.exists(CLEAN_CSV_PATH):
#         print(f" Missing required file: {CLEAN_CSV_PATH}")
#         return False
#     else:
#         print(" Training data file found")
    
#     return True

# def create_sample_data():
#     """Create sample data for testing if main data file is missing"""
#     print("\n Creating sample data for testing...")
    
#     sample_data = [
#         {"question_clean": "What is artificial intelligence?", 
#          "answer_clean": "Artificial intelligence (AI) is a branch of computer science that aims to create machines capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."},
#         {"question_clean": "How does machine learning work?", 
#          "answer_clean": "Machine learning works by training algorithms on data to recognize patterns and make predictions or decisions without being explicitly programmed for each specific task. It uses statistical techniques to improve performance through experience."},
#         {"question_clean": "What is deep learning?", 
#          "answer_clean": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (deep neural networks) to model and understand complex patterns in data, particularly useful for tasks like image recognition and natural language processing."},
#         {"question_clean": "What are neural networks?", 
#          "answer_clean": "Neural networks are computing systems inspired by biological neural networks. They consist of interconnected nodes (neurons) that process information through weighted connections, learning to recognize patterns and make predictions through training."},
#         {"question_clean": "What is natural language processing?", 
#          "answer_clean": "Natural Language Processing (NLP) is a field of AI that focuses on enabling computers to understand, interpret, and generate human language in a meaningful way, including tasks like translation, sentiment analysis, and text summarization."}
#     ]
    
#     os.makedirs(PREPROCESSED_DIR, exist_ok=True)
#     df = pd.DataFrame(sample_data)
#     df.to_csv(CLEAN_CSV_PATH, index=False)
#     print(f" Sample data created at {CLEAN_CSV_PATH}")

# class FixedQADataset(Dataset):
#     def __init__(self, df, tokenizer, max_input_length, max_target_length):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_input_length = max_input_length
#         self.max_target_length = max_target_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         question = str(self.df.loc[idx, "question_clean"]).strip()
#         answer = str(self.df.loc[idx, "answer_clean"]).strip()
        
#         input_text = f"Question: {question}"
#         target_text = answer
        
#         input_encoding = self.tokenizer(
#             input_text,
#             max_length=self.max_input_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         target_encoding = self.tokenizer(
#             target_text,
#             max_length=self.max_target_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
        
#         input_ids = input_encoding.input_ids.squeeze()
#         attention_mask = input_encoding.attention_mask.squeeze()
#         labels = target_encoding.input_ids.squeeze()
        
#         labels[labels == self.tokenizer.pad_token_id] = -100
        
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

# def fixed_train_model(train_df, test_df, output_dir, epochs):
#     """Fixed training function with proper loss calculation"""
#     print(f" Training T5 model on CPU with fixed dataset...")
    
#     model, tokenizer = get_model_and_tokenizer()
    
#     train_dataset = FixedQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     eval_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     training_args = TrainingArguments(
#         output_dir=output_dir,
#         overwrite_output_dir=True,
#         do_train=True,
#         do_eval=True,
#         # evaluation_strategy="steps",
#         eval_steps=50,
#         save_steps=100,
#         save_total_limit=2,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=epochs,
#         logging_steps=10,
#         logging_dir=os.path.join(output_dir, "logs"),
#         learning_rate=5e-5,
#         warmup_steps=100,
#         weight_decay=0.01,
#         seed=42,
#         use_cpu=True,  
#         dataloader_pin_memory=False,
#         fp16=False,
#         gradient_checkpointing=False,  
#         remove_unused_columns=False, 
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     print(f" Starting training on CPU")
#     print(f" Training examples: {len(train_dataset)}")
#     print(f" Validation examples: {len(eval_dataset)}")
    
#     print(" Checking first training example...")
#     sample = train_dataset[0]
#     print(f" Input IDs shape: {sample['input_ids'].shape}")
#     print(f" Labels shape: {sample['labels'].shape}")
#     print(f" Sample input: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
#     print(f" Sample target: {tokenizer.decode([id for id in sample['labels'] if id != -100], skip_special_tokens=True)}")
    
#     trainer.train()
    
#     trainer.save_model(output_dir)
#     tokenizer.save_pretrained(output_dir)
#     print(" T5 model training completed!")

# def fixed_retrain_on_feedback(model_dir, memory_system):
#     """Fixed retraining with proper dataset"""
#     feedback_list = memory_system.load_feedback()
#     if not feedback_list:
#         print("No feedback found in memory.")
#         return
    
#     feedback_data = []
#     for entry in feedback_list:
#         feedback_data.append({
#             "question_clean": entry["question"],
#             "answer_clean": entry["improvement"]
#         })
    
#     fb_df = pd.DataFrame(feedback_data)
#     train_df, test_df = load_and_split_dataset()
    
#     combined_df = pd.concat([train_df, fb_df], ignore_index=True)
    
#     print(f" Retraining T5 model on {len(fb_df)} feedback items + {len(train_df)} original = {len(combined_df)} examples")
    
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
#     tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
#     train_dataset = FixedQADataset(combined_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     eval_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     training_args = TrainingArguments(
#         output_dir=model_dir,
#         overwrite_output_dir=False,
#         do_train=True,
#         do_eval=True,
#         # evaluation_strategy="steps",
#         eval_steps=50,
#         save_steps=100,
#         save_total_limit=2,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         num_train_epochs=RETRAIN_EPOCHS,
#         logging_steps=10,
#         learning_rate=3e-5,
#         warmup_steps=50,
#         weight_decay=0.01,
#         use_cpu=True,
#         fp16=False,
#         remove_unused_columns=False,
#     )
    
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=eval_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     trainer.train()
#     trainer.save_model(model_dir)
#     tokenizer.save_pretrained(model_dir)
#     print(" T5 model retraining completed!")

# def main():
#     """Main function to run the hybrid system"""
#     print("=" * 60)
#     print("    HYBRID GROQ-T5 REFLECTIVE QA SYSTEM")
#     print("=" * 60)
    
#     if not check_requirements():
#         print("\n Creating sample data for testing...")
#         create_sample_data()
    
#     while True:
#         print("\n Select an option:")
#         print("1. Train T5 model from scratch")
#         print("2. Interactive QA session (GROQ + T5)")
#         print("3. Test hybrid system")
#         print("4. Show memory overview")
#         print("5. Retrain with feedback")
#         print("6. Exit")
        
#         choice = input("\nEnter choice (1-6): ").strip()
        
#         if choice == "1":
#             print("\n Starting T5 model training...")
#             train_df, test_df = load_and_split_dataset()
#             print(f"Training data: {len(train_df)} examples")
#             print(f"Test data: {len(test_df)} examples")
            
#             fixed_train_model(train_df, test_df, MODEL_DIR, NUM_TRAIN_EPOCHS)
            
#         elif choice == "2":
            
#             checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
#             model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
#             os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

#             if not (checkpoint_exists or model_file_exists):
#                 print(" No trained model found. Please train the model first (option 1).")
#                 continue

#             interactive_session(MODEL_DIR)
            
#         elif choice == "3":
            
#             checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
#             model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
#                         os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

#             if not (checkpoint_exists or model_file_exists):
#                 print(" No trained model found. Please train the model first (option 1).")
#                 continue

#             _, test_df = load_and_split_dataset()
#             test_hybrid_system(MODEL_DIR, test_df)

            
#         elif choice == "4":
#             show_memory_overview()
            
#         elif choice == "5":
#             checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
#             model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "pytorch_model.bin")) or \
#                         os.path.exists(os.path.join(MODEL_DIR, "model.safetensors"))

#             if not (checkpoint_exists or model_file_exists):
#                 print(" No trained model found. Please train the model first (option 1).")
#                 continue

#             memory_system = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
#             fixed_retrain_on_feedback(MODEL_DIR, memory_system)
            
#         elif choice == "6":
#             print(" Goodbye!")
#             break
            
#         else:
#             print(" Invalid choice. Please enter 1-6.")

# if __name__ == "__main__":
#     main()


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

PREPROCESSED_DIR   = "Dev/preprocessed_data"
CLEAN_CSV_PATH     = os.path.join(PREPROCESSED_DIR, "cleaned_qa_data.csv")
TRAIN_SPLIT_PATH   = os.path.join(PREPROCESSED_DIR, "train_split.csv")
TEST_SPLIT_PATH    = os.path.join(PREPROCESSED_DIR, "test_split.csv")

MODEL_DIR          = "Dev/flan_t5_reflective_model"
DATABASE_PATH      = os.path.join(MODEL_DIR, "feedback_system.db")
LEGACY_FEEDBACK_JSON = os.path.join(MODEL_DIR, "feedback.json") 
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
GROQ_MODEL = "openai/gpt-oss-20b"  
GROQ_MAX_TOKENS = 512
DEVICE = "cpu"

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

    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)

    print("\n=== Interactive Hybrid Reflective QA with Database ===")
    print("Commands: help | stats | cleanup | exit")

    while True:
        q = input("\nQuestion: ").strip()
        if not q: continue
        if q.lower() in ("exit", "quit"): break
        if q.lower() == "help":
            print(" help    • show this\n stats   • memory stats\n cleanup • clean old data\n exit    • quit session")
            continue
        if q.lower() == "stats":
            stats = memory.get_memory_stats()
            for k, v in stats.items():
                print(f" {k}: {v}")
            continue
        if q.lower() == "cleanup":
            days = input("Delete entries older than how many days? (default 90): ").strip()
            try:
                days = int(days) if days else 90
                deleted = memory.cleanup_old_data(days)
                print(f" Cleaned up {deleted} old entries")
            except ValueError:
                print(" Invalid number of days")
            continue

        print("Generating initial answer with GROQ…")
        t0 = time.time()
        base = groq.generate_answer(q)
        memory.increment_groq_usage()
        console.print(Panel.fit(Markdown(f"### LLM Answer\n\n{format_bullets(base)}"), title="GROQ", border_style="magenta"))

        existing = memory.find_similar_question(q)
        if existing:
            sim = memory.calculate_similarity(q, existing["question"])
            console.print(Panel.fit(Markdown(f"### Database Recall (sim={sim:.2f})\n\n{format_bullets(existing['improvement'])}"), border_style="green"))
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

        print(" Auto-saving reflection to database...")
        memory.auto_save_reflection(q, base, corrected, reflection)

        feedback = input("\nIs this improved answer satisfactory? (y/n/provide correction): ").strip().lower()
        
        if feedback.startswith('n') or (feedback not in ['y', 'yes', '']):
            if feedback in ['n', 'no']:
                user_correction = input("Please provide the correct answer: ").strip()
            else:
                user_correction = feedback
            
            if user_correction:
                
                user_critique_prompt = (
                    f"critique: Compare this user correction with the previous answer and "
                    f"explain what was improved.\n\n"
                    f"Q: {q}\n"
                    f"Previous Answer: {corrected}\n"
                    f"User Correction: {user_correction}"
                )
                
                user_reflection = safe_t5(
                    user_critique_prompt,
                    max_new_tokens=128,
                    num_beams=2,
                    do_sample=False
                )
                
                memory.save_user_correction(q, base, user_correction, user_reflection)
                display_answer("User Correction Saved", user_correction)
        
        elapsed = time.time() - t0
        print(f" Total time: {elapsed:.1f}s")

def train_model():
    """Train the T5 model with enhanced dataset loading"""
    print(" Starting T5 model training...")
    
    train_df, test_df = load_and_split_dataset()
    print(f" Dataset loaded: {len(train_df)} train, {len(test_df)} test samples")
    
    print(f" Loading pretrained model: {PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    
    special_tokens = {"additional_special_tokens": ["<critique>", "<improve>", "<reflect>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    train_dataset = FixedQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    test_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    
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
        evaluation_strategy="steps",
        eval_steps=200,
        save_steps=200,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(" Starting training...")
    trainer.train()
    
    print(" Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(MODEL_DIR)
    
    print("Training completed!")

def retrain_from_feedback():
    """Retrain model using feedback from database"""
    print(" Retraining model from database feedback...")
    
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    feedback_list = memory.load_feedback()
    
    if len(feedback_list) < 10:
        print(" Not enough feedback entries for retraining (minimum 10 required)")
        return False
    
    print(f" Using {len(feedback_list)} feedback entries for retraining")
    
    feedback_df = pd.DataFrame([
        {
            "question_clean": entry["question"],
            "answer_clean": entry["improvement"]
        }
        for entry in feedback_list
        if entry.get("confidence_score", 0) > 0.5  
    ])
    
    if len(feedback_df) == 0:
        print(" No high-confidence feedback entries found")
        return False
    
    print(f" Filtered to {len(feedback_df)} high-confidence entries")
    
    try:
        model, tokenizer = get_model_and_tokenizer()
    except Exception as e:
        print(f" Error loading existing model: {e}")
        return False
    
    feedback_dataset = FixedQADataset(
        feedback_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    retrain_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "retrained"),
        overwrite_output_dir=True,
        num_train_epochs=RETRAIN_EPOCHS,
        per_device_train_batch_size=max(1, BATCH_SIZE // 2),
        learning_rate=1e-5,  
        warmup_steps=50,
        weight_decay=0.01,
        logging_dir=os.path.join(MODEL_DIR, "retrain_logs"),
        logging_steps=25,
        save_steps=100,
        save_total_limit=2,
        report_to=None,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
    )
    
    trainer = Trainer(
        model=model,
        args=retrain_args,
        train_dataset=feedback_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print(" Starting retraining...")
    trainer.train()
    
    print(" Saving retrained model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    print(" Retraining completed!")
    return True

def export_feedback_data(export_path: str = None):
    """Export feedback data from database to various formats"""
    if not export_path:
        export_path = os.path.join(MODEL_DIR, "feedback_export")
    
    os.makedirs(export_path, exist_ok=True)
    
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    feedback_list = memory.load_feedback()
    
    if not feedback_list:
        print(" No feedback data to export")
        return
    
    df = pd.DataFrame(feedback_list)
    csv_path = os.path.join(export_path, "feedback_data.csv")
    df.to_csv(csv_path, index=False)
    print(f" Exported CSV: {csv_path}")
    
    json_path = os.path.join(export_path, "feedback_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)
    print(f" Exported JSON: {json_path}")
    
    stats = memory.get_memory_stats()
    stats_path = os.path.join(export_path, "system_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f" Exported stats: {stats_path}")
    
    print(f" Export completed to {export_path}")

def database_maintenance():
    """Perform database maintenance operations"""
    print(" Starting database maintenance...")
    
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    
    stats = memory.get_memory_stats()
    print(f" Current database stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print("\nAvailable maintenance operations:")
    print("1. Cleanup old entries (90+ days)")
    print("2. Vacuum database (optimize storage)")
    print("3. Export backup")
    print("4. View recent feedback")
    print("5. All of the above")
    
    choice = input("Select operation (1-5): ").strip()
    
    if choice in ['1', '5']:
        days = input("Delete entries older than how many days? (default 90): ").strip()
        try:
            days = int(days) if days else 90
            deleted = memory.cleanup_old_data(days)
            print(f" Cleaned up {deleted} old entries")
        except ValueError:
            print(" Invalid number of days")
    
    if choice in ['2', '5']:
  
        with memory.db_manager.get_connection() as conn:
            conn.execute('VACUUM')
            print(" Database vacuumed and optimized")
    
    if choice in ['3', '5']:
        
        backup_dir = os.path.join(MODEL_DIR, "backup", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        export_feedback_data(backup_dir)
        print(f" Backup created at {backup_dir}")
    
    if choice in ['4', '5']:
     
        feedback_list = memory.load_feedback()
        recent_feedback = feedback_list[:5] 
        
        console = Console()
        for i, entry in enumerate(recent_feedback, 1):
            console.print(f"\n[bold cyan]Recent Feedback #{i}[/bold cyan]")
            console.print(f"[green]Question:[/green] {entry['question'][:100]}...")
            console.print(f"[blue]Source:[/blue] {entry['source']}")
            console.print(f"[yellow]Confidence:[/yellow] {entry.get('confidence_score', 'N/A')}")
            console.print(f"[dim]Timestamp:[/dim] {entry['timestamp']}")
    
    print(" Database maintenance completed!")

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

def create_sample_data():
    """Create sample data for testing if main dataset is missing"""
    print(" Creating sample QA dataset...")
    
    sample_data = {
        'question_clean': [
            "What is the capital of France?",
            "How does photosynthesis work?",
            "What is machine learning?",
            "Explain the water cycle",
            "What are the benefits of exercise?",
            "How do computers work?",
            "What is climate change?",
            "Explain gravity",
            "What is DNA?",
            "How do vaccines work?"
        ],
        'answer_clean': [
            "The capital of France is Paris, located in the north-central part of the country.",
            "Photosynthesis is the process by which plants convert sunlight, carbon dioxide, and water into glucose and oxygen.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            "The water cycle is the continuous movement of water through evaporation, condensation, precipitation, and collection.",
            "Regular exercise improves cardiovascular health, strengthens muscles, boosts mental health, and helps maintain a healthy weight.",
            "Computers work by processing binary data through electronic circuits, following instructions stored in memory.",
            "Climate change refers to long-term shifts in global temperatures and weather patterns, primarily caused by human activities.",
            "Gravity is a fundamental force that attracts objects with mass toward each other, keeping us grounded on Earth.",
            "DNA is the hereditary material in living organisms that contains genetic instructions for development and function.",
            "Vaccines work by training the immune system to recognize and fight specific diseases without causing illness."
        ]
    }
    
    df = pd.DataFrame(sample_data)
    df.to_csv(CLEAN_CSV_PATH, index=False)
    print(f" Created sample dataset: {CLEAN_CSV_PATH}")
    return True


def show_memory_overview():
    """Display comprehensive memory and database statistics"""
    print("\n" + "="*60)
    print("    MEMORY & DATABASE OVERVIEW")
    print("="*60)
    
    try:
        memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
        stats = memory.get_memory_stats()
        
        if not stats:
            print(" No statistics available")
            return
        
        print(f"\n DATABASE STATISTICS")
        print("-" * 30)
        print(f"Total Feedback Entries: {stats.get('total_feedback', 0)}")
        print(f"Auto-saved Reflections: {stats.get('auto_saved', 0)}")
        print(f"User Corrections: {stats.get('user_corrections', 0)}")
        print(f"Recent Activity (7 days): {stats.get('recent_week', 0)}")
        print(f"Average Confidence: {stats.get('avg_confidence', 0.0):.3f}")
        
        print(f"\n SYSTEM USAGE")
        print("-" * 30)
        print(f"Memory Reuse Count: {stats.get('reuse_count', 0)}")
        print(f"GROQ API Calls: {stats.get('groq_responses', 0)}")
        print(f"T5 Reflections: {stats.get('t5_reflections', 0)}")
        
        print(f"\n LAST UPDATED")
        print("-" * 30)
        last_updated = stats.get('last_updated', 'Never')
        print(f"System Stats: {last_updated}")
        
        if os.path.exists(DATABASE_PATH):
            db_size = os.path.getsize(DATABASE_PATH) / 1024  # KB
            print(f"Database Size: {db_size:.1f} KB")
        
    except Exception as e:
        print(f" Error accessing memory system: {e}")


def test_hybrid_system(model_dir, test_df):
    """Test the hybrid system with sample questions"""
    print("\n" + "="*60)
    print("    TESTING HYBRID SYSTEM")
    print("="*60)
    
    try:
        groq = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq.check_connection():
            print(" GROQ API connection failed")
            return
        
        model, tokenizer = get_model_and_tokenizer()
        device_id = 0 if torch.cuda.is_available() else -1
        t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
        
        memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
        
        test_questions = test_df.head(3)['question_clean'].tolist()
        
        console = Console()
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\n[bold cyan]Test {i}/3: {question}[/bold cyan]")
            
            existing = memory.find_similar_question(question)
            if existing:
                console.print(f"[green]✓ Found in memory[/green]")
                continue
            
            
            print("Generating GROQ answer...")
            groq_answer = groq.generate_answer(question)
            memory.increment_groq_usage()
            
            print("Generating T5 reflection...")
            critique_prompt = f"critique: List factual errors or omissions.\n\nQ: {question}\nA: {groq_answer}"
            reflection = t5(critique_prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            memory.increment_t5_usage()
            
            memory.auto_save_reflection(question, groq_answer, groq_answer, reflection)
            
            console.print(f"[green]✓ Test {i} completed and saved[/green]")
        
        print("\n Hybrid system test completed successfully!")
        
    except Exception as e:
        print(f" Error during hybrid system test: {e}")

def main():
    """Main function with enhanced menu interface"""
    print("=" * 60)
    print("    HYBRID GROQ-T5 REFLECTIVE QA SYSTEM")
    print("    Enhanced with SQLite Database & Memory")
    print("=" * 60)
    
    if not check_requirements():
        response = input("\n Missing required files. Create sample data for testing? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            create_sample_data()
        else:
            print(" Cannot continue without required files.")
            return
    
    while True:
        print("\n" + "="*40)
        print("         SELECT AN OPTION")
        print("="*40)
        print("1. Train T5 model from scratch")
        print("2. Interactive QA session (GROQ + T5)")
        print("3. Test hybrid system")
        print("4. Show memory overview")
        print("5. Retrain with feedback")
        print("6. Database maintenance")
        print("7. Export feedback data")
        print("8. System statistics")
        print("9. Exit")
        print("="*40)
        
        choice = input("\nEnter choice (1-9): ").strip()
        
        if choice == "1":
            print("\n Starting T5 model training...")
            try:
                train_df, test_df = load_and_split_dataset()
                print(f" Training data: {len(train_df)} examples")
                print(f" Test data: {len(test_df)} examples")
                
                train_model()
                
            except Exception as e:
                print(f" Training failed: {e}")
        
        elif choice == "2":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n Starting interactive QA session...")
            interactive_session(MODEL_DIR)
        
        elif choice == "3":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n Testing hybrid system...")
            try:
                _, test_df = load_and_split_dataset()
                test_hybrid_system(MODEL_DIR, test_df)
            except Exception as e:
                print(f" Test failed: {e}")
        
        elif choice == "4":
            show_memory_overview()
        
        elif choice == "5":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print(" No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n Retraining with feedback...")
            try:
                success = retrain_from_feedback()
                if success:
                    print(" Retraining completed successfully!")
                else:
                    print(" Retraining failed or insufficient feedback data.")
            except Exception as e:
                print(f" Retraining failed: {e}")
        
        elif choice == "6":
            print("\n Database maintenance...")
            try:
                database_maintenance()
            except Exception as e:
                print(f" Maintenance failed: {e}")
        
        elif choice == "7":
            print("\n Exporting feedback data...")
            export_path = input("Enter export path (press Enter for default): ").strip()
            if not export_path:
                export_path = None
            
            try:
                export_feedback_data(export_path)
            except Exception as e:
                print(f" Export failed: {e}")
        
        elif choice == "8":
            print("\n System Statistics")
            print("-" * 30)
            try:
                memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
                stats = memory.get_memory_stats()
                
                for key, value in stats.items():
                    print(f"{key}: {value}")
                    
                print(f"\nDatabase Path: {DATABASE_PATH}")
                print(f"Model Directory: {MODEL_DIR}")
                print(f"Device: {DEVICE}")
                print(f"GROQ Model: {GROQ_MODEL}")
                
            except Exception as e:
                print(f" Error retrieving stats: {e}")
        
        elif choice == "9":
            print("\n Goodbye!")
            print("Thank you for using the Hybrid QA System!")
            break
        
        else:
            print(" Invalid choice. Please enter 1-9.")
    
    print("\n" + "="*60)
    print("    SESSION ENDED")
    print("="*60)

if __name__ == "__main__":
    main()



