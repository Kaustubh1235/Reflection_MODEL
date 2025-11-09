import os
import json
import glob
import pandas as pd
import torch
import torch, time, re
import sqlite3
import re
from typing import Set, List, Dict, Optional, Tuple
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
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

# Configuration
PREPROCESSED_DIR   = "./preprocessed_data"
CLEAN_CSV_PATH     = os.path.join(PREPROCESSED_DIR, "cleaned_qa_data.csv")
TRAIN_SPLIT_PATH   = os.path.join(PREPROCESSED_DIR, "train_split.csv")
TEST_SPLIT_PATH    = os.path.join(PREPROCESSED_DIR, "test_split.csv")

MODEL_DIR          = "./flan_t5_reflective_model"
DATABASE_PATH      = os.path.join(MODEL_DIR, "feedback_system.db")
LEGACY_FEEDBACK_JSON = os.path.join(MODEL_DIR, "feedback.json")  # For migration
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
GROQ_MODEL = "Llama3-70B-8192"  
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
            
            # Feedback table
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
            
            # Statistics table
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
            
            # Initialize stats if empty
            cursor.execute('SELECT COUNT(*) FROM system_stats')
            if cursor.fetchone()[0] == 0:
                cursor.execute('''
                    INSERT INTO system_stats (id, last_updated) 
                    VALUES (1, datetime('now'))
                ''')
            
            # Create indexes for better performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_question_hash ON feedback(question_hash)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_source ON feedback(source)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON feedback(timestamp)')
            
            conn.commit()
            print("✓ Database initialized successfully")
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
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
            
            # Backup and remove old JSON file
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
                # Get recent entries for similarity comparison
                cursor.execute('''
                    SELECT * FROM feedback 
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit * 5,))  # Get more entries for better similarity matching
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
                
                # Basic counts
                cursor.execute('SELECT COUNT(*) FROM feedback')
                total_feedback = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE source = "auto-reflection"')
                auto_saved = cursor.fetchone()[0]
                
                cursor.execute('SELECT COUNT(*) FROM feedback WHERE source = "user-correction"')
                user_corrections = cursor.fetchone()[0]
                
                # Recent activity
                cursor.execute('''
                    SELECT COUNT(*) FROM feedback 
                    WHERE created_at >= datetime('now', '-7 days')
                ''')
                recent_week = cursor.fetchone()[0]
                
                # Average confidence
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
                
                # Build dynamic update query
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
                    # Add feedback stats
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

class DomainValidator:
    """Domain validation for cybersecurity and educational content"""
    
    def __init__(self):
        # Cybersecurity keywords
        self.cybersecurity_keywords = {
            # Core security concepts
            'cybersecurity', 'information security', 'infosec', 'network security', 
            'data security', 'cyber defense', 'cyber attack', 'cyber threat',
            
            # Attack types
            'malware', 'virus', 'trojan', 'worm', 'ransomware', 'phishing', 
            'spoofing', 'ddos', 'dos attack', 'sql injection', 'xss', 
            'cross-site scripting', 'buffer overflow', 'zero-day', 'exploit',
            'social engineering', 'man-in-the-middle', 'mitm', 'brute force',
            
            # Security tools & technologies
            'firewall', 'antivirus', 'ids', 'ips', 'intrusion detection',
            'intrusion prevention', 'siem', 'vulnerability scanner', 'penetration testing',
            'ethical hacking', 'red team', 'blue team', 'threat hunting',
            
            # Cryptography
            'encryption', 'decryption', 'cryptography', 'hash', 'digital signature',
            'pki', 'ssl', 'tls', 'vpn', 'certificate', 'key management',
            'symmetric encryption', 'asymmetric encryption', 'aes', 'rsa',
            
            # Authentication & Access Control
            'authentication', 'authorization', 'access control', 'identity management',
            'multi-factor authentication', 'mfa', '2fa', 'single sign-on', 'sso',
            'rbac', 'role-based access control', 'privilege escalation',
            
            # Compliance & Governance
            'gdpr', 'hipaa', 'sox', 'pci dss', 'iso 27001', 'nist', 'compliance',
            'risk assessment', 'security audit', 'security policy', 'incident response',
            
            # Network Security
            'network security', 'secure protocols', 'https', 'ssh', 'sftp',
            'network monitoring', 'packet analysis', 'wireshark', 'nmap',
            
            # Data Protection
            'data loss prevention', 'dlp', 'backup security', 'data breach',
            'privacy protection', 'anonymization', 'pseudonymization'
        }
        
        # Educational keywords
        self.educational_keywords = {
            # Learning concepts
            'learning', 'education', 'teaching', 'training', 'course', 'curriculum',
            'pedagogy', 'educational technology', 'e-learning', 'online learning',
            'distance learning', 'blended learning', 'instructional design',
            
            # Academic terms
            'student', 'teacher', 'instructor', 'professor', 'academic', 'university',
            'college', 'school', 'classroom', 'lesson', 'assignment', 'homework',
            'exam', 'test', 'quiz', 'grade', 'assessment', 'evaluation',
            
            # Educational methods
            'tutorial', 'workshop', 'seminar', 'lecture', 'discussion', 'study',
            'research', 'thesis', 'dissertation', 'project-based learning',
            'collaborative learning', 'active learning', 'flipped classroom',
            
            # Educational technology
            'lms', 'learning management system', 'moodle', 'blackboard', 'canvas',
            'educational software', 'simulation', 'virtual classroom', 'mooc',
            'educational app', 'digital literacy', 'computer literacy',
            
            # Cybersecurity education specific
            'cybersecurity training', 'security awareness', 'security education',
            'cyber literacy', 'digital citizenship', 'online safety', 'internet safety'
        }
        
        # Combined domain keywords
        self.domain_keywords = self.cybersecurity_keywords.union(self.educational_keywords)
        
        # Question patterns that indicate educational intent
        self.educational_patterns = [
            r'\bhow to learn\b', r'\bexplain\b', r'\bwhat is\b', r'\bdefine\b',
            r'\btutorial\b', r'\bteach me\b', r'\blearn about\b', r'\bunderstand\b',
            r'\bstep by step\b', r'\bguide\b', r'\bexample\b'
        ]
        
        # Initialize stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback if NLTK data not available
            self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    def extract_keywords(self, text: str) -> set[str]:
        """Extract meaningful keywords from text"""
        # Convert to lowercase and tokenize
        text_lower = text.lower()
        
        # Simple tokenization if NLTK not available
        try:
            tokens = word_tokenize(text_lower)
        except LookupError:
            tokens = re.findall(r'\b\w+\b', text_lower)
        
        # Remove stopwords and short words
        keywords = {token for token in tokens 
                   if token not in self.stop_words and len(token) > 2}
        
        # Also check for multi-word phrases
        for keyword in self.domain_keywords:
            if keyword in text_lower:
                keywords.add(keyword)
        
        return keywords
    
    def calculate_domain_relevance(self, question: str) -> Tuple[float, str]:
        """Calculate relevance score for cybersecurity/education domains"""
        question_lower = question.lower()
        question_keywords = self.extract_keywords(question)
        
        # Check for direct keyword matches
        cyber_matches = len(question_keywords.intersection(self.cybersecurity_keywords))
        edu_matches = len(question_keywords.intersection(self.educational_keywords))
        
        # Check for educational question patterns
        pattern_matches = sum(1 for pattern in self.educational_patterns 
                            if re.search(pattern, question_lower))
        
        # Calculate scores
        total_keywords = len(question_keywords)
        if total_keywords == 0:
            return 0.0, "unknown"
        
        cyber_score = cyber_matches / max(total_keywords, 1)
        edu_score = (edu_matches + pattern_matches) / max(total_keywords, 1)
        
        # Determine dominant domain
        if cyber_score > edu_score and cyber_score > 0.1:
            return cyber_score, "cybersecurity"
        elif edu_score > 0.1:
            return edu_score, "education"
        else:
            # Check if it's a general question that could be educational
            if any(pattern in question_lower for pattern in ['what', 'how', 'why', 'explain']):
                return 0.3, "general-educational"
            return 0.0, "out-of-domain"
    
    def is_domain_relevant(self, question: str, threshold: float = 0.2) -> Tuple[bool, str, float]:
        """Check if question is relevant to cybersecurity or education"""
        score, domain = self.calculate_domain_relevance(question)
        is_relevant = score >= threshold or domain in ["cybersecurity", "education", "general-educational"]
        return is_relevant, domain, score

class DomainRestrictedGroqHandler(GroqLLMHandler):
    """Enhanced GROQ handler with domain restrictions"""
    
    def __init__(self, api_key: str, model: str = GROQ_MODEL):
        super().__init__(api_key, model)
        self.domain_validator = DomainValidator()
    
    def generate_domain_specific_answer(self, question: str, domain: str, max_tokens: int = GROQ_MAX_TOKENS) -> str:
        """Generate answer with domain-specific system prompt"""
        
        # Domain-specific system prompts
        if domain == "cybersecurity":
            system_prompt = """You are a cybersecurity expert and educator. Provide accurate, detailed answers about cybersecurity topics including:
- Network security, malware, and cyber threats
- Security tools, technologies, and best practices  
- Cryptography, authentication, and access control
- Incident response and risk management
- Compliance and security frameworks

Focus on practical, educational content. If asked about non-cybersecurity topics, politely redirect to cybersecurity-related aspects."""

        elif domain in ["education", "general-educational"]:
            system_prompt = """You are an educational expert specializing in cybersecurity and technology education. Provide clear, educational answers that:
- Explain concepts step-by-step
- Use examples and analogies for better understanding
- Focus on learning objectives and practical application
- Encourage further learning and exploration

If the question is not related to education or cybersecurity, politely explain that you specialize in these domains."""

        else:
            system_prompt = """You are a specialized assistant focused on cybersecurity and educational topics. 
I can only provide information about cybersecurity concepts, tools, practices, and educational methods. 
For questions outside these domains, I'll politely explain my limitations."""

        try:
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
    
    def generate_out_of_domain_response(self, question: str) -> str:
        """Generate polite response for out-of-domain questions"""
        return """I'm specialized in cybersecurity and educational topics. I can help you with:

🔒 **Cybersecurity**: Network security, malware analysis, ethical hacking, cryptography, incident response, security frameworks, and best practices.

📚 **Education**: Learning methodologies, cybersecurity training, educational technology, curriculum development, and instructional design.

Could you please rephrase your question to focus on cybersecurity or educational aspects? I'd be happy to help with topics in these domains!"""


# Modified interactive session with domain validation
def domain_restricted_interactive_session(model_dir):
    """Interactive session with domain validation"""
    console = Console()
    
    def format_bullets(text):
        text = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", text)
        text = re.sub(r"\n{2,}", "\n", text)
        return text.strip()

    def display_answer(title: str, answer: str, border_color: str = "cyan"):
        formatted = format_bullets(answer)
        console.print(Panel.fit(Markdown(f"### {title}\n\n{formatted}"), 
                              border_style=border_color, title="Domain-Restricted Answer"))

    def safe_t5(prompt, **kwargs):
        try:
            return t5(prompt, **kwargs)[0]["generated_text"].strip()
        except Exception:
            short = " ".join(prompt.split()[:100])
            return t5(short, **kwargs)[0]["generated_text"].strip()

    # Initialize domain-restricted GROQ handler
    try:
        groq = DomainRestrictedGroqHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq.check_connection():
            print("❌ GROQ API failed.")
            return
    except Exception as e:
        print(f"❌ GROQ initialization failed: {e}")
        return
    print("✅ Domain-restricted GROQ ready")

    # Load T5 model
    model, tokenizer = get_model_and_tokenizer()
    device_id = 0 if torch.cuda.is_available() else -1
    t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
    print(f"✅ T5 model loaded on {'GPU' if device_id >= 0 else 'CPU'}")

    # Initialize memory system
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)

    print("\n" + "="*70)
    print("    DOMAIN-RESTRICTED CYBERSECURITY & EDUCATION QA SYSTEM")
    print("="*70)
    print("🔒 Specialized in: Cybersecurity, Information Security, Education")
    print("📚 Commands: help | stats | domains | cleanup | exit")

    while True:
        q = input("\n🔍 Question: ").strip()
        if not q: 
            continue
            
        if q.lower() in ("exit", "quit"): 
            break
            
        if q.lower() == "help":
            console.print("""
[bold cyan]Available Commands:[/bold cyan]
help     • Show this help
stats    • Memory and usage statistics  
domains  • Show supported domains and examples
cleanup  • Clean old database entries
exit     • Quit session

[bold green]Supported Topics:[/bold green]
🔒 Cybersecurity: Malware, network security, encryption, ethical hacking
📚 Education: Learning methods, cybersecurity training, educational technology
            """)
            continue
            
        if q.lower() == "stats":
            stats = memory.get_memory_stats()
            console.print("\n[bold cyan]📊 System Statistics[/bold cyan]")
            for k, v in stats.items():
                console.print(f"{k}: {v}")
            continue
            
        if q.lower() == "domains":
            console.print("""
[bold green]🔒 Cybersecurity Topics I Can Help With:[/bold green]
• Network security and firewalls
• Malware analysis and prevention  
• Ethical hacking and penetration testing
• Cryptography and encryption
• Incident response and forensics
• Security frameworks (NIST, ISO 27001)
• Risk assessment and management

[bold blue]📚 Educational Topics I Can Help With:[/bold blue]
• Cybersecurity training and awareness
• Learning methodologies and pedagogy
• Educational technology and e-learning
• Curriculum development
• Student assessment and evaluation

[bold yellow]Example Questions:[/bold yellow]
• "How does AES encryption work?"
• "What is a SQL injection attack?"
• "How to design effective cybersecurity training?"
• "Explain the NIST cybersecurity framework"
            """)
            continue
            
        if q.lower() == "cleanup":
            days = input("Delete entries older than how many days? (default 90): ").strip()
            try:
                days = int(days) if days else 90
                deleted = memory.cleanup_old_data(days)
                print(f"🗑️ Cleaned up {deleted} old entries")
            except ValueError:
                print("❌ Invalid number of days")
            continue

        # Domain validation
        print("🔍 Validating domain relevance...")
        is_relevant, domain, score = groq.domain_validator.is_domain_relevant(q)
        
        console.print(f"[dim]Domain: {domain} (relevance: {score:.2f})[/dim]")
        
        if not is_relevant:
            console.print(Panel.fit(
                Markdown(groq.generate_out_of_domain_response(q)),
                border_style="red",
                title="❌ Out of Domain"
            ))
            continue

        # Check memory for similar questions
        existing = memory.find_similar_question(q)
        if existing:
            sim = memory.calculate_similarity(q, existing["question"])
            display_answer(f"📚 Retrieved from Memory (similarity: {sim:.2f})", 
                         existing['improvement'], "green")
            continue

        # Generate domain-specific answer
        print("🤖 Generating domain-specific answer...")
        t0 = time.time()
        
        base_answer = groq.generate_domain_specific_answer(q, domain)
        memory.increment_groq_usage()
        
        display_answer(f"🔒 {domain.title()} Answer", base_answer, "magenta")

        # Domain-specific reflection prompt
        if domain == "cybersecurity":
            critique_prompt = f"""critique: As a cybersecurity expert, review this answer for:
- Technical accuracy of security concepts
- Missing security considerations or risks
- Incomplete threat analysis
- Outdated security practices

Question: {q}
Answer: {base_answer}"""
        else:
            critique_prompt = f"""critique: As an education expert, review this answer for:
- Clarity and educational value
- Missing learning objectives
- Need for examples or analogies  
- Pedagogical effectiveness

Question: {q}
Answer: {base_answer}"""

        print("🧠 Generating expert reflection...")
        reflection = safe_t5(
            critique_prompt,
            max_new_tokens=150,
            num_beams=2,
            do_sample=False
        )
        memory.increment_t5_usage()
        display_answer("🎯 Expert Reflection", reflection, "yellow")

        # Domain-specific improvement prompt
        improve_prompt = f"""improve: As a {domain} expert, enhance this answer by addressing the critique.
Make it more comprehensive, accurate, and educational.

Question: {q}
Critique: {reflection}
Original Answer: {base_answer}

Enhanced Answer:"""

        enhanced_answer = safe_t5(
            improve_prompt,
            max_new_tokens=250,
            do_sample=True,
            temperature=0.8,
            top_p=0.9
        )
        
        display_answer("✅ Enhanced Expert Answer", enhanced_answer, "green")

        # Auto-save with domain information
        print("💾 Saving to domain-specific memory...")
        entry = memory.auto_save_reflection(q, base_answer, enhanced_answer, reflection)
        entry['domain'] = domain
        entry['relevance_score'] = score

        # User feedback
        feedback = input("\n👍 Is this enhanced answer satisfactory? (y/n/provide correction): ").strip().lower()
        
        if feedback.startswith('n') or (feedback not in ['y', 'yes', '']):
            if feedback in ['n', 'no']:
                user_correction = input("Please provide the correct answer: ").strip()
            else:
                user_correction = feedback
            
            if user_correction:
                user_reflection = safe_t5(
                    f"critique: Compare this expert correction with the previous answer.\n\n"
                    f"Question: {q}\nPrevious: {enhanced_answer}\nCorrection: {user_correction}",
                    max_new_tokens=128
                )
                memory.save_user_correction(q, base_answer, user_correction, user_reflection)
                display_answer("💡 Expert Correction Saved", user_correction, "blue")
        
        elapsed = time.time() - t0
        console.print(f"[dim]⏱️ Total time: {elapsed:.1f}s | Domain: {domain}[/dim]")





class EnhancedMemorySystem:
    def __init__(self, db_path: str, stats_json_path: str, similarity_threshold: float = 0.7):
        self.db_manager = DatabaseManager(db_path)
        self.stats_path = stats_json_path
        self.similarity_threshold = similarity_threshold
        
        # Migrate from JSON if exists
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
                
                # Update database stats
                self.db_manager.update_system_stats(legacy_stats)
                
                # Backup legacy stats
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
        # This method is kept for compatibility but not recommended
        # Use save_single_feedback instead
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
            # Update reuse count
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
            # Update stats
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
            # Update stats
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

# [Rest of the code remains the same - training functions, dataset classes, etc.]
# I'll include the key modified functions below:

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

# Updated interactive session with database integration
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

    # Initialize with database
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

        # Check database for similar questions
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
        # Continuing from where the code left off - display_answer function and beyond

        display_answer("T5 Improved Answer", corrected)

        # Auto-save the reflection to database
        print("💾 Auto-saving reflection to database...")
        memory.auto_save_reflection(q, base, corrected, reflection)

        # Ask for user feedback
        feedback = input("\nIs this improved answer satisfactory? (y/n/provide correction): ").strip().lower()
        
        if feedback.startswith('n') or (feedback not in ['y', 'yes', '']):
            if feedback in ['n', 'no']:
                user_correction = input("Please provide the correct answer: ").strip()
            else:
                user_correction = feedback
            
            if user_correction:
                # Generate reflection on user correction
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
        print(f"⏱️ Total time: {elapsed:.1f}s")

def train_model():
    """Train the T5 model with enhanced dataset loading"""
    print("🚀 Starting T5 model training...")
    
    # Load dataset
    train_df, test_df = load_and_split_dataset()
    print(f"📊 Dataset loaded: {len(train_df)} train, {len(test_df)} test samples")
    
    # Initialize tokenizer and model
    print(f"🔧 Loading pretrained model: {PRETRAINED_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_MODEL)
    
    # Add special tokens if needed
    special_tokens = {"additional_special_tokens": ["<critique>", "<improve>", "<reflect>"]}
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    # Create datasets
    train_dataset = FixedQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    test_dataset = FixedQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Training arguments
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
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🔥 Starting training...")
    trainer.train()
    
    print("💾 Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(MODEL_DIR)
    
    print("✅ Training completed!")

def retrain_from_feedback():
    """Retrain model using feedback from database"""
    print("🔄 Retraining model from database feedback...")
    
    # Initialize database memory system
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    feedback_list = memory.load_feedback()
    
    if len(feedback_list) < 10:
        print("❌ Not enough feedback entries for retraining (minimum 10 required)")
        return False
    
    print(f"📊 Using {len(feedback_list)} feedback entries for retraining")
    
    # Convert feedback to training format
    feedback_df = pd.DataFrame([
        {
            "question_clean": entry["question"],
            "answer_clean": entry["improvement"]
        }
        for entry in feedback_list
        if entry.get("confidence_score", 0) > 0.5  # Only use high-confidence feedback
    ])
    
    if len(feedback_df) == 0:
        print("❌ No high-confidence feedback entries found")
        return False
    
    print(f"📈 Filtered to {len(feedback_df)} high-confidence entries")
    
    # Load existing model
    try:
        model, tokenizer = get_model_and_tokenizer()
    except Exception as e:
        print(f"❌ Error loading existing model: {e}")
        return False
    
    # Create feedback dataset
    feedback_dataset = FixedQADataset(
        feedback_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )
    
    # Retraining arguments (fewer epochs, lower learning rate)
    retrain_args = TrainingArguments(
        output_dir=os.path.join(MODEL_DIR, "retrained"),
        overwrite_output_dir=True,
        num_train_epochs=RETRAIN_EPOCHS,
        per_device_train_batch_size=max(1, BATCH_SIZE // 2),
        learning_rate=1e-5,  # Lower learning rate for fine-tuning
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
    
    # Initialize trainer for retraining
    trainer = Trainer(
        model=model,
        args=retrain_args,
        train_dataset=feedback_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("🔥 Starting retraining...")
    trainer.train()
    
    # Save retrained model back to main directory
    print("💾 Saving retrained model...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    
    print("✅ Retraining completed!")
    return True

def export_feedback_data(export_path: str = None):
    """Export feedback data from database to various formats"""
    if not export_path:
        export_path = os.path.join(MODEL_DIR, "feedback_export")
    
    os.makedirs(export_path, exist_ok=True)
    
    # Initialize database
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    feedback_list = memory.load_feedback()
    
    if not feedback_list:
        print("❌ No feedback data to export")
        return
    
    # Export as CSV
    df = pd.DataFrame(feedback_list)
    csv_path = os.path.join(export_path, "feedback_data.csv")
    df.to_csv(csv_path, index=False)
    print(f"📄 Exported CSV: {csv_path}")
    
    # Export as JSON (formatted)
    json_path = os.path.join(export_path, "feedback_data.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(feedback_list, f, indent=2, ensure_ascii=False)
    print(f"📄 Exported JSON: {json_path}")
    
    # Export statistics
    stats = memory.get_memory_stats()
    stats_path = os.path.join(export_path, "system_stats.json")
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    print(f"📊 Exported stats: {stats_path}")
    
    print(f"✅ Export completed to {export_path}")

def database_maintenance():
    """Perform database maintenance operations"""
    print("🔧 Starting database maintenance...")
    
    memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
    
    # Get current stats
    stats = memory.get_memory_stats()
    print(f"📊 Current database stats:")
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # Ask user for maintenance operations
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
            print(f"🗑️ Cleaned up {deleted} old entries")
        except ValueError:
            print("❌ Invalid number of days")
    
    if choice in ['2', '5']:
        # Vacuum database
        with memory.db_manager.get_connection() as conn:
            conn.execute('VACUUM')
            print("🗜️ Database vacuumed and optimized")
    
    if choice in ['3', '5']:
        # Export backup
        backup_dir = os.path.join(MODEL_DIR, "backup", pd.Timestamp.now().strftime("%Y%m%d_%H%M%S"))
        export_feedback_data(backup_dir)
        print(f"💾 Backup created at {backup_dir}")
    
    if choice in ['4', '5']:
        # Show recent feedback
        feedback_list = memory.load_feedback()
        recent_feedback = feedback_list[:5]  # Show last 5 entries
        
        console = Console()
        for i, entry in enumerate(recent_feedback, 1):
            console.print(f"\n[bold cyan]Recent Feedback #{i}[/bold cyan]")
            console.print(f"[green]Question:[/green] {entry['question'][:100]}...")
            console.print(f"[blue]Source:[/blue] {entry['source']}")
            console.print(f"[yellow]Confidence:[/yellow] {entry.get('confidence_score', 'N/A')}")
            console.print(f"[dim]Timestamp:[/dim] {entry['timestamp']}")
    
    print("✅ Database maintenance completed!")

def check_requirements():
    """Check if required files and directories exist"""
    required_files = [CLEAN_CSV_PATH]
    required_dirs = [PREPROCESSED_DIR, MODEL_DIR]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    missing_dirs = [d for d in required_dirs if not os.path.exists(d)]
    
    if missing_dirs:
        for dir_path in missing_dirs:
            os.makedirs(dir_path, exist_ok=True)
        print(f"✓ Created missing directories: {missing_dirs}")
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
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
    print(f"✓ Created sample dataset: {CLEAN_CSV_PATH}")
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
            print("❌ No statistics available")
            return
        
        print(f"\n📊 DATABASE STATISTICS")
        print("-" * 30)
        print(f"Total Feedback Entries: {stats.get('total_feedback', 0)}")
        print(f"Auto-saved Reflections: {stats.get('auto_saved', 0)}")
        print(f"User Corrections: {stats.get('user_corrections', 0)}")
        print(f"Recent Activity (7 days): {stats.get('recent_week', 0)}")
        print(f"Average Confidence: {stats.get('avg_confidence', 0.0):.3f}")
        
        print(f"\n🔄 SYSTEM USAGE")
        print("-" * 30)
        print(f"Memory Reuse Count: {stats.get('reuse_count', 0)}")
        print(f"GROQ API Calls: {stats.get('groq_responses', 0)}")
        print(f"T5 Reflections: {stats.get('t5_reflections', 0)}")
        
        print(f"\n🕐 LAST UPDATED")
        print("-" * 30)
        last_updated = stats.get('last_updated', 'Never')
        print(f"System Stats: {last_updated}")
        
        # Database file info
        if os.path.exists(DATABASE_PATH):
            db_size = os.path.getsize(DATABASE_PATH) / 1024  # KB
            print(f"Database Size: {db_size:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error accessing memory system: {e}")


def test_hybrid_system(model_dir, test_df):
    """Test the hybrid system with sample questions"""
    print("\n" + "="*60)
    print("    TESTING HYBRID SYSTEM")
    print("="*60)
    
    try:
        # Initialize GROQ
        groq = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
        if not groq.check_connection():
            print("❌ GROQ API connection failed")
            return
        
        # Load T5 model
        model, tokenizer = get_model_and_tokenizer()
        device_id = 0 if torch.cuda.is_available() else -1
        t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=device_id)
        
        # Initialize memory system
        memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
        
        # Test with a few sample questions
        test_questions = test_df.head(3)['question_clean'].tolist()
        
        console = Console()
        
        for i, question in enumerate(test_questions, 1):
            console.print(f"\n[bold cyan]Test {i}/3: {question}[/bold cyan]")
            
            # Check for existing memory
            existing = memory.find_similar_question(question)
            if existing:
                console.print(f"[green]✓ Found in memory[/green]")
                continue
            
            # Generate GROQ answer
            print("Generating GROQ answer...")
            groq_answer = groq.generate_answer(question)
            memory.increment_groq_usage()
            
            # Generate T5 reflection
            print("Generating T5 reflection...")
            critique_prompt = f"critique: List factual errors or omissions.\n\nQ: {question}\nA: {groq_answer}"
            reflection = t5(critique_prompt, max_new_tokens=100, do_sample=False)[0]["generated_text"]
            memory.increment_t5_usage()
            
            # Auto-save
            memory.auto_save_reflection(question, groq_answer, groq_answer, reflection)
            
            console.print(f"[green]✓ Test {i} completed and saved[/green]")
        
        print("\n✅ Hybrid system test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during hybrid system test: {e}")

def main():
    """Main function with enhanced menu interface"""
    print("=" * 70)
    print("    DOMAIN-RESTRICTED CYBERSECURITY & EDUCATION QA SYSTEM")
    print("=" * 70)
    
    if not check_requirements():
        response = input("\n❓ Missing required files. Create sample data for testing? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            create_sample_data()
        else:
            print("❌ Cannot continue without required files.")
            return
    
    while True:
        print("\n" + "="*50)
        print("         SELECT AN OPTION")
        print("="*50)
        print("1. Train T5 model from scratch")
        print("2. Interactive QA (Domain-Restricted)")  # Modified
        print("3. Interactive QA (General) - Legacy")   # Added legacy option
        print("4. Test hybrid system")
        print("5. Show memory overview")
        print("6. Retrain with feedback")
        print("7. Database maintenance")
        print("8. Export feedback data")
        print("9. System statistics")
        print("10. Exit")
        print("="*50)
        
        choice = input("\nEnter choice (1-10): ").strip()
        
        if choice == "1":
            print("\n🚀 Starting T5 model training...")
            try:
                train_df, test_df = load_and_split_dataset()
                print(f"📊 Training data: {len(train_df)} examples")
                print(f"📊 Test data: {len(test_df)} examples")
                
                # Call the existing train_model function
                train_model()
                
            except Exception as e:
                print(f"❌ Training failed: {e}")
        
        elif choice == "2":
            # Check if model exists
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print("❌ No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n🎯 Starting domain-restricted QA session...")
            domain_restricted_interactive_session(MODEL_DIR)
            
        elif choice =="3":
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print("❌ No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n🎯 Starting general QA session (legacy)...")
            interactive_session(MODEL_DIR)    
        
        elif choice == "4":
            # Check if model exists
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print("❌ No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n🧪 Testing hybrid system...")
            try:
                _, test_df = load_and_split_dataset()
                test_hybrid_system(MODEL_DIR, test_df)
            except Exception as e:
                print(f"❌ Test failed: {e}")
        
        elif choice == "5":
            show_memory_overview()
        
        elif choice == "6":
            # Check if model exists
            checkpoint_exists = bool(glob.glob(os.path.join(MODEL_DIR, "checkpoint-*")))
            model_file_exists = os.path.exists(os.path.join(MODEL_DIR, "config.json"))
            
            if not (checkpoint_exists or model_file_exists):
                print("❌ No trained model found. Please train the model first (option 1).")
                continue
            
            print("\n🔄 Retraining with feedback...")
            try:
                success = retrain_from_feedback()
                if success:
                    print("✅ Retraining completed successfully!")
                else:
                    print("❌ Retraining failed or insufficient feedback data.")
            except Exception as e:
                print(f"❌ Retraining failed: {e}")
        
        elif choice == "7":
            print("\n🔧 Database maintenance...")
            try:
                database_maintenance()
            except Exception as e:
                print(f"❌ Maintenance failed: {e}")
        
        elif choice == "8":
            print("\n📤 Exporting feedback data...")
            export_path = input("Enter export path (press Enter for default): ").strip()
            if not export_path:
                export_path = None
            
            try:
                export_feedback_data(export_path)
            except Exception as e:
                print(f"❌ Export failed: {e}")
        
        elif choice == "9":
            print("\n📊 System Statistics")
            print("-" * 30)
            try:
                memory = EnhancedMemorySystem(DATABASE_PATH, MEMORY_STATS_JSON)
                stats = memory.get_memory_stats()
                
                for key, value in stats.items():
                    print(f"{key}: {value}")
                    
                # Additional system info
                print(f"\nDatabase Path: {DATABASE_PATH}")
                print(f"Model Directory: {MODEL_DIR}")
                print(f"Device: {DEVICE}")
                print(f"GROQ Model: {GROQ_MODEL}")
                
            except Exception as e:
                print(f"❌ Error retrieving stats: {e}")
        
        elif choice == "10":
            print("\n👋 Goodbye!")
            print("Thank you for using the Hybrid QA System!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-9.")
    
    print("\n" + "="*60)
    print("    SESSION ENDED")
    print("="*60)

if __name__ == "__main__":
    main()