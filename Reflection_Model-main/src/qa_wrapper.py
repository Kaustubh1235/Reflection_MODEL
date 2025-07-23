import os
import re
from dotenv import load_dotenv
from transformers import pipeline

from src.dev import (
    get_model_and_tokenizer,
    EnhancedMemorySystem,
    GroqLLMHandler,
)

# === Load environment variables ===
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "mixtral-8x7b")  # Optional

if not GROQ_API_KEY:
    raise ValueError("\u274c GROQ_API_KEY not found. Make sure it's defined in your .env file.")

MODEL_DIR = "./flan_t5_reflective_model"
FEEDBACK_JSON = "feedback.json"
MEMORY_STATS_JSON = "stats.json"
SIMILARITY_THRESHOLD = 0.8

# === Initialize systems ===
groq = GroqLLMHandler(GROQ_API_KEY, GROQ_MODEL)
model, tokenizer = get_model_and_tokenizer()
t5 = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
memory = EnhancedMemorySystem(FEEDBACK_JSON, MEMORY_STATS_JSON, SIMILARITY_THRESHOLD)
feedback_list = memory.load_feedback()


def format_bullets(text):
    text = re.sub(r"(?<!\n)(\d\.\s)", r"\n\1", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def generate_response(question: str):
    # Check memory
    existing = memory.find_similar_question(question, feedback_list)
    if existing:
        return ("\ud83d\udcc2 Answer loaded from memory:", "", format_bullets(existing["improvement"]), "memory", question)

    # Generate with GROQ
    try:
        base = groq.generate_answer(question)
    except Exception as e:
        return (f"❌ LLM error: {str(e)}", "", "", "error", question)

    # Reflect
    critique_prompt = f"critique: List any flaws in the answer.\n\nQ: {question}\nA: {base}"
    try:
        reflection = t5(critique_prompt, max_new_tokens=128, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"].strip()
    except Exception as e:
        reflection = f"⚠️ Reflection error: {str(e)}"

    # Improve
    improve_prompt = f"improve: Based on the critique, rewrite a better version.\n\nQ: {question}\nCritique: {reflection}\nOriginal Answer: {base}"
    try:
        improved = t5(improve_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_p=0.9)[0]["generated_text"].strip()
    except Exception as e:
        improved = f"⚠️ Improvement error: {str(e)}"

    return (
        format_bullets(base),
        format_bullets(reflection),
        format_bullets(improved),
        "fresh",
        question
    )


def store_user_feedback(question, user_input):
    existing = memory.find_similar_question(question, feedback_list)
    if existing:
        memory.update_feedback(existing["id"], user_input)
    else:
        memory.add_feedback({
            "question": question,
            "improvement": user_input,
            "source": "user"
        })
    memory.save_feedback()
    return " Improvement stored to memory."
