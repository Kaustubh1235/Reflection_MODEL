# import modal
# import os
# import json
# import pandas as pd
# import torch
# import time
# import sqlite3
# from contextlib import contextmanager
# from transformers import (
#     AutoTokenizer,
#     AutoModelForSeq2SeqLM,
#     Trainer,
#     TrainingArguments,
#     DataCollatorForSeq2Seq,
#     pipeline,
# )
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from typing import Dict, List, Optional, Tuple
# import hashlib
# from groq import Groq
# from dotenv import load_dotenv

# # Modal app configuration
# app = modal.App("legal-advisor-finetuning")

# # Modal image with all required dependencies
# image = modal.Image.debian_slim(python_version="3.10").pip_install([
#     "torch>=2.0.0",
#     "transformers>=4.30.0",
#     "datasets>=2.12.0",
#     "accelerate>=0.20.0",
#     "pandas>=1.5.0",
#     "scikit-learn>=1.3.0",
#     "groq>=0.4.0",
#     "python-dotenv>=1.0.0",
#     "rich>=13.0.0",
#     "wandb>=0.15.0",  # For experiment tracking
#     "deepspeed>=0.9.0",  # For memory optimization
# ])

# # Modal volumes for persistent storage
# model_volume = modal.Volume.from_name("legal-model-storage", create_if_missing=True)
# data_volume = modal.Volume.from_name("legal-data-storage", create_if_missing=True)

# # ENHANCED CONFIGURATIONS FOR A100 40GB
# PRETRAINED_MODEL = "google/flan-t5-large"  # Upgraded to large model
# MAX_INPUT_LENGTH = 1024  # Increased for legal complexity
# MAX_TARGET_LENGTH = 1024
# BATCH_SIZE = 8  # Optimized for A100 40GB
# GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 32
# NUM_TRAIN_EPOCHS = 8
# LEARNING_RATE = 5e-5
# WEIGHT_DECAY = 0.01
# WARMUP_RATIO = 0.1
# FP16 = True  # Mixed precision for A100
# DATALOADER_NUM_WORKERS = 4

# # Paths for Modal environment
# MODEL_DIR = "/model_storage"
# DATA_DIR = "/data_storage"
# PREPROCESSED_DIR = f"{DATA_DIR}/preprocessed_data/processed_constitution"
# CLEAN_CSV_PATH = f"{PREPROCESSED_DIR}/train.csv"
# TRAIN_SPLIT_PATH = f"{PREPROCESSED_DIR}/train.csv"
# TEST_SPLIT_PATH = f"{PREPROCESSED_DIR}/test.csv"
# VALIDATION_SPLIT_PATH = f"{PREPROCESSED_DIR}/validation.csv"

# class LegalQADataset(Dataset):
#     """Enhanced dataset class for legal Q&A with GPU optimization"""
    
#     def __init__(self, df, tokenizer, max_input_length, max_target_length):
#         self.df = df.reset_index(drop=True)
#         self.tokenizer = tokenizer
#         self.max_input_length = max_input_length
#         self.max_target_length = max_target_length

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         row = self.df.iloc[idx]
#         question = str(row["question_clean"]).strip()
#         answer = str(row["answer_clean"]).strip()
        
#         # Enhanced legal prompt with better structure
#         input_text = f"Constitutional Legal Analysis: {question}\n\nProvide a comprehensive legal response with relevant constitutional principles, precedents, and practical implications:"
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
        
#         # Set pad tokens to -100 for loss calculation
#         labels[labels == self.tokenizer.pad_token_id] = -100
        
#         return {
#             "input_ids": input_ids,
#             "attention_mask": attention_mask,
#             "labels": labels
#         }

# def load_and_split_legal_dataset(data_path: str, test_size=0.1, validation_size=0.1, seed=42):
#     """Load, clean, and split the legal dataset for Modal"""
    
#     # Check if your actual dataset files exist
#     actual_train_path = f"{DATA_DIR}/preprocessed_data/processed_constitution/train.csv"
#     actual_test_path = f"{DATA_DIR}/preprocessed_data/processed_constitution/test.csv" 
#     actual_val_path = f"{DATA_DIR}/preprocessed_data/processed_constitution/validation.csv"
    
#     # If splits already exist, use them directly
#     if all(os.path.exists(p) for p in [actual_train_path, actual_test_path, actual_val_path]):
#         print("Found existing dataset splits, loading them directly...")
#         train_df = pd.read_csv(actual_train_path)
#         test_df = pd.read_csv(actual_test_path)
#         validation_df = pd.read_csv(actual_val_path)
        
#         print(f"Loaded existing splits - Train: {len(train_df)}, Test: {len(test_df)}, Validation: {len(validation_df)}")
#         return train_df, test_df, validation_df
    
#     # If main dataset exists, load and split it
#     if os.path.exists(data_path):
#         print(f"Loading dataset from: {data_path}")
#         df = pd.read_csv(data_path)
#         print(f"Loaded source dataset with {len(df)} rows and columns: {list(df.columns)}")

#         # Handle different CSV formats
#         if len(df.columns) >= 2 and list(df.columns)[:2] == ['question_clean', 'answer_clean']:
#             print("Dataset already in clean format")
#             df_clean = df[['question_clean', 'answer_clean']].copy()
#         elif len(df.columns) == 3:
#             print("Processing raw 3-column dataset")
#             df.columns = ['raw_question', 'answer', 'extra']
#             df_clean = df[['raw_question', 'answer']].copy()
#             df_clean.columns = ["question_clean", "answer_clean"]
            
#             # Clean the data
#             df_clean["question_clean"] = df_clean["question_clean"].astype(str).str.replace(r'\[.*?\]\s*', '', regex=True).str.strip()
#             df_clean["answer_clean"] = df_clean["answer_clean"].astype(str).str.strip()
#         elif 'question' in df.columns and 'answer' in df.columns:
#             print("Using question/answer columns")
#             df_clean = df[['question', 'answer']].copy()
#             df_clean.columns = ["question_clean", "answer_clean"]
#         else:
#             print(f"Unexpected CSV format. Columns: {list(df.columns)}")
#             print("Creating sample data instead...")
#             return create_sample_legal_data(data_path, test_size, validation_size, seed)

#         # Remove null/empty rows
#         original_len = len(df_clean)
#         df_clean = df_clean.dropna()
#         df_clean = df_clean[df_clean["question_clean"].str.len() > 10]
#         df_clean = df_clean[df_clean["answer_clean"].str.len() > 20]
        
#         print(f"Cleaned dataset: {original_len} -> {len(df_clean)} valid pairs")
        
#         if len(df_clean) < 10:
#             print("Warning: Very few valid pairs, creating sample data...")
#             return create_sample_legal_data(data_path, test_size, validation_size, seed)

#         # Split dataset
#         train_val_df, test_df = train_test_split(
#             df_clean, test_size=test_size, random_state=seed, shuffle=True
#         )
        
#         adjusted_val_size = validation_size / (1 - test_size)
#         train_df, validation_df = train_test_split(
#             train_val_df, test_size=adjusted_val_size, random_state=seed, shuffle=True
#         )
        
#         # Save splits
#         os.makedirs(os.path.dirname(TRAIN_SPLIT_PATH), exist_ok=True)
#         train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
#         test_df.to_csv(TEST_SPLIT_PATH, index=False)
#         validation_df.to_csv(VALIDATION_SPLIT_PATH, index=False)
        
#         print(f"Dataset splits saved - Train: {len(train_df)}, Validation: {len(validation_df)}, Test: {len(test_df)}")
        
#         return train_df, test_df, validation_df
    
#     else:
#         print(f"Dataset not found at {data_path}, creating sample data...")
#         return create_sample_legal_data(data_path, test_size, validation_size, seed)

# def create_sample_legal_data(data_path: str, test_size=0.1, validation_size=0.1, seed=42):
#     """Create comprehensive sample legal/constitutional data"""
#     print("Creating expanded sample legal constitutional dataset...")
    
#     legal_sample_data = {
#         'question_clean': [
#             "What are the fundamental rights guaranteed under the Constitution?",
#             "How does the separation of powers work in constitutional law?",
#             "What is due process and when does it apply?",
#             "Explain the concept of judicial review and its constitutional basis",
#             "What are the limits of free speech under the First Amendment?",
#             "How does the equal protection clause work in practice?",
#             "What is the role of federalism in constitutional law?",
#             "When can the government restrict constitutional rights?",
#             "What are Miranda rights and their constitutional foundation?",
#             "How does constitutional interpretation work in modern courts?",
#             "What is the difference between procedural and substantive due process?",
#             "How do the Bill of Rights apply to state governments?",
#             "What is the commerce clause and how has it evolved?",
#             "Explain the concept of constitutional standing in federal courts",
#             "What are the key principles of Fourth Amendment search and seizure?",
#             "How does the Establishment Clause limit government action?",
#             "What is the role of precedent in constitutional interpretation?",
#             "How do constitutional amendments get ratified?",
#             "What are the president's constitutional powers in foreign affairs?",
#             "How does the Supreme Court handle constitutional challenges?",
#             "What is the significance of the Fourteenth Amendment?",
#             "How do states' rights interact with federal constitutional law?",
#             "What are the constitutional requirements for criminal prosecutions?",
#             "How does the Constitution protect property rights?",
#             "What is the role of constitutional conventions in interpretation?"
#         ],
#         'answer_clean': [
#             "Fundamental constitutional rights include life, liberty, and property protected by due process; equal protection under law; freedom of speech, religion, press, assembly, and petition; protection against unreasonable searches and seizures; right to counsel and fair trial; protection against cruel and unusual punishment; and voting rights. These rights form the foundation of constitutional democracy and are enforceable against government infringement through judicial review.",
            
#             "Separation of powers divides government authority among three coordinate branches: legislative (Congress makes laws), executive (President enforces laws), and judicial (courts interpret laws). Each branch has distinct constitutional roles and checking powers over others - Congress can impeach, override vetoes, and control budgets; President can veto legislation and appoint judges; courts can declare actions unconstitutional. This system prevents concentration of power and protects constitutional governance.",
            
#             "Due process requires fair legal procedures before government can deprive someone of life, liberty, or property. Procedural due process mandates adequate notice, opportunity to be heard, neutral decision-maker, and appropriate procedures. Substantive due process protects fundamental rights from arbitrary government action regardless of procedures used. Both Fifth and Fourteenth Amendments contain due process clauses applying to federal and state governments respectively.",
            
#             "Judicial review is the power of federal courts to examine government actions and declare them unconstitutional. Established in Marbury v. Madison (1803), this doctrine makes courts final arbiters of constitutional meaning. Courts can review legislative acts, executive actions, and state laws for constitutional compliance. This power is implied from constitutional structure, judicial oath, and supremacy clause, serving as crucial check on other branches.",
            
#             "First Amendment free speech protections are broad but not absolute. Government cannot restrict speech based on content or viewpoint without compelling justification. Protected speech includes political expression, symbolic speech, and even offensive speech. Unprotected categories include incitement to imminent violence, true threats, defamation, obscenity, and commercial fraud. Time, place, and manner regulations must be content-neutral, narrowly tailored, and leave ample alternative channels.",
            
#             "Equal Protection Clause requires that similarly situated people receive similar treatment under law. Government classifications must serve legitimate purposes and use appropriate means. Strict scrutiny applies to suspect classifications (race, national origin, alienage) requiring compelling government interest and least restrictive means. Intermediate scrutiny applies to quasi-suspect classifications (gender, illegitimacy). Rational basis review applies to other classifications, requiring legitimate government purpose and rational relationship.",
            
#             "Federalism divides governmental power between national and state governments, with each having distinct constitutional spheres of authority. Constitution grants enumerated powers to federal government while reserving others to states via Tenth Amendment. Supremacy Clause makes valid federal law supreme over conflicting state law. This structure protects liberty by preventing concentration of power while maintaining national unity and allowing state experimentation within constitutional limits.",
            
#             "Constitutional rights may be restricted when government demonstrates sufficient justification using appropriate level of scrutiny. Fundamental rights require compelling government interest and least restrictive means (strict scrutiny). Important rights require important government interest and substantially related means (intermediate scrutiny). Other rights require legitimate government purpose and rational relationship (rational basis). Emergency powers, national security, and public safety can justify some restrictions under proper constitutional analysis.",
            
#             "Miranda rights inform suspects of Fifth Amendment right against self-incrimination and Sixth Amendment right to counsel during custodial interrogation. Police must warn suspects they have right to remain silent, statements can be used against them, right to attorney, and right to appointed counsel if indigent. These warnings are required before custodial interrogation begins. Failure to give Miranda warnings makes statements inadmissible, but doesn't invalidate arrest or prevent other evidence use.",
            
#             "Constitutional interpretation involves multiple methodologies including textualism (plain meaning of words), originalism (original public meaning or framers' intent), living constitutionalism (evolving interpretation for modern circumstances), precedent (stare decisis), and structural arguments (constitutional design). Courts balance these approaches considering text, history, precedent, consequences, and constitutional structure. Different justices emphasize different interpretive methods, leading to diverse constitutional jurisprudence.",
            
#             "Procedural due process requires fair procedures before government deprives someone of protected interests, including adequate notice, opportunity to be heard, neutral decision-maker, and procedures appropriate to the situation. Substantive due process protects certain fundamental rights from government interference regardless of procedures used, including privacy rights, family autonomy, bodily integrity, and other liberty interests deemed fundamental by courts through constitutional interpretation.",
            
#             "Bill of Rights originally applied only to federal government, but Fourteenth Amendment's Due Process Clause has been interpreted to incorporate most Bill of Rights protections against state governments. Through selective incorporation, Supreme Court has applied First Amendment, most Fourth through Eighth Amendment protections, and other rights to state and local governments. Only a few provisions remain unincorporated, such as grand jury requirement and excessive fines clause.",
            
#             "Commerce Clause grants Congress power to regulate interstate commerce, originally understood as trade between states. Modern interpretation, developed through cases like Wickard v. Filburn, expanded federal power to regulate activities substantially affecting interstate commerce, including local activities with cumulative effects. Recent decisions like Lopez and Morrison have attempted to limit Commerce Clause power, requiring genuine connection to interstate commercial activity.",
            
#             "Constitutional standing requires plaintiffs to demonstrate injury-in-fact (concrete, particularized harm), causation (injury caused by defendant's conduct), and redressability (judicial remedy can address injury). These Article III requirements ensure federal courts only hear actual cases and controversies, not abstract disputes. Standing doctrine limits who can challenge government action and maintains separation of powers by restricting judicial intervention to proper cases.",
            
#             "Fourth Amendment protects against unreasonable searches and seizures, requiring warrants based on probable cause for most searches. Key principles include reasonable expectation of privacy test, warrant requirement with exceptions (exigent circumstances, consent, search incident to arrest, plain view), probable cause standard for warrants, and exclusionary rule preventing use of illegally obtained evidence. Balances law enforcement needs with privacy protection through constitutional reasonableness analysis.",
            
#             "Establishment Clause prohibits government from establishing religion or excessively entangling with religious institutions. Courts use various tests including Lemon test (secular purpose, neutral effect, no excessive entanglement), endorsement test (government cannot endorse religion), and coercion test (government cannot coerce religious participation). Clause requires government neutrality toward religion while accommodating religious exercise and historical practices within constitutional bounds.",
            
#             "Precedent (stare decisis) promotes legal stability by requiring courts to follow earlier decisions on similar legal questions. Strong presumption favors following precedent unless clearly erroneous, unworkable, or inconsistent with later developments. Supreme Court can overrule its own precedents but rarely does so. Lower courts must follow higher court precedents. Constitutional precedents receive strong protection but can be overruled when constitutional interpretation requires correction of fundamental errors.",
            
#             "Constitutional amendments require proposal by two-thirds of both houses of Congress or national convention called by two-thirds of state legislatures, then ratification by three-fourths of state legislatures or state conventions. Only amendment method used successfully is congressional proposal with state legislative ratification. Process intentionally difficult to ensure broad consensus for constitutional change while allowing adaptation to changing circumstances through democratic process.",
            
#             "President's foreign affairs powers include treaty negotiation (with Senate consent), diplomatic recognition, commander-in-chief authority, and execution of foreign policy. These powers derive from constitutional text, historical practice, and functional necessity. Courts generally defer to executive branch in foreign affairs due to president's institutional advantages and constitutional role. However, Congress retains important foreign affairs powers including war declaration, commerce regulation, and treaty implementation through legislation.",
            
#             "Supreme Court handles constitutional challenges through careful case selection (certiorari), thorough briefing and argument, constitutional analysis using various interpretive methods, and reasoned opinions explaining constitutional meaning. Court considers text, history, precedent, structure, and consequences in constitutional interpretation. Different levels of scrutiny apply depending on rights involved and government interests asserted. Court's constitutional decisions are supreme law binding on all government actors.",
            
#             "Fourteenth Amendment fundamentally transformed constitutional law by requiring states to provide equal protection and due process, granting citizenship to all persons born in United States, and providing enforcement power to Congress. It reversed Dred Scott decision, established birthright citizenship, incorporated Bill of Rights against states, created modern equal protection doctrine, and expanded federal power to protect individual rights against state government violations through constitutional enforcement.",
            
#             "States' rights and federal constitutional law interact through federalism principles balancing national unity with state autonomy. States retain reserved powers under Tenth Amendment but cannot violate federal constitutional rights or valid federal law under Supremacy Clause. Federal government has enumerated powers but cannot commandeer state governments. Courts balance competing federalism values while ensuring constitutional rights protection and maintaining constitutional structure.",
            
#             "Constitutional requirements for criminal prosecutions include grand jury indictment for felonies (Fifth Amendment), speedy and public trial by impartial jury (Sixth Amendment), due process protections (Fifth and Fourteenth Amendments), right to counsel (Sixth Amendment), protection against self-incrimination (Fifth Amendment), confrontation of witnesses (Sixth Amendment), and protection against cruel and unusual punishment (Eighth Amendment). These requirements ensure fair criminal process and protect individual rights.",
            
#             "Constitution protects property rights through Takings Clause (Fifth Amendment) requiring just compensation for government taking of private property for public use, Due Process Clauses protecting against arbitrary deprivation of property, and Contract Clause (Article I, Section 10) limiting state impairment of contracts. Property rights receive constitutional protection but are subject to reasonable regulation for public health, safety, and welfare under police power.",
            
#             "Constitutional conventions play important role in interpretation through historical practice, contemporary understanding, and institutional arrangements not explicitly detailed in constitutional text. Examples include executive privilege, judicial review, legislative committee system, and political party roles. These conventions supplement written constitution by providing operational framework for constitutional government while remaining subject to constitutional text and judicial interpretation when conflicts arise."
#         ]
#     }
    
#     df = pd.DataFrame(legal_sample_data)
    
#     # Ensure directory exists
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
#     df.to_csv(data_path, index=False)
    
#     print(f"Created expanded legal dataset: {len(df)} Q&A pairs")
    
#     # Split the data
#     train_val_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, shuffle=True)
#     adjusted_val_size = validation_size / (1 - test_size)
#     train_df, validation_df = train_test_split(train_val_df, test_size=adjusted_val_size, random_state=seed, shuffle=True)
    
#     # Save splits
#     train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
#     test_df.to_csv(TEST_SPLIT_PATH, index=False)
#     validation_df.to_csv(VALIDATION_SPLIT_PATH, index=False)
    
#     return train_df, test_df, validation_df

# @app.function(
#     image=image,
#     gpu="A100-40GB",
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
#     timeout=3600 * 6,  # 6 hours timeout
#     memory=32768,  # 32GB RAM
#     cpu=8,
# )
# def train_legal_model():
#     """Train the T5 model specifically for legal advisory on A100 40GB"""
#     from accelerate import Accelerator
    
#     # Initialize accelerate for optimal GPU usage (disable W&B for now)
#     accelerator = Accelerator(
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         mixed_precision="fp16" if FP16 else None,
#         # log_with="wandb"  # Disabled for now
#     )
    
#     # Skip W&B initialization to avoid API key issues
#     print("Skipping W&B logging (no API key configured)")
    
#     print("Starting Legal Advisor T5 model training on Modal A100...")
    
#     try:
#         # Load and prepare dataset
#         train_df, test_df, validation_df = load_and_split_legal_dataset(CLEAN_CSV_PATH)
#         print(f"Dataset loaded - Train: {len(train_df)}, Val: {len(validation_df)}, Test: {len(test_df)}")
#     except Exception as e:
#         print(f"Error loading legal dataset: {e}")
#         return {"status": "failed", "error": str(e)}
    
#     print(f"Loading pretrained model: {PRETRAINED_MODEL}")
    
#     # Load model and tokenizer with optimizations
#     tokenizer = AutoTokenizer.from_pretrained(
#         PRETRAINED_MODEL,
#         use_fast=True,
#         trust_remote_code=True
#     )
    
#     model = AutoModelForSeq2SeqLM.from_pretrained(
#         PRETRAINED_MODEL,
#         torch_dtype=torch.float16 if FP16 else torch.float32,
#         trust_remote_code=True
#     )
    
#     # Add legal-specific special tokens
#     legal_special_tokens = {
#         "additional_special_tokens": [
#             "<legal_analysis>", "<constitutional_law>", "<precedent>", 
#             "<statute>", "<case_law>", "<jurisdiction>", "<legal_principle>",
#             "<due_process>", "<equal_protection>", "<first_amendment>",
#             "<fourth_amendment>", "<fourteenth_amendment>"
#         ]
#     }
#     tokenizer.add_special_tokens(legal_special_tokens)
#     model.resize_token_embeddings(len(tokenizer))
    
#     # Create optimized datasets
#     train_dataset = LegalQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     validation_dataset = LegalQADataset(validation_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#     test_dataset = LegalQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
    
#     data_collator = DataCollatorForSeq2Seq(
#         tokenizer=tokenizer,
#         model=model,
#         padding=True,
#         return_tensors="pt"
#     )
    
#     # Optimized training arguments for A100 40GB
#     training_args = TrainingArguments(
#         output_dir=MODEL_DIR,
#         overwrite_output_dir=True,
#         num_train_epochs=NUM_TRAIN_EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         per_device_eval_batch_size=BATCH_SIZE,
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         learning_rate=LEARNING_RATE,
#         weight_decay=WEIGHT_DECAY,
#         warmup_ratio=WARMUP_RATIO,
        
#         # Optimization settings for A100
#         fp16=FP16,
#         dataloader_num_workers=DATALOADER_NUM_WORKERS,
#         dataloader_pin_memory=True,
#         group_by_length=True,  # Group similar length sequences
        
#         # Evaluation and saving - Fixed strategy mismatch
#         evaluation_strategy="steps",
#         eval_steps=100,
#         save_strategy="steps",
#         save_steps=200,
#         save_total_limit=3,
#         load_best_model_at_end=False,  # Disabled to avoid strategy mismatch
#         # metric_for_best_model="eval_loss",
#         # greater_is_better=False,
        
#         # Logging
#         logging_dir=f"{MODEL_DIR}/logs",
#         logging_steps=50,
#         logging_strategy="steps",
#         report_to=None,  # Disable W&B
        
#         # Memory optimization
#         gradient_checkpointing=True,
#         remove_unused_columns=False,
        
#         # Performance settings
#         tf32=True,  # Use TF32 on A100 for better performance
#     )
    
#     # Initialize trainer
#     trainer = Trainer(
#         model=model,
#         args=training_args,
#         train_dataset=train_dataset,
#         eval_dataset=validation_dataset,
#         data_collator=data_collator,
#         tokenizer=tokenizer,
#     )
    
#     print("Starting legal model training on A100...")
#     start_time = time.time()
    
#     try:
#         # Train the model
#         trainer.train()
        
#         # Save final model
#         trainer.save_model()
#         tokenizer.save_pretrained(MODEL_DIR)
        
#         # Evaluate on test set
#         print("Evaluating on test set...")
#         test_results = trainer.evaluate(eval_dataset=test_dataset)
        
#         # Commit volumes to persist changes
#         model_volume.commit()
#         data_volume.commit()
        
#         training_time = time.time() - start_time
        
#         result = {
#             "status": "success",
#             "training_time": training_time,
#             "test_results": test_results,
#             "model_path": MODEL_DIR,
#             "dataset_info": {
#                 "train_samples": len(train_df),
#                 "val_samples": len(validation_df),
#                 "test_samples": len(test_df)
#             }
#         }
        
#         print(f"Legal Advisor training completed successfully in {training_time:.2f}s!")
#         print(f"Test Results: {test_results}")
        
#         return result
        
#     except Exception as e:
#         print(f"Training error: {e}")
#         return {"status": "failed", "error": str(e)}

# @app.function(
#     image=image,
#     gpu="A100-40GB",
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
#     timeout=600,
# )
# def legal_inference(question: str, max_length: int = 512):
#     """Run inference on trained legal model"""
#     import torch
#     from transformers import pipeline
    
#     try:
#         # Load the trained model
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             MODEL_DIR,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#         # Create pipeline
#         legal_pipeline = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             device=0,  # Use GPU
#             torch_dtype=torch.float16
#         )
        
#         # Enhanced legal prompt
#         prompt = f"Constitutional Legal Analysis: {question}\n\nProvide a comprehensive legal response with relevant constitutional principles, precedents, and practical implications:"
        
#         # Generate response
#         response = legal_pipeline(
#             prompt,
#             max_length=max_length,
#             min_length=50,
#             do_sample=True,
#             temperature=0.7,
#             top_p=0.9,
#             num_beams=3,
#             repetition_penalty=1.1,
#             no_repeat_ngram_size=3,
#         )
        
#         generated_text = response[0]["generated_text"].strip()
        
#         # Add legal disclaimer
#         if "not legal advice" not in generated_text.lower():
#             generated_text += "\n\nDisclaimer: This is general legal information, not legal advice. Please consult with a qualified attorney for your specific situation."
        
#         return {
#             "status": "success",
#             "question": question,
#             "answer": generated_text,
#             "model_used": "flan-t5-large-legal"
#         }
        
#     except Exception as e:
#         return {
#             "status": "error",
#             "error": str(e),
#             "question": question
#         }

# @app.function(
#     image=image,
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
# )
# def check_model_status():
#     """Check if trained model exists and get basic info"""
#     import glob
    
#     model_files = glob.glob(f"{MODEL_DIR}/*")
#     checkpoint_dirs = glob.glob(f"{MODEL_DIR}/checkpoint-*")
    
#     return {
#         "model_dir_exists": os.path.exists(MODEL_DIR),
#         "model_files": len(model_files),
#         "checkpoints": len(checkpoint_dirs),
#         "config_exists": os.path.exists(f"{MODEL_DIR}/config.json"),
#         "tokenizer_exists": os.path.exists(f"{MODEL_DIR}/tokenizer.json"),
#         "latest_checkpoint": max(checkpoint_dirs) if checkpoint_dirs else None
#     }

# # Local functions for Modal deployment
# @app.local_entrypoint()
# def main(command: str = "help", question: str = ""):
#     """Main entry point for Modal deployment"""
    
#     print("Legal Advisor Fine-tuning on Modal A100")
#     print("=" * 50)
    
#     if command == "train":
#         print("Starting training on Modal A100...")
#         print("This will take several hours. Training in progress...")
#         result = train_legal_model.remote()
#         print(f"Training completed!")
#         print(f"Result: {result}")
        
#     elif command == "advanced_train":
#         print("Starting advanced training with curriculum learning...")
#         result = advanced_legal_training.remote(use_curriculum=True)
#         print(f"Advanced training completed!")
#         print(f"Result: {result}")
        
#     elif command == "status":
#         print("Checking model status...")
#         status = check_model_status.remote()
#         print(f"Model status: {status}")
        
#     elif command == "inference":
#         if not question:
#             question = "What is the role of judicial review in constitutional law?"
#             print(f"Using default question: {question}")
#         print(f"Running inference on: {question}")
#         result = legal_inference.remote(question)
#         print(f"Inference result:")
#         print(f"Answer: {result.get('answer', 'Error occurred')}")
        
#     elif command == "evaluate":
#         print("Evaluating model performance...")
#         result = evaluate_model_performance.remote()
#         print(f"Evaluation results: {result}")
        
#     elif command == "export":
#         print("Exporting model for deployment...")
#         result = export_model_for_deployment.remote()
#         print(f"Export result: {result}")
        
#     elif command == "batch_test":
#         print("Running batch inference test...")
#         results = example_batch_inference.remote()
#         print(f"Batch test completed with {len(results)} results")
        
#     elif command == "check_data":
#         print("Checking available datasets...")
#         data_info = check_available_datasets.remote()
#         print(f"Dataset information: {data_info}")
        
#     elif command == "list_data":
#         print("Listing data directory contents...")
#         dir_info = list_data_directory.remote()
#         print(f"Directory structure: {dir_info}")
        
#     elif command == "info":
#         print("Getting model information...")
#         info = get_model_info.remote()
#         print(f"Model info: {info}")
        
#     else:
#         print("Available commands:")
#         print("  train          - Train the legal model")
#         print("  advanced_train - Train with curriculum learning")
#         print("  status         - Check model status")
#         print("  inference      - Test inference")
#         print("  evaluate       - Evaluate model performance")
#         print("  export         - Export model for deployment")
#         print("  batch_test     - Run batch inference test")
#         print("  check_data     - Check available datasets")
#         print("  list_data      - List data directory contents")
#         print("  info           - Get model information")
#         print("\nUsage examples:")
#         print("  modal run modal_legal_advisor.py --command train")
#         print("  modal run modal_legal_advisor.py --command check_data")
#         print("  modal run modal_legal_advisor.py --command inference --question 'What is due process?'")
#         print("  modal run modal_legal_advisor.py --command status")

# # Example usage functions
# @app.function(image=image)
# def example_batch_inference():
#     """Example of batch inference for multiple questions"""
#     sample_questions = [
#         "What is the role of judicial review in constitutional law?",
#         "How does the First Amendment protect freedom of speech?",
#         "Explain the concept of due process under the Fourteenth Amendment",
#         "What are the limits of federal power under the Commerce Clause?",
#         "How does the Equal Protection Clause apply to government actions?"
#     ]
    
#     results = []
#     for question in sample_questions:
#         result = legal_inference.remote(question, max_length=512)
#         results.append(result)
    
#     return results

# @app.function(
#     image=image,
#     gpu="A100-40GB", 
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
#     timeout=1800,
# )
# def evaluate_model_performance():
#     """Comprehensive model evaluation on test set"""
#     import torch
#     from transformers import pipeline
#     import pandas as pd
#     from sklearn.metrics import accuracy_score
#     import numpy as np
    
#     try:
#         # Load test dataset
#         if not os.path.exists(TEST_SPLIT_PATH):
#             print("Test dataset not found")
#             return {"status": "error", "message": "Test dataset not found"}
            
#         test_df = pd.read_csv(TEST_SPLIT_PATH)
        
#         # Load trained model
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             MODEL_DIR,
#             torch_dtype=torch.float16,
#             device_map="auto"
#         )
        
#         legal_pipeline = pipeline(
#             "text2text-generation",
#             model=model,
#             tokenizer=tokenizer,
#             device=0,
#             torch_dtype=torch.float16
#         )
        
#         # Evaluate on sample of test set
#         sample_size = min(20, len(test_df))  # Evaluate on up to 20 samples
#         test_sample = test_df.sample(n=sample_size, random_state=42)
        
#         results = []
#         for _, row in test_sample.iterrows():
#             question = row['question_clean']
#             expected_answer = row['answer_clean']
            
#             prompt = f"Constitutional Legal Analysis: {question}\n\nProvide a comprehensive legal response:"
            
#             generated = legal_pipeline(
#                 prompt,
#                 max_length=400,
#                 min_length=50,
#                 do_sample=True,
#                 temperature=0.7,
#                 top_p=0.9,
#                 num_beams=2,
#             )
            
#             generated_text = generated[0]["generated_text"].strip()
            
#             # Simple evaluation metrics
#             results.append({
#                 "question": question,
#                 "expected": expected_answer[:200] + "...",  # Truncate for display
#                 "generated": generated_text[:200] + "...",
#                 "length_ratio": len(generated_text) / len(expected_answer),
#                 "contains_legal_terms": any(term in generated_text.lower() for term in 
#                     ["constitutional", "amendment", "court", "law", "legal", "rights", "clause"])
#             })
        
#         # Calculate aggregate metrics
#         avg_length_ratio = np.mean([r["length_ratio"] for r in results])
#         legal_term_coverage = np.mean([r["contains_legal_terms"] for r in results])
        
#         evaluation_summary = {
#             "status": "success",
#             "samples_evaluated": len(results),
#             "avg_length_ratio": avg_length_ratio,
#             "legal_term_coverage": legal_term_coverage,
#             "detailed_results": results[:5]  # Return first 5 for inspection
#         }
        
#         return evaluation_summary
        
#     except Exception as e:
#         return {"status": "error", "error": str(e)}

# @app.function(
#     image=image,
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
#     timeout=300,
# )
# def export_model_for_deployment():
#     """Export trained model in format suitable for deployment"""
#     import torch
#     import shutil
    
#     try:
#         if not os.path.exists(f"{MODEL_DIR}/config.json"):
#             return {"status": "error", "message": "No trained model found"}
        
#         # Create deployment package
#         deployment_dir = f"{MODEL_DIR}/deployment_package"
#         os.makedirs(deployment_dir, exist_ok=True)
        
#         # Copy essential model files
#         essential_files = [
#             "config.json",
#             "pytorch_model.bin",
#             "tokenizer.json",
#             "tokenizer_config.json",
#             "special_tokens_map.json",
#             "generation_config.json"
#         ]
        
#         copied_files = []
#         for file in essential_files:
#             src = f"{MODEL_DIR}/{file}"
#             if os.path.exists(src):
#                 shutil.copy2(src, f"{deployment_dir}/{file}")
#                 copied_files.append(file)
        
#         # Create deployment info
#         deployment_info = {
#             "model_name": "legal-advisor-flan-t5-large",
#             "model_type": "text2text-generation",
#             "base_model": PRETRAINED_MODEL,
#             "max_input_length": MAX_INPUT_LENGTH,
#             "max_target_length": MAX_TARGET_LENGTH,
#             "training_config": {
#                 "batch_size": BATCH_SIZE,
#                 "learning_rate": LEARNING_RATE,
#                 "epochs": NUM_TRAIN_EPOCHS,
#                 "fp16": FP16
#             },
#             "usage": {
#                 "task": "legal_advisory",
#                 "domain": "constitutional_law",
#                 "prompt_template": "Constitutional Legal Analysis: {question}\\n\\nProvide a comprehensive legal response:"
#             }
#         }
        
#         with open(f"{deployment_dir}/deployment_info.json", "w") as f:
#             json.dump(deployment_info, f, indent=2)
        
#         # Commit changes
#         model_volume.commit()
        
#         return {
#             "status": "success",
#             "deployment_dir": deployment_dir,
#             "copied_files": copied_files,
#             "deployment_info": deployment_info
#         }
        
#     except Exception as e:
#         return {"status": "error", "error": str(e)}

# # Advanced training function with curriculum learning
# @app.function(
#     image=image,
#     gpu="A100-40GB",
#     volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume},
#     timeout=3600 * 8,  # 8 hours for advanced training
#     memory=32768,
#     cpu=8,
# )
# def advanced_legal_training(use_curriculum=True, enable_deepspeed=False):
#     """Advanced training with curriculum learning and optimization techniques"""
#     from accelerate import Accelerator
#     import numpy as np
    
#     # Initialize accelerator with advanced settings (disable W&B for now)
#     accelerator = Accelerator(
#         gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#         mixed_precision="fp16" if FP16 else None,
#         # log_with="wandb",  # Disabled for now
#         split_batches=True,
#         cpu=False,
#     )
    
#     # Skip W&B initialization to avoid API key issues
#     print("Skipping W&B logging (no API key configured)")
    
#     print("Starting Advanced Legal Training with optimizations...")
    
#     try:
#         # Load dataset
#         train_df, test_df, validation_df = load_and_split_legal_dataset(CLEAN_CSV_PATH)
        
#         # Curriculum learning: sort by complexity if enabled
#         if use_curriculum:
#             # Simple heuristic: sort by combined length of question and answer
#             train_df['complexity'] = train_df['question_clean'].str.len() + train_df['answer_clean'].str.len()
#             train_df = train_df.sort_values('complexity').reset_index(drop=True)
#             print("Applied curriculum learning: training on simpler examples first")
        
#         # Load model with advanced configurations
#         tokenizer = AutoTokenizer.from_pretrained(
#             PRETRAINED_MODEL,
#             use_fast=True,
#             trust_remote_code=True
#         )
        
#         model = AutoModelForSeq2SeqLM.from_pretrained(
#             PRETRAINED_MODEL,
#             torch_dtype=torch.float16 if FP16 else torch.float32,
#             trust_remote_code=True,
#             gradient_checkpointing=True,  # Save memory
#         )
        
#         # Enhanced tokenizer with more legal terms
#         extended_legal_tokens = {
#             "additional_special_tokens": [
#                 "<legal_analysis>", "<constitutional_law>", "<precedent>", 
#                 "<statute>", "<case_law>", "<jurisdiction>", "<legal_principle>",
#                 "<due_process>", "<equal_protection>", "<first_amendment>",
#                 "<fourth_amendment>", "<fourteenth_amendment>", "<commerce_clause>",
#                 "<establishment_clause>", "<free_speech>", "<search_seizure>",
#                 "<miranda_rights>", "<judicial_review>", "<separation_powers>",
#                 "<federalism>", "<substantive_due_process>", "<procedural_due_process>"
#             ]
#         }
#         tokenizer.add_special_tokens(extended_legal_tokens)
#         model.resize_token_embeddings(len(tokenizer))
        
#         # Create datasets
#         train_dataset = LegalQADataset(train_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#         validation_dataset = LegalQADataset(validation_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
        
#         data_collator = DataCollatorForSeq2Seq(
#             tokenizer=tokenizer,
#             model=model,
#             padding=True,
#             return_tensors="pt"
#         )
        
#         # Advanced training arguments
#         training_args = TrainingArguments(
#             output_dir=MODEL_DIR,
#             overwrite_output_dir=True,
#             num_train_epochs=NUM_TRAIN_EPOCHS + 2,  # Extended training
#             per_device_train_batch_size=BATCH_SIZE,
#             per_device_eval_batch_size=BATCH_SIZE,
#             gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
#             learning_rate=LEARNING_RATE,
#             weight_decay=WEIGHT_DECAY,
#             warmup_ratio=WARMUP_RATIO,
            
#             # Advanced optimization
#             fp16=FP16,
#             tf32=True,
#             dataloader_num_workers=DATALOADER_NUM_WORKERS,
#             dataloader_pin_memory=True,
#             group_by_length=True,
            
#             # Learning rate scheduling
#             lr_scheduler_type="cosine",
#             warmup_steps=200,
            
#             # Evaluation and saving - Fixed strategy mismatch
#             evaluation_strategy="steps",
#             eval_steps=50,
#             save_strategy="steps", 
#             save_steps=100,
#             save_total_limit=5,
#             load_best_model_at_end=False,  # Disabled to avoid strategy mismatch
#             # metric_for_best_model="eval_loss",
#             # greater_is_better=False,
            
#             # Logging
#             logging_dir=f"{MODEL_DIR}/advanced_logs",
#             logging_steps=25,
#             logging_strategy="steps",
#             report_to=None,  # Disable W&B
            
#             # Memory and performance optimization
#             gradient_checkpointing=True,
#             remove_unused_columns=False,
#             max_grad_norm=1.0,
            
#             # Advanced features
#             prediction_loss_only=False,
#             include_inputs_for_metrics=False,
#         )
        
#         # Custom trainer for advanced features
#         trainer = Trainer(
#             model=model,
#             args=training_args,
#             train_dataset=train_dataset,
#             eval_dataset=validation_dataset,
#             data_collator=data_collator,
#             tokenizer=tokenizer,
#         )
        
#         print("Starting advanced legal model training...")
#         start_time = time.time()
        
#         # Training with monitoring
#         trainer.train()
        
#         # Save the final model
#         trainer.save_model()
#         tokenizer.save_pretrained(MODEL_DIR)
        
#         # Comprehensive evaluation
#         test_dataset = LegalQADataset(test_df, tokenizer, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH)
#         test_results = trainer.evaluate(eval_dataset=test_dataset)
        
#         # Additional metrics
#         train_loss = trainer.state.log_history[-1].get('train_loss', 0)
#         eval_loss = test_results.get('eval_loss', 0)
        
#         # Commit volumes
#         model_volume.commit()
#         data_volume.commit()
        
#         training_time = time.time() - start_time
        
#         result = {
#             "status": "success",
#             "training_type": "advanced",
#             "training_time": training_time,
#             "curriculum_learning": use_curriculum,
#             "final_train_loss": train_loss,
#             "final_eval_loss": eval_loss,
#             "test_results": test_results,
#             "model_path": MODEL_DIR,
#             "total_parameters": sum(p.numel() for p in model.parameters()),
#             "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)
#         }
        
#         print(f"Advanced training completed in {training_time:.2f}s!")
#         print(f"Final train loss: {train_loss:.4f}")
#         print(f"Final eval loss: {eval_loss:.4f}")
        
#         return result
        
#     except Exception as e:
#         print(f"Advanced training error: {e}")
#         return {"status": "failed", "error": str(e)}
#         if accelerator.is_main_process:
#             wandb.log({
#                 "final_train_loss": train_loss,
#                 "final_eval_loss": eval_loss,
#                 "training_time": training_time
#             })
#             wandb.finish()
        
#         return result
        
#     except Exception as e:
#         print(f"Advanced training error: {e}")
#         return {"status": "failed", "error": str(e)}

# # Usage examples and deployment helpers
# if __name__ == "__main__":
#     # This section shows example usage - these would be called remotely via Modal
    
#     print("""
#     Legal Advisor Fine-tuning on Modal A100 - Usage Examples:
    
#     1. Basic Training:
#        modal run modal_legal_advisor.py train
    
#     2. Check Model Status:
#        modal run modal_legal_advisor.py status
    
#     3. Test Inference:
#        modal run modal_legal_advisor.py inference "What is judicial review?"
    
#     4. Advanced Training (via Python):
#        result = advanced_legal_training.remote(use_curriculum=True)
    
#     5. Model Evaluation:
#        eval_results = evaluate_model_performance.remote()
    
#     6. Export for Deployment:
#        export_result = export_model_for_deployment.remote()
    
#     7. Batch Inference:
#        batch_results = example_batch_inference.remote()
#     """)

# # Additional utility functions for production deployment
# @app.function(image=image, volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume})
# def check_available_datasets():
#     """Check what datasets are available in the data directory"""
#     import glob
    
#     try:
#         # Check the data directory structure
#         data_info = {
#             "data_dir_exists": os.path.exists(DATA_DIR),
#             "preprocessed_dir_exists": os.path.exists(PREPROCESSED_DIR),
#             "available_csv_files": [],
#             "directory_structure": {}
#         }
        
#         if os.path.exists(DATA_DIR):
#             # List all CSV files in the data directory
#             csv_files = glob.glob(f"{DATA_DIR}/**/*.csv", recursive=True)
#             data_info["available_csv_files"] = [f.replace(DATA_DIR, "") for f in csv_files]
            
#             # Check directory structure
#             for root, dirs, files in os.walk(DATA_DIR):
#                 rel_path = root.replace(DATA_DIR, "").strip("/")
#                 if not rel_path:
#                     rel_path = "root"
#                 data_info["directory_structure"][rel_path] = {
#                     "directories": dirs,
#                     "files": files
#                 }
        
#         # Check specific expected files
#         expected_files = {
#             "main_dataset": CLEAN_CSV_PATH,
#             "train_split": TRAIN_SPLIT_PATH,
#             "test_split": TEST_SPLIT_PATH,
#             "validation_split": VALIDATION_SPLIT_PATH
#         }
        
#         for name, path in expected_files.items():
#             exists = os.path.exists(path)
#             size = os.path.getsize(path) if exists else 0
#             data_info[f"{name}_exists"] = exists
#             data_info[f"{name}_size"] = size
#             if exists:
#                 # Try to read first few rows
#                 try:
#                     df = pd.read_csv(path, nrows=3)
#                     data_info[f"{name}_columns"] = list(df.columns)
#                     data_info[f"{name}_sample_rows"] = len(df)
#                 except Exception as e:
#                     data_info[f"{name}_read_error"] = str(e)
        
#         return data_info
        
#     except Exception as e:
#         return {"error": str(e)}

# @app.function(image=image, volumes={MODEL_DIR: model_volume, DATA_DIR: data_volume})
# def list_data_directory():
#     """List contents of the data directory to debug dataset loading"""
#     try:
#         result = {}
        
#         if os.path.exists(DATA_DIR):
#             for root, dirs, files in os.walk(DATA_DIR):
#                 rel_path = root.replace(DATA_DIR, "").strip("/") or "root"
#                 result[rel_path] = {
#                     "path": root,
#                     "directories": dirs,
#                     "files": files,
#                     "csv_files": [f for f in files if f.endswith('.csv')]
#                 }
                
#                 # For CSV files, get basic info
#                 for csv_file in result[rel_path]["csv_files"]:
#                     csv_path = os.path.join(root, csv_file)
#                     try:
#                         df = pd.read_csv(csv_path, nrows=2)
#                         result[rel_path][f"{csv_file}_info"] = {
#                             "columns": list(df.columns),
#                             "shape": f"~{len(pd.read_csv(csv_path))} rows"
#                         }
#                     except Exception as e:
#                         result[rel_path][f"{csv_file}_error"] = str(e)
#         else:
#             result["error"] = f"Data directory {DATA_DIR} does not exist"
            
#         return result
#     except Exception as e:
#         return {"error": str(e)}
#         """Get detailed information about the trained model"""
#         import json
        
#         try:
#             # Read config
#             with open(f"{MODEL_DIR}/config.json", "r") as f:
#                 config = json.load(f)
            
#             # Read tokenizer config
#             with open(f"{MODEL_DIR}/tokenizer_config.json", "r") as f:
#                 tokenizer_config = json.load(f)
            
#             # Check for deployment info
#             deployment_info = {}
#             if os.path.exists(f"{MODEL_DIR}/deployment_package/deployment_info.json"):
#                 with open(f"{MODEL_DIR}/deployment_package/deployment_info.json", "r") as f:
#                     deployment_info = json.load(f)
            
#             return {
#                 "model_config": config,
#                 "tokenizer_config": tokenizer_config,
#                 "deployment_info": deployment_info,
#                 "model_size_mb": sum(os.path.getsize(os.path.join(MODEL_DIR, f)) 
#                                 for f in os.listdir(MODEL_DIR) 
#                                 if os.path.isfile(os.path.join(MODEL_DIR, f))) / (1024*1024)
#             }
        
#         except Exception as e:
#             return {"status": "error", "error": str(e)}

