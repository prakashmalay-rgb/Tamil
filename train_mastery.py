"""
Tamil LLM Mastery Fine-Tuning Script
Pushes cross-lingual dataset and executes QLoRA fine-tuning on Kaggle GPU.
"""

import asyncio
import os
import sys
import json
import base64
from sync_kaggle import execute_remote, push_file

async def run_mastery_pipeline():
    print("=== Step 1: Uploading Grammar & Cross-Lingual Datasets to Kaggle ===")
    local_data = r"C:\Users\Admin\.gemini\antigravity-ide\scratch\kaggle_project\data\tamil_crosslingual_mastery.jsonl"
    await push_file(local_data, "/kaggle/working")
    
    local_validator = r"C:\Users\Admin\.gemini\antigravity-ide\scratch\kaggle_project\grammar_validator.py"
    await push_file(local_validator, "/kaggle/working")

    print("\n=== Step 2: Launching Cross-Lingual & Grammar Fine-Tuning on Kaggle GPU ===")
    kaggle_train_script = """
import os, json, torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 1. Load both Phase 2 and Cross-Lingual Datasets
all_data = []

# Cross-lingual pairs
cl_file = "/kaggle/working/tamil_crosslingual_mastery.jsonl"
if os.path.exists(cl_file):
    with open(cl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

# Phase 2 pairs
p2_file = "/kaggle/working/tamil_phase2_full_approved.jsonl"
if os.path.exists(p2_file):
    with open(p2_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

print(f"Total training corpus combined: {len(all_data)} high-precision examples.")

# 2. Format with Qwen ChatML Template
formatted_texts = []
for item in all_data:
    msgs = item.get("messages", [])
    if msgs:
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted_texts.append({"text": chat_text})

mastery_dataset = Dataset.from_list(formatted_texts)

def tok_fn(b):
    tok = tokenizer(b["text"], truncation=True, max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_mastery = mastery_dataset.map(tok_fn, batched=True)

# 3. Train model with QLoRA
model.train()
output_mastery_dir = "/kaggle/working/tamil_qwen3_mastery_adapter"

training_args = TrainingArguments(
    output_dir=output_mastery_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mastery,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
)

print(f"Training Mastery Adapter on {len(tokenized_mastery)} samples...")
train_res = trainer.train()

trainer.save_model(output_mastery_dir)
tokenizer.save_pretrained(output_mastery_dir)
print(f"Mastery fine-tuning complete! Loss: {train_res.training_loss:.4f}")
print("Saved to:", output_mastery_dir)
"""
    await execute_remote(kaggle_train_script)

if __name__ == "__main__":
    asyncio.run(run_mastery_pipeline())
