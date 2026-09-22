import asyncio
import sys
sys.path.insert(0, r"C:\Users\Admin\.gemini\antigravity-ide\scratch\kaggle_project")
from sync_kaggle import execute_remote

phase2_train_code = """
import os, json, torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

print("=== Starting Phase 2: Full Multi-Domain Training (60 Records) ===")
print("Domains included:")
print("1. Business Email (10)")
print("2. Customer Support (10)")
print("3. Scheduling (10)")
print("4. General Assistance (10)")
print("5. Tanglish to Tamil (10)")
print("6. Safety Refusal (10)")

# Format all 60 records using ChatML template
formatted_data = []
for r in records:
    msgs = r.get("messages", [])
    if msgs:
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted_data.append({"text": chat_text})

dataset_60 = Dataset.from_list(formatted_data)
print(f"Created dataset with {len(dataset_60)} examples.")

# Save approved Phase 2 dataset to disk
export_phase2_path = "/kaggle/working/tamil_phase2_full_approved.jsonl"
with open(export_phase2_path, "w", encoding="utf-8") as f:
    for rec in records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\\n")
print(f"Saved Phase 2 dataset to {export_phase2_path}")

def tok_fn(b):
    tok = tokenizer(b["text"], truncation=True, max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_60 = dataset_60.map(tok_fn, batched=True)

# Training configuration for Phase 2
model.train()
output_phase2_dir = "/kaggle/working/tamil_qwen3_phase2_adapter"

training_args = TrainingArguments(
    output_dir=output_phase2_dir,
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
    train_dataset=tokenized_60,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
)

print(f"Launching Phase 2 Trainer on 60 samples for 3 epochs (45 optimization steps)...")
train_result = trainer.train()

trainer.save_model(output_phase2_dir)
tokenizer.save_pretrained(output_phase2_dir)
print(f"PHASE 2 SUCCESS! Model saved to {output_phase2_dir}")
print(f"Final Phase 2 Training Loss: {train_result.training_loss:.4f}")
"""

asyncio.run(execute_remote(phase2_train_code))
