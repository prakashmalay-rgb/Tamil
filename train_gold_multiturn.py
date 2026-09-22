"""
Gold Multi-Turn Distillation & Alignment Fine-Tuning Pipeline
Eliminates template placeholder artifacts and teaches conversational entity resolution.
Runs on Kaggle 2x Tesla T4 GPUs.
"""

import asyncio
import os
import sys
import sync_kaggle

async def main():
    print("=== Step 1: Uploading Gold Multi-Turn Corpus to Kaggle ===")
    local_gold = r"C:\Users\Admin\.gemini\antigravity-ide\scratch\kaggle_project\data\tamil_gold_multiturn.jsonl"
    await sync_kaggle.push_file(local_gold, "/kaggle/working")
    
    local_cl = r"C:\Users\Admin\.gemini\antigravity-ide\scratch\kaggle_project\data\tamil_crosslingual_mastery.jsonl"
    await sync_kaggle.push_file(local_cl, "/kaggle/working")

    print("\n=== Step 2: Running Gold Multi-Turn Alignment Training on Kaggle ===")
    train_code = r"""
import os, json, torch
from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. Load Gold Multi-Turn Sessions
gold_file = "/kaggle/working/tamil_gold_multiturn.jsonl"
gold_data = []
if os.path.exists(gold_file):
    with open(gold_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gold_data.append(json.loads(line))

print(f"Loaded {len(gold_data)} gold multi-turn conversational trees.")

# 2. Load Cross-Lingual Pairs
cl_file = "/kaggle/working/tamil_crosslingual_mastery.jsonl"
cl_data = []
if os.path.exists(cl_file):
    with open(cl_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cl_data.append(json.loads(line))

# 3. Load 200 clean Alpaca samples for broad knowledge retention
print("Loading diverse base knowledge...")
alpaca_ds = load_dataset('abhinand/tamil-alpaca', split='train[:200]')
alpaca_samples = []
for item in alpaca_ds:
    inst = item.get('instruction', '').strip()
    inp = item.get('input', '').strip()
    out = item.get('output', '').strip()
    if inst and out and len(out) > 20:
        u_msg = inst + ("\n\n" + inp if inp else "")
        alpaca_samples.append({
            "messages": [
                {"role": "system", "content": "You are a helpful, articulate, and native Tamil AI assistant. Always respond in pure, natural, grammatically correct Tamil."},
                {"role": "user", "content": u_msg},
                {"role": "assistant", "content": out}
            ]
        })

# Heavy weighting on gold multi-turn data (25x weight) to deeply re-align generation policy
combined_data = (gold_data * 25) + (cl_data * 10) + alpaca_samples
print(f"Total alignment corpus: {len(combined_data)} conversation trees.")

# 4. Format with Qwen ChatML
formatted_texts = []
for item in combined_data:
    msgs = item.get("messages", [])
    if msgs:
        chat_text = tokenizer.apply_chat_template(msgs, tokenize=False)
        formatted_texts.append({"text": chat_text})

train_dataset = Dataset.from_list(formatted_texts)

def tok_fn(b):
    tok = tokenizer(b["text"], truncation=True, max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok

tokenized_dataset = train_dataset.map(tok_fn, batched=True, remove_columns=["text"])
print(f"Tokenized dataset ready with {len(tokenized_dataset)} samples.")

# 5. Prepare Model for LoRA
if not hasattr(model, "peft_config"):
    try:
        model = prepare_model_for_kbit_training(model)
    except Exception as e:
        print("Note on prepare_model_for_kbit_training:", e)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    print("Attached fresh LoRA adapter matrices!")
else:
    print("Reusing existing LoRA adapter configuration.")

model.train()
output_dir = "/kaggle/working/tamil_qwen3_mastery_adapter"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=2,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=15,
    save_strategy="no",
    report_to="none",
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
)

print("Starting Gold Multi-Turn Alignment Training on 2x Tesla T4...")
train_res = trainer.train()

trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)
model.eval()
globals()['model'] = model
print(f"Gold Alignment Complete! Final Loss: {train_res.training_loss:.4f}")
print("Saved newly aligned adapter to:", output_dir)
"""
    await sync_kaggle.execute_remote(train_code)

if __name__ == "__main__":
    asyncio.run(main())
