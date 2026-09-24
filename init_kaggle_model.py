"""
Loads Qwen3-4B in 4-bit NF4 into Kaggle GPU memory and prepares it for live studio inference.
"""
import asyncio
import sync_kaggle

code = """
import os, sys, importlib
importlib.invalidate_caches()
import bitsandbytes as bnb
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

print("bitsandbytes version:", bnb.__version__)
BASE_MODEL = "Qwen/Qwen3-4B"
print(f"Loading {BASE_MODEL} in 4-bit NF4...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

globals()['model'] = model
globals()['tokenizer'] = tokenizer
print("SUCCESS: Qwen3-4B model and tokenizer are live in GPU memory!")
"""

async def main():
    await sync_kaggle.execute_remote(code)

if __name__ == "__main__":
    asyncio.run(main())
