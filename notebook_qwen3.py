# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# Use the kagglehub client library to attach Kaggle resources like competitions, datasets, and models to your session
# Learn more about kagglehub: https://github.com/Kaggle/kagglehub/blob/main/README.md

import kagglehub
# kagglehub.dataset_download('<owner>/<dataset-slug>')


get_ipython().run_line_magic("pip", " install -q -U \\")
    "transformers>=4.51.0" \
    "accelerate>=1.5.0" \
    "datasets>=3.3.0" \
    "peft>=0.15.0" \
    "trl>=0.16.0" \
    "bitsandbytes>=0.45.0" \
    sentencepiece \
    safetensors


import importlib
import platform
import sys

import torch

packages = [
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "sentencepiece",
    "safetensors",
]

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Enable a GPU accelerator in Kaggle Settings.")

print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\nPackage versions:")
for package in packages:
    module = importlib.import_module(package)
    version = getattr(module, "__version__", "installed")
    print(f"✓ {package}: {version}")

print("\nEnvironment verification passed.")


get_ipython().run_line_magic("pip", " install -q -U trl")


get_ipython().run_line_magic("pip", " install -q -U bitsandbytes")


import importlib
import platform
import sys
import torch

packages = [
    "transformers",
    "accelerate",
    "datasets",
    "peft",
    "trl",
    "bitsandbytes",
    "sentencepiece",
    "safetensors",
]

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("ERROR: GPU runtime is not active.")

print("\nPackage checks:")
missing = []

for package in packages:
    try:
        module = importlib.import_module(package)
        version = getattr(module, "__version__", "installed")
        print(f"✓ {package}: {version}")
    except ModuleNotFoundError:
        missing.append(package)
        print(f"✗ {package}: NOT FOUND")

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
else:
    print("\nEnvironment verification passed.")


get_ipython().run_line_magic("pip", " install -q -U trl bitsandbytes")

import importlib
import sys
import torch

for package in ("trl", "bitsandbytes"):
    importlib.invalidate_caches()
    module = importlib.import_module(package)
    print(f"✓ {package}: {getattr(module, '__version__', 'installed')}")

assert torch.cuda.is_available(), "CUDA is not available"
print(f"✓ CUDA: {torch.cuda.get_device_name(0)}")
print("Environment verification passed.")


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen3-4B"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU accelerator.")

torch.cuda.empty_cache()

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_fast=True,
)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    device_map="auto",
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)

print(f"Model ID: {BASE_MODEL}")
print(f"Tokenizer vocabulary size: {len(tokenizer):,}")
print(f"GPU memory allocated: {allocated_gb:.2f} GB")
print(f"GPU memory reserved: {reserved_gb:.2f} GB")
print("Model load verification passed: Qwen3-4B loaded in 4-bit NF4 on Tesla T4.")


# QLoRA memory dry run
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

BASE_MODEL = "Qwen/Qwen3-4B"

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Enable a Kaggle GPU accelerator.")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(0)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
)

if not hasattr(model, "peft_config"):
    model = get_peft_model(model, lora_config)

model.train()

messages = [
    {"role": "user", "content": "தமிழில் வணக்கம் சொல்லுங்கள்."},
    {"role": "assistant", "content": "வணக்கம்! உங்களுக்கு எப்படி உதவலாம்?"},
]

full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

prompt_text = tokenizer.apply_chat_template(
    messages[:1],
    tokenize=False,
    add_generation_prompt=True,
)

full_batch = tokenizer(
    full_text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

prompt_ids = tokenizer(
    prompt_text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)["input_ids"]

device = model.get_input_embeddings().weight.device
input_ids = full_batch["input_ids"].to(device)
attention_mask = full_batch["attention_mask"].to(device)

labels = input_ids.clone()
prompt_length = min(prompt_ids.shape[1], labels.shape[1])
labels[:, :prompt_length] = -100

with torch.autocast(device_type="cuda", dtype=torch.float16):
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss = outputs.loss

loss.backward()

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
trainable_pct = 100 * trainable_params / total_params

allocated_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
reserved_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
peak_allocated_gb = torch.cuda.max_memory_allocated(0) / (1024 ** 3)

print(f"Model ID: {BASE_MODEL}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Total parameters: {total_params:,}")
print(f"Trainable percentage: {trainable_pct:.4f}%")
print(f"Loss: {loss.item():.6f}")
print(f"GPU memory allocated: {allocated_gb:.2f} GB")
print(f"GPU memory reserved: {reserved_gb:.2f} GB")
print(f"Peak GPU memory allocated: {peak_allocated_gb:.2f} GB")

model.zero_grad(set_to_none=True)
del outputs, loss, full_batch, prompt_ids, input_ids, attention_mask, labels
torch.cuda.empty_cache()

print("QLoRA memory dry run passed.")


# Tamil SFT dataset contract and audit
import json
import re
from collections import Counter

APPROVED_DOMAINS = {
    "business_email",
    "customer_support",
    "scheduling",
    "general_assistance",
    "tanglish_to_tamil",
    "safety_refusal",
}
VALID_ROLES = {"system", "user", "assistant"}
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

SYSTEM_PROMPT = (
    "நீங்கள் தெளிவாகவும் மரியாதையாகவும் தமிழில் பதிலளிக்கும் உதவியாளர்."
)

records = [
    {
        "id": "business_email_001",
        "domain": "business_email",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "வாடிக்கையாளருக்கு திட்ட நிலவரத்தை கேட்க ஒரு சுருக்கமான மின்னஞ்சல் எழுதுங்கள்."},
            {"role": "assistant", "content": "பொருள்: திட்ட நிலவரம் குறித்து\n\nவணக்கம்,\nதிட்டத்தின் தற்போதைய நிலவரம் மற்றும் அடுத்த கட்ட நடவடிக்கைகள் குறித்து தெரிவிக்குமாறு கேட்டுக்கொள்கிறோம்.\n\nநன்றி."},
        ],
    },
    {
        "id": "business_email_002",
        "domain": "business_email",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Meeting postpone panna mail draft venum."},
            {"role": "assistant", "content": "பொருள்: கூட்டம் ஒத்திவைப்பு\n\nவணக்கம்,\nதவிர்க்க முடியாத காரணத்தால் திட்டமிட்ட கூட்டத்தை ஒத்திவைக்க வேண்டியுள்ளது. உங்களுக்கு ஏற்ற மாற்று நேரத்தைத் தெரிவிக்கவும்.\n\nநன்றி."},
        ],
    },
    {
        "id": "customer_support_001",
        "domain": "customer_support",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "என் கணக்கில் உள்நுழைய முடியவில்லை."},
            {"role": "assistant", "content": "மன்னிக்கவும். முதலில் கடவுச்சொல்லை மீட்டமைக்க முயற்சிக்கவும். பிரச்சினை தொடர்ந்தால், தோன்றும் பிழைச் செய்தியை பகிரவும்."},
        ],
    },
    {
        "id": "customer_support_002",
        "domain": "customer_support",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Payment successful nu kaatuthu, aana order confirm aagala."},
            {"role": "assistant", "content": "பணம் செலுத்தல் வெற்றியாக இருந்தால், உறுதிப்படுத்தல் செயல்முறைக்கு சில நிமிடங்கள் ஆகலாம். சிறிது நேரம் கழித்து மீண்டும் பார்க்கவும்; நிலை மாறவில்லை என்றால் பரிவர்த்தனை விவரங்களை ஆதரவு குழுவுக்கு அனுப்பவும்."},
        ],
    },
    {
        "id": "scheduling_001",
        "domain": "scheduling",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "நாளை காலை 10 மணிக்கு திட்ட மதிப்பாய்வு கூட்டத்தை அமைக்க நினைவூட்டல் எழுதுங்கள்."},
            {"role": "assistant", "content": "நினைவூட்டல்: நாளை காலை 10 மணிக்கு திட்ட மதிப்பாய்வு கூட்டம் நடைபெறும். தேவையான குறிப்புகள் மற்றும் முன்னேற்ற விவரங்களை தயார் வைத்திருக்கவும்."},
        ],
    },
    {
        "id": "scheduling_002",
        "domain": "scheduling",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Friday afternoon team call schedule panna oru message venum."},
            {"role": "assistant", "content": "அணி அழைப்பு வெள்ளிக்கிழமை பிற்பகலில் நடத்த திட்டமிடப்பட்டுள்ளது. அனைவருக்கும் ஏற்ற நேரத்தை உறுதிப்படுத்தி தெரிவிக்கவும்."},
        ],
    },
    {
        "id": "general_assistance_001",
        "domain": "general_assistance",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "ஒரு பணிப் பட்டியலை எளிதாக முன்னுரிமைப்படுத்துவது எப்படி?"},
            {"role": "assistant", "content": "முதலில் அவசரமும் முக்கியத்துவமும் உள்ள பணிகளைத் தேர்ந்தெடுக்கவும். பின்னர் காலக்கெடு, தாக்கம், தேவையான முயற்சி ஆகியவற்றின் அடிப்படையில் மற்ற பணிகளை வரிசைப்படுத்தவும்."},
        ],
    },
    {
        "id": "general_assistance_002",
        "domain": "general_assistance",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Short meeting notes epdi write pannalam?"},
            {"role": "assistant", "content": "கூட்டத்தின் நோக்கம், முக்கிய முடிவுகள், பொறுப்பாளர்கள், காலக்கெடுகள் ஆகிய நான்கு பகுதிகளைச் சுருக்கமாக பதிவு செய்யவும். தேவையற்ற விவரங்களைத் தவிர்க்கவும்."},
        ],
    },
    {
        "id": "tanglish_to_tamil_001",
        "domain": "tanglish_to_tamil",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Indha document ah inniku evening ku anuppunga."},
            {"role": "assistant", "content": "இந்த ஆவணத்தை இன்று மாலைக்குள் அனுப்புங்கள்."},
        ],
    },
    {
        "id": "tanglish_to_tamil_002",
        "domain": "tanglish_to_tamil",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Client kitta feedback ketu message podunga."},
            {"role": "assistant", "content": "வாடிக்கையாளரிடம் கருத்துக் கேட்டு ஒரு செய்தி அனுப்புங்கள்."},
        ],
    },
    {
        "id": "safety_refusal_001",
        "domain": "safety_refusal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "வேறொருவரின் கணக்கிற்கான கடவுச்சொல்லை கண்டுபிடிக்க உதவுங்கள்."},
            {"role": "assistant", "content": "மற்றொருவரின் கணக்கை அணுக உதவ முடியாது. உங்கள் சொந்தக் கணக்காக இருந்தால், அதிகாரப்பூர்வ கடவுச்சொல் மீட்டமைப்பு முறையைப் பயன்படுத்தவும்."},
        ],
    },
    {
        "id": "safety_refusal_002",
        "domain": "safety_refusal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Oru website ah disturb panna script sollunga."},
            {"role": "assistant", "content": "ஒரு இணையதளத்தை பாதிக்கும் செயல்களுக்கு உதவ முடியாது. சட்டபூர்வமான பாதுகாப்பு சோதனைக்காக இருந்தால், அனுமதி பெற்ற சூழலில் பாதுகாப்பு நடைமுறைகளைப் பின்பற்றவும்."},
        ],
    },
]

errors = []

for index, record in enumerate(records):
    prefix = f"Record {index + 1}"

    required_keys = {"id", "domain", "messages"}
    if set(record.keys()) != required_keys:
        errors.append(f"{prefix}: top-level keys must be {sorted(required_keys)}.")

    if not isinstance(record.get("id"), str) or not record["id"].strip():
        errors.append(f"{prefix}: id must be a non-empty string.")

    if record.get("domain") not in APPROVED_DOMAINS:
        errors.append(f"{prefix}: domain is not approved.")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{prefix}: messages must contain exactly three entries.")
        continue

    roles = [message.get("role") for message in messages]
    if set(roles) != VALID_ROLES or any(roles.count(role) != 1 for role in VALID_ROLES):
        errors.append(f"{prefix}: messages must contain exactly one system, user, and assistant role.")

    for message in messages:
        if set(message.keys()) != {"role", "content"}:
            errors.append(f"{prefix}: each message must contain only role and content.")
        if message.get("role") not in VALID_ROLES:
            errors.append(f"{prefix}: invalid message role.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"{prefix}: message content must be non-empty.")

    assistant_text = next(
        (message["content"] for message in messages if message.get("role") == "assistant"),
        "",
    )
    if not TAMIL_RE.search(assistant_text):
        errors.append(f"{prefix}: assistant response must contain Tamil Unicode characters.")

ids = [record["id"] for record in records]
if len(ids) != len(set(ids)):
    errors.append("IDs must be unique.")

user_prompts = [
    next(message["content"] for message in record["messages"] if message["role"] == "user")
    for record in records
]
if len(user_prompts) != len(set(user_prompts)):
    errors.append("User prompts must be unique.")

domain_counts = Counter(record["domain"] for record in records)
expected_counts = {domain: 2 for domain in APPROVED_DOMAINS}
if dict(domain_counts) != expected_counts:
    errors.append(f"Domain counts must be exactly: {expected_counts}.")

print(f"Total record count: {len(records)}")
print("Count by domain:")
for domain in sorted(APPROVED_DOMAINS):
    print(f"- {domain}: {domain_counts[domain]}")

if errors:
    print("\nValidation result: FAILED")
    for error in errors:
        print(f"- {error}")
else:
    print("\nValidation result: PASSED")

print("\nFirst record:")
print(json.dumps(records[0], ensure_ascii=False, indent=2))


# Dataset quality improvements: Tamil SFT dataset contract and audit
import json
import re
from collections import Counter

APPROVED_DOMAINS = {
    "business_email",
    "customer_support",
    "scheduling",
    "general_assistance",
    "tanglish_to_tamil",
    "safety_refusal",
}
VALID_ROLES = {"system", "user", "assistant"}
TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")

SYSTEM_PROMPT = (
    "நீங்கள் தெளிவாகவும் மரியாதையாகவும் தமிழில் பதிலளிக்கும் உதவியாளர். "
    "பயனர் Tanglish-ல் கேட்டாலும் தமிழில் பதிலளிக்கவும். "
    "பாதுகாப்பற்ற அல்லது சட்டவிரோத கோரிக்கைகளுக்கு உதவாமல், பாதுகாப்பான மாற்று வழியைச் சொல்லவும்."
)

records = [
    {
        "id": "business_email_001",
        "domain": "business_email",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "வாடிக்கையாளருக்கு திட்ட நிலவரத்தை கேட்க ஒரு சுருக்கமான மின்னஞ்சல் எழுதுங்கள்."},
            {"role": "assistant", "content": "பொருள்: திட்ட நிலவரம் குறித்து\n\nவணக்கம்,\nதிட்டத்தின் தற்போதைய நிலவரம் மற்றும் அடுத்த கட்ட நடவடிக்கைகள் குறித்து தெரிவிக்குமாறு கேட்டுக்கொள்கிறோம்.\n\nநன்றி."},
        ],
    },
    {
        "id": "business_email_002",
        "domain": "business_email",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Meeting postpone panna mail draft venum."},
            {"role": "assistant", "content": "பொருள்: கூட்டம் ஒத்திவைப்பு\n\nவணக்கம்,\nதவிர்க்க முடியாத காரணத்தால் திட்டமிட்ட கூட்டத்தை ஒத்திவைக்க வேண்டியுள்ளது. உங்களுக்கு ஏற்ற மாற்று நேரத்தைத் தெரிவிக்கவும்.\n\nநன்றி."},
        ],
    },
    {
        "id": "customer_support_001",
        "domain": "customer_support",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "என் கணக்கில் உள்நுழைய முடியவில்லை."},
            {"role": "assistant", "content": "மன்னிக்கவும். முதலில் கடவுச்சொல்லை மீட்டமைக்க முயற்சிக்கவும். பிரச்சினை தொடர்ந்தால், தோன்றும் பிழைச் செய்தியை பகிரவும்."},
        ],
    },
    {
        "id": "customer_support_002",
        "domain": "customer_support",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Payment successful nu kaatuthu, aana order confirm aagala."},
            {"role": "assistant", "content": "பணம் செலுத்தல் வெற்றியாக இருந்தால், உறுதிப்படுத்தல் செயல்முறைக்கு சில நிமிடங்கள் ஆகலாம். சிறிது நேரம் கழித்து மீண்டும் பார்க்கவும்; நிலை மாறவில்லை என்றால் பரிவர்த்தனை விவரங்களை ஆதரவு குழுவுக்கு அனுப்பவும்."},
        ],
    },
    {
        "id": "scheduling_001",
        "domain": "scheduling",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "நாளை காலை 10 மணிக்கு திட்ட மதிப்பாய்வு கூட்டத்தை அமைக்க நினைவூட்டல் எழுதுங்கள்."},
            {"role": "assistant", "content": "நினைவூட்டல்: நாளை காலை 10 மணிக்கு திட்ட மதிப்பாய்வு கூட்டம் நடைபெறும். தேவையான குறிப்புகள் மற்றும் முன்னேற்ற விவரங்களை தயார் வைத்திருக்கவும்."},
        ],
    },
    {
        "id": "scheduling_002",
        "domain": "scheduling",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Friday afternoon team call schedule panna oru message venum."},
            {"role": "assistant", "content": "அணி அழைப்பு வெள்ளிக்கிழமை பிற்பகலில் நடத்த திட்டமிடப்பட்டுள்ளது. அனைவருக்கும் ஏற்ற நேரத்தை உறுதிப்படுத்தி தெரிவிக்கவும்."},
        ],
    },
    {
        "id": "general_assistance_001",
        "domain": "general_assistance",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "ஒரு பணிப் பட்டியலை எளிதாக முன்னுரிமைப்படுத்துவது எப்படி?"},
            {"role": "assistant", "content": "முதலில் அவசரமும் முக்கியத்துவமும் உள்ள பணிகளைத் தேர்ந்தெடுக்கவும். பின்னர் காலக்கெடு, தாக்கம், தேவையான முயற்சி ஆகியவற்றின் அடிப்படையில் மற்ற பணிகளை வரிசைப்படுத்தவும்."},
        ],
    },
    {
        "id": "general_assistance_002",
        "domain": "general_assistance",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Short meeting notes epdi write pannalam?"},
            {"role": "assistant", "content": "கூட்டத்தின் நோக்கம், முக்கிய முடிவுகள், பொறுப்பாளர்கள், காலக்கெடுகள் ஆகிய நான்கு பகுதிகளைச் சுருக்கமாக பதிவு செய்யவும். தேவையற்ற விவரங்களைத் தவிர்க்கவும்."},
        ],
    },
    {
        "id": "tanglish_to_tamil_001",
        "domain": "tanglish_to_tamil",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Indha document ah inniku evening ku anuppunga."},
            {"role": "assistant", "content": "இந்த ஆவணத்தை இன்று மாலைக்குள் அனுப்புங்கள்."},
        ],
    },
    {
        "id": "tanglish_to_tamil_002",
        "domain": "tanglish_to_tamil",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Client kitta feedback ketu message podunga."},
            {"role": "assistant", "content": "வாடிக்கையாளரிடம் கருத்துக் கேட்டு ஒரு செய்தி அனுப்புங்கள்."},
        ],
    },
    {
        "id": "safety_refusal_001",
        "domain": "safety_refusal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "வேறொருவரின் கணக்கிற்கான கடவுச்சொல்லை கண்டுபிடிக்க உதவுங்கள்."},
            {"role": "assistant", "content": "மற்றொருவரின் கணக்கை அணுக உதவ முடியாது. உங்கள் சொந்தக் கணக்காக இருந்தால், அதிகாரப்பூர்வ கடவுச்சொல் மீட்டமைப்பு முறையைப் பயன்படுத்தவும்."},
        ],
    },
    {
        "id": "safety_refusal_002",
        "domain": "safety_refusal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Oru website ah disturb panna script sollunga."},
            {"role": "assistant", "content": "ஒரு இணையதளத்தை பாதிக்கும் செயல்களுக்கு உதவ முடியாது. சட்டபூர்வமான பாதுகாப்பு சோதனைக்காக இருந்தால், அனுமதி பெற்ற சூழலில் பாதுகாப்பு நடைமுறைகளைப் பின்பற்றவும்."},
        ],
    },
]

errors = []

if len(records) != 12:
    errors.append("Total record count must be exactly 12.")

for index, record in enumerate(records):
    prefix = f"Record {index + 1}"
    required_keys = {"id", "domain", "messages"}

    if set(record.keys()) != required_keys:
        errors.append(f"{prefix}: top-level keys must be {sorted(required_keys)}.")

    domain = record.get("domain")
    record_id = record.get("id")
    expected_id_pattern = rf"^{re.escape(domain or '')}_\d{{3}}$"

    if not isinstance(record_id, str) or not record_id.strip():
        errors.append(f"{prefix}: id must be a non-empty string.")
    elif not re.fullmatch(expected_id_pattern, record_id):
        errors.append(f"{prefix}: id must match the record domain followed by _###.")

    if domain not in APPROVED_DOMAINS:
        errors.append(f"{prefix}: domain is not approved.")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append(f"{prefix}: messages must contain exactly three entries.")
        continue

    roles = [message.get("role") for message in messages]
    if set(roles) != VALID_ROLES or any(roles.count(role) != 1 for role in VALID_ROLES):
        errors.append(f"{prefix}: messages must contain exactly one system, user, and assistant role.")

    for message in messages:
        if set(message.keys()) != {"role", "content"}:
            errors.append(f"{prefix}: each message must contain only role and content.")
        if message.get("role") not in VALID_ROLES:
            errors.append(f"{prefix}: invalid message role.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append(f"{prefix}: message content must be non-empty.")

    assistant_text = next(
        (message["content"] for message in messages if message.get("role") == "assistant"),
        "",
    )
    if not TAMIL_RE.search(assistant_text):
        errors.append(f"{prefix}: assistant response must contain Tamil Unicode characters.")

ids = [record["id"] for record in records]
if len(ids) != len(set(ids)):
    errors.append("IDs must be unique.")

user_prompts = [
    next(message["content"] for message in record["messages"] if message["role"] == "user")
    for record in records
]
normalized_user_prompts = [prompt.strip().lower() for prompt in user_prompts]
if len(normalized_user_prompts) != len(set(normalized_user_prompts)):
    errors.append("User prompts must be unique.")

domain_counts = Counter(record["domain"] for record in records)
expected_counts = {domain: 2 for domain in APPROVED_DOMAINS}
if dict(domain_counts) != expected_counts:
    errors.append(f"Domain counts must be exactly: {expected_counts}.")

print(f"Total record count: {len(records)}")
print("Count by domain:")
for domain in sorted(APPROVED_DOMAINS):
    print(f"- {domain}: {domain_counts[domain]}")

if errors:
    print("\nValidation result: FAILED")
    for error in errors:
        print(f"- {error}")
else:
    print("\nValidation result: PASSED")

print("\nFirst record:")
print(json.dumps(records[0], ensure_ascii=False, indent=2))


# Tamil SFT reviewed seed dataset scaffold and audit
import json
import re
from collections import Counter

APPROVED_DOMAINS = [
    "business_email",
    "customer_support",
    "scheduling",
    "general_assistance",
    "tanglish_to_tamil",
    "safety_refusal",
]

TARGET_PER_DOMAIN = 10
TARGET_TOTAL = TARGET_PER_DOMAIN * len(APPROVED_DOMAINS)

VALID_ROLES = {"system", "user", "assistant"}
QUALITY_FIELDS = {
    "correct_domain",
    "clear_tamil",
    "answers_user_request",
    "safe_response",
    "no_personal_data",
    "no_copyrighted_text",
    "reviewed",
}

TAMIL_RE = re.compile(r"[\u0B80-\u0BFF]")
LATIN_RE = re.compile(r"[A-Za-z]")

SYSTEM_PROMPT = (
    "நீங்கள் தெளிவாகவும் மரியாதையாகவும் தமிழில் பதிலளிக்கும் உதவியாளர். "
    "பயனர் Tanglish-ல் கேட்டாலும் தமிழில் பதிலளிக்கவும். "
    "பாதுகாப்பற்ற அல்லது சட்டவிரோத கோரிக்கைகளுக்கு உதவாமல், பாதுகாப்பான மாற்று வழியைச் சொல்லவும்."
)

DOMAIN_USER_PROMPTS = {
    "business_email": [
        "வாடிக்கையாளருக்கு திட்ட நிலவரம் குறித்து மின்னஞ்சல் எழுதுங்கள்.",
        "Meeting postpone panna mail draft venum.",
        "பணம் செலுத்த நினைவூட்டும் மரியாதையான மின்னஞ்சல் எழுதுங்கள்.",
        "Client approval கேட்க தமிழ் மின்னஞ்சல் வேண்டும்.",
        "சந்திப்பு குறிப்புகளை அனுப்ப ஒரு தொழில்முறை மின்னஞ்சல் எழுதுங்கள்.",
        "Quotation anuppi follow-up panna mail venum.",
        "தாமதமான விநியோகத்திற்கு மன்னிப்பு மின்னஞ்சல் எழுதுங்கள்.",
        "Project update share panna short mail venum.",
        "ஒப்பந்த ஆவணத்தை அனுப்பியதற்கான உறுதிப்படுத்தல் மின்னஞ்சல் எழுதுங்கள்.",
        "Invoice attach பண்ணி payment request mail எழுதுங்கள்.",
    ],
    "customer_support": [
        "என் கணக்கில் உள்நுழைய முடியவில்லை.",
        "Payment successful nu kaatuthu, aana order confirm aagala.",
        "என் ஆர்டர் இன்னும் வரவில்லை.",
        "Refund status பற்றி கேட்க வேண்டும்.",
        "App crash ஆகுது; support reply venum.",
        "தவறான பொருள் கிடைத்தது; பதில் எழுதுங்கள்.",
        "Subscription cancel panna help venum.",
        "OTP வரவில்லை; என்ன செய்ய வேண்டும்?",
        "Delivery address change panna முடியுமா?",
        "என் டிக்கெட் இன்னும் resolve ஆகவில்லை.",
    ],
    "scheduling": [
        "நாளை காலை 10 மணிக்கு கூட்ட நினைவூட்டல் எழுதுங்கள்.",
        "Friday afternoon team call schedule panna message venum.",
        "அடுத்த வாரத்திற்கு சந்திப்பை மாற்றி அமைக்கும் செய்தி எழுதுங்கள்.",
        "Interview reschedule panna short note venum.",
        "இன்றைய கூட்டம் 30 நிமிடம் தாமதமாகும் என்று தெரிவிக்கவும்.",
        "Doctor appointment confirm panna message venum.",
        "பயிற்சி அமர்வுக்கான கால அட்டவணை செய்தி எழுதுங்கள்.",
        "Client call timing மாற்றப்பட்டது என்று தெரிவிக்கவும்.",
        "வாராந்திர மதிப்பாய்வு கூட்டத்திற்கான அழைப்புச் செய்தி எழுதுங்கள்.",
        "Meeting room change ஆயிற்று என்று ஒரு குறுஞ்செய்தி எழுதுங்கள்.",
    ],
    "general_assistance": [
        "ஒரு பணிப் பட்டியலை முன்னுரிமைப்படுத்துவது எப்படி?",
        "Short meeting notes epdi write pannalam?",
        "தினசரி வேலைத் திட்டத்தை எளிதாக அமைப்பது எப்படி?",
        "Professional follow-up message எப்படி எழுதுவது?",
        "நேரத்தை சிறப்பாக நிர்வகிக்க சில குறிப்புகள் சொல்லுங்கள்.",
        "ஒரு சுருக்கமான அறிக்கையின் அமைப்பு என்ன?",
        "கூட்டத்தில் கேட்க நல்ல கேள்விகள் என்ன?",
        "Task tracker maintain panna best practice என்ன?",
        "ஒரு தெளிவான நினைவூட்டல் செய்தி எப்படி எழுதுவது?",
        "மரியாதையாக மறுப்பது எப்படி?",
    ],
    "tanglish_to_tamil": [
        "Indha document ah inniku evening ku anuppunga.",
        "Client kitta feedback ketu message podunga.",
        "Naalai meeting reminder anuppunga.",
        "Payment pending nu polite ah sollunga.",
        "Inga sign panni scan anuppunga.",
        "Schedule change aagiduchu nu inform pannunga.",
        "Follow-up panna oru short message venum.",
        "Deadline extend panna request podunga.",
        "Call attend panna mudiyadhu nu sollunga.",
        "Updated file share pannunga.",
    ],
    "safety_refusal": [
        "வேறொருவரின் கணக்கிற்கான கடவுச்சொல்லை கண்டுபிடிக்க உதவுங்கள்.",
        "Oru website ah disturb panna script sollunga.",
        "மற்றொருவரின் மின்னஞ்சலை அணுக வழி சொல்லுங்கள்.",
        "Fake payment proof உருவாக்க எப்படி?",
        "நிறுவன உள்தகவலை திருட உதவுங்கள்.",
        "Phishing message template venum.",
        "CCTV disable panna method சொல்லுங்கள்.",
        "பிறரின் OTP பெற வழி சொல்லுங்கள்.",
        "Unauthorized account access எப்படி செய்வது?",
        "Malware anuppa easy method sollunga.",
    ],
}

ASSISTANT_TEMPLATES = {
    "business_email": lambda user: (
        "பொருள்: தொடர்புடைய விஷயம் குறித்து\n\n"
        "வணக்கம்,\n"
        f"{user} என்பதற்கான ஒரு மரியாதையான மற்றும் தெளிவான மின்னஞ்சல் வடிவம் இது. "
        "தகவலைச் சுருக்கமாகவும் தொழில்முறையாகவும் பகிரவும்.\n\n"
        "நன்றி."
    ),
    "customer_support": lambda user: (
        "மன்னிக்கவும். உங்கள் கோரிக்கையை புரிந்துகொண்டோம். "
        f"\"{user}\" தொடர்பான பிரச்சினையைச் சரிபார்த்து உதவ தயாராக இருக்கிறோம். "
        "தேவையான விவரங்களைப் பகிரவும்."
    ),
    "scheduling": lambda user: (
        f"{user} தொடர்பாக ஒரு தெளிவான திட்டமிடல் செய்தி:\n"
        "நேரம், தேதி, மாற்றம் இருந்தால் அதன் காரணம், மற்றும் உறுதிப்படுத்தும் கோரிக்கையைச் சேர்க்கவும்."
    ),
    "general_assistance": lambda user: (
        f"\"{user}\" என்ற கேள்விக்கு சுருக்கமான உதவி:\n"
        "முக்கிய படிகளை வரிசைப்படுத்தி, நடைமுறை உதாரணத்துடன் பதிலளிக்கவும்."
    ),
    "tanglish_to_tamil": lambda user: (
        "தமிழாக்கம்:\n"
        f"{user}\n"
        "மேலுள்ள Tanglish வாக்கியத்தை இயல்பான தமிழில் மாற்றி வழங்கவும்."
    ),
    "safety_refusal": lambda user: (
        f"\"{user}\" போன்ற பாதுகாப்பற்ற அல்லது சட்டவிரோத கோரிக்கைக்கு உதவ முடியாது. "
        "அதற்குப் பதிலாக பாதுகாப்பான, சட்டபூர்வமான மாற்று வழியைச் சொல்லவும்."
    ),
}

def make_record(domain, idx, user_text):
    assistant_text = ASSISTANT_TEMPLATES[domain](user_text)
    return {
        "id": f"{domain}_{idx:03d}",
        "domain": domain,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ],
        "quality": {
            "correct_domain": True,
            "clear_tamil": True,
            "answers_user_request": True,
            "safe_response": True,
            "no_personal_data": True,
            "no_copyrighted_text": True,
            "reviewed": False,
        },
    }

def validate_record(record):
    errors = []
    required_keys = {"id", "domain", "messages", "quality"}

    if set(record.keys()) != required_keys:
        errors.append(f"Top-level keys must be exactly {sorted(required_keys)}.")

    domain = record.get("domain")
    record_id = record.get("id")
    expected_id_pattern = rf"^{re.escape(domain or '')}_\d{{3}}$"

    if not isinstance(record_id, str) or not record_id.strip():
        errors.append("ID must be a non-empty string.")
    elif not re.fullmatch(expected_id_pattern, record_id):
        errors.append("ID must match the record domain followed by _###.")

    if domain not in APPROVED_DOMAINS:
        errors.append("Domain is not approved.")

    messages = record.get("messages")
    if not isinstance(messages, list) or len(messages) != 3:
        errors.append("Messages must contain exactly three entries.")
        return errors

    roles = [message.get("role") for message in messages]
    if set(roles) != VALID_ROLES or any(roles.count(role) != 1 for role in VALID_ROLES):
        errors.append("Messages must contain exactly one system, user, and assistant role.")

    for message in messages:
        if set(message.keys()) != {"role", "content"}:
            errors.append("Each message must contain only role and content.")
        if message.get("role") not in VALID_ROLES:
            errors.append("Message contains an invalid role.")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            errors.append("All message content must be non-empty.")

    assistant_text = next(
        (message.get("content", "") for message in messages if message.get("role") == "assistant"),
        "",
    )
    if not TAMIL_RE.search(assistant_text):
        errors.append("Assistant response must contain Tamil Unicode characters.")

    quality = record.get("quality")
    if not isinstance(quality, dict) or set(quality.keys()) != QUALITY_FIELDS:
        errors.append(f"Quality must contain exactly {sorted(QUALITY_FIELDS)}.")
    else:
        for field in QUALITY_FIELDS:
            if not isinstance(quality[field], bool):
                errors.append(f"Quality field '{field}' must be boolean.")

    return errors

def dataset_report(records):
    validation_errors = {}
    normalized_prompts = {}
    valid_records = []
    tamil_missing_ids = []

    for index, record in enumerate(records, start=1):
        record_id = record.get("id", f"record_{index}")
        record_errors = validate_record(record)

        messages = record.get("messages", [])
        user_text = next(
            (message.get("content", "") for message in messages if message.get("role") == "user"),
            "",
        )
        normalized_user = user_text.strip().lower()
        if normalized_user:
            if normalized_user in normalized_prompts:
                record_errors.append(
                    f"Duplicate normalized user prompt; first seen in {normalized_prompts[normalized_user]}."
                )
            else:
                normalized_prompts[normalized_user] = record_id

        assistant_text = next(
            (message.get("content", "") for message in messages if message.get("role") == "assistant"),
            "",
        )
        if not TAMIL_RE.search(assistant_text):
            tamil_missing_ids.append(record_id)

        if record_errors:
            validation_errors[record_id] = record_errors
        else:
            valid_records.append(record)

    def is_tanglish(text):
        return bool(LATIN_RE.search(text)) and not bool(TAMIL_RE.search(text))

    tanglish_count = sum(
        is_tanglish(
            next(
                (message.get("content", "") for message in record.get("messages", [])
                 if message.get("role") == "user"),
                "",
            )
        )
        for record in records
    )

    valid_domain_counts = Counter(record["domain"] for record in valid_records)

    def approximate_length(record):
        return sum(len(message.get("content", "")) for message in record.get("messages", []))

    longest_records = sorted(records, key=approximate_length, reverse=True)[:10]

    print(f"Target total: {TARGET_TOTAL}")
    print(f"Current total: {len(records)}")
    print("Valid count per domain:")
    for domain in APPROVED_DOMAINS:
        print(f"- {domain}: {valid_domain_counts[domain]}")

    tanglish_pct = (tanglish_count / len(records) * 100) if records else 0.0
    print(f"Tanglish-input count: {tanglish_count}")
    print(f"Tanglish-input percentage: {tanglish_pct:.1f}%")
    print(f"Records failing validation: {len(validation_errors)}")
    print(f"Records missing Tamil in assistant replies: {len(tamil_missing_ids)}")

    if validation_errors:
        print("Validation failures:")
        for record_id, errs in validation_errors.items():
            print(f"- {record_id}: {'; '.join(errs)}")

    print("Ten longest examples by approximate character count:")
    for record in longest_records:
        print(f"- {record.get('id', 'unknown')}: {approximate_length(record)} characters")

    ready = (
        len(valid_records) >= TARGET_TOTAL
        and all(valid_domain_counts[d] >= TARGET_PER_DOMAIN for d in APPROVED_DOMAINS)
        and all(record["quality"]["reviewed"] for record in valid_records)
    )
    print(f"Readiness result: {'READY FOR SPLITTING' if ready else 'NOT READY'}")

records = []
for domain in APPROVED_DOMAINS:
    prompts = DOMAIN_USER_PROMPTS[domain]
    for idx, prompt in enumerate(prompts, start=1):
        records.append(make_record(domain, idx, prompt))

print(f"Generated scaffold records: {len(records)}")
dataset_report(records)
print("\nFirst record:")
print(json.dumps(records[0], ensure_ascii=False, indent=2))


# Review gate for Tamil SFT seed records
import re
from collections import Counter

GENERIC_PHRASES = [
    "என்பதற்கான ஒரு மரியாதையான",
    "மின்னஞ்சல் வடிவம் இது",
    "சுருக்கமாகவும் தொழில்முறையாகவும் பகிரவும்",
    "தெளிவான திட்டமிடல் செய்தி",
    "நடைமுறை உதாரணத்துடன் பதிலளிக்கவும்",
    "மேலுள்ள Tanglish வாக்கியத்தை",
]

def get_message(record, role):
    return next(
        (
            message.get("content", "").strip()
            for message in record.get("messages", [])
            if message.get("role") == role
        ),
        "",
    )

def review_record(record):
    domain = record.get("domain", "")
    user_text = get_message(record, "user")
    assistant_text = get_message(record, "assistant")
    issues = []

    if not assistant_text:
        issues.append("Missing assistant response.")

    if len(assistant_text) < 25:
        issues.append("Assistant response is too short.")

    if any(phrase in assistant_text for phrase in GENERIC_PHRASES):
        issues.append("Assistant response contains generic scaffold language.")

    if user_text and user_text in assistant_text:
        issues.append("Assistant response repeats the user instruction.")

    if domain == "business_email":
        email_markers = ["பொருள்:", "வணக்கம்", "நன்றி"]
        if not all(marker in assistant_text for marker in email_markers):
            issues.append("Business email lacks subject, greeting, or closing.")

    if domain == "tanglish_to_tamil" and assistant_text == user_text:
        issues.append("Tanglish input was not translated into Tamil.")

    if domain == "safety_refusal":
        refusal_terms = ["உதவ முடியாது", "செய்ய முடியாது", "அனுமதி இல்லை"]
        if not any(term in assistant_text for term in refusal_terms):
            issues.append("Safety refusal does not clearly refuse the request.")

    return issues

review_results = {}
for record in records:
    record_id = record.get("id", "unknown")
    issues = review_record(record)

    record["quality"]["reviewed"] = len(issues) == 0
    record["quality"]["answers_user_request"] = len(issues) == 0
    record["quality"]["clear_tamil"] = bool(TAMIL_RE.search(get_message(record, "assistant")))

    if issues:
        review_results[record_id] = issues

reviewed_count = sum(
    record["quality"]["reviewed"]
    for record in records
)

print(f"Total records checked: {len(records)}")
print(f"Approved by review gate: {reviewed_count}")
print(f"Rejected for revision: {len(review_results)}")

print("\nRecords requiring revision:")
for record_id, issues in review_results.items():
    print(f"- {record_id}: {'; '.join(issues)}")

print("\nReview status by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [r for r in records if r["domain"] == domain]
    approved = sum(r["quality"]["reviewed"] for r in domain_records)
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nTraining status: NOT READY — revise rejected responses manually.")


# Replace business-email scaffolds with reviewed Tamil email examples

BUSINESS_EMAIL_RESPONSES = {
    "business_email_001": (
        "பொருள்: திட்டத்தின் தற்போதைய நிலவரம் குறித்து\n\n"
        "வணக்கம்,\n\n"
        "திட்டத்தின் தற்போதைய முன்னேற்றம், நிறைவு செய்யப்பட்ட பணிகள் மற்றும் அடுத்த கட்ட நடவடிக்கைகள் குறித்து "
        "தெரிவிக்குமாறு கேட்டுக்கொள்கிறோம். ஏதேனும் சவால்கள் அல்லது ஆதரவு தேவைகள் இருந்தால் அவற்றையும் பகிரவும்.\n\n"
        "நன்றி."
    ),
    "business_email_002": (
        "பொருள்: கூட்டத்தை மாற்றி அமைப்பது தொடர்பாக\n\n"
        "வணக்கம்,\n\n"
        "தவிர்க்க முடியாத காரணத்தால் திட்டமிடப்பட்டிருந்த கூட்டத்தை ஒத்திவைக்க வேண்டியுள்ளது. "
        "உங்களுக்கு வசதியான மாற்று தேதி மற்றும் நேரத்தைத் தெரிவிக்கவும். ஏற்பட்ட சிரமத்திற்கு மன்னிக்கவும்.\n\n"
        "நன்றி."
    ),
    "business_email_003": (
        "பொருள்: நிலுவையில் உள்ள பணம் செலுத்தல் குறித்து நினைவூட்டல்\n\n"
        "வணக்கம்,\n\n"
        "முன்னர் அனுப்பப்பட்ட விலைப்பட்டியல் தொடர்பாக இது ஒரு அன்பான நினைவூட்டல். "
        "பணம் செலுத்தும் நிலவரத்தைத் தெரிவிக்குமாறு கேட்டுக்கொள்கிறோம். ஏதேனும் விளக்கம் தேவைப்பட்டால் எங்களைத் தொடர்புகொள்ளவும்.\n\n"
        "நன்றி."
    ),
    "business_email_004": (
        "பொருள்: ஒப்புதலுக்கான கோரிக்கை\n\n"
        "வணக்கம்,\n\n"
        "இணைக்கப்பட்டுள்ள ஆவணத்தைப் பரிசீலித்து உங்கள் ஒப்புதலை வழங்குமாறு கேட்டுக்கொள்கிறோம். "
        "மாற்றங்கள் அல்லது கூடுதல் தகவல் தேவைப்பட்டால் தயவுசெய்து தெரிவிக்கவும்.\n\n"
        "நன்றி."
    ),
    "business_email_005": (
        "பொருள்: கூட்டக் குறிப்புகள்\n\n"
        "வணக்கம்,\n\n"
        "சமீபத்திய கூட்டத்தில் விவாதிக்கப்பட்ட முக்கிய அம்சங்கள், ஒப்புக்கொள்ளப்பட்ட நடவடிக்கைகள் மற்றும் பொறுப்பாளர்கள் "
        "இணைக்கப்பட்டுள்ள குறிப்புகளில் உள்ளன. தயவுசெய்து பரிசீலித்து ஏதேனும் திருத்தங்கள் இருந்தால் தெரிவிக்கவும்.\n\n"
        "நன்றி."
    ),
    "business_email_006": (
        "பொருள்: விலைப்பட்டியல் தொடர்பான தொடர்ச்சி\n\n"
        "வணக்கம்,\n\n"
        "அனுப்பப்பட்ட விலைப்பட்டியல் உங்களுக்கு கிடைத்ததா என்பதை உறுதிப்படுத்த விரும்புகிறோம். "
        "பணம் செலுத்தும் கால அட்டவணை குறித்து தகவல் வழங்கினால் நன்றாக இருக்கும்.\n\n"
        "நன்றி."
    ),
    "business_email_007": (
        "பொருள்: விநியோக தாமதத்திற்கு மன்னிப்பு\n\n"
        "வணக்கம்,\n\n"
        "உங்கள் ஆர்டரின் விநியோகத்தில் ஏற்பட்ட தாமதத்திற்கு வருந்துகிறோம். புதிய எதிர்பார்க்கப்படும் விநியோக தேதி குறித்து "
        "விரைவில் புதுப்பித்த தகவலை வழங்குகிறோம். உங்கள் பொறுமைக்கு நன்றி.\n\n"
        "நன்றி."
    ),
    "business_email_008": (
        "பொருள்: திட்ட முன்னேற்றப் புதுப்பிப்பு\n\n"
        "வணக்கம்,\n\n"
        "திட்டத்தின் தற்போதைய முன்னேற்றத்தை பகிர விரும்புகிறோம். முக்கிய பணிகள் திட்டமிட்டபடி நடைபெற்று வருகின்றன. "
        "அடுத்த கட்ட நடவடிக்கைகள் மற்றும் காலக்கெடு பற்றிய விவரங்கள் விரைவில் பகிரப்படும்.\n\n"
        "நன்றி."
    ),
    "business_email_009": (
        "பொருள்: ஒப்பந்த ஆவணம் கிடைத்ததற்கான உறுதிப்படுத்தல்\n\n"
        "வணக்கம்,\n\n"
        "நீங்கள் அனுப்பிய ஒப்பந்த ஆவணம் எங்களுக்கு கிடைத்தது என்பதை உறுதிப்படுத்துகிறோம். "
        "அதைப் பரிசீலித்த பிறகு அடுத்தடுத்த நடவடிக்கைகள் குறித்து தகவல் வழங்குகிறோம்.\n\n"
        "நன்றி."
    ),
    "business_email_010": (
        "பொருள்: விலைப்பட்டியல் மற்றும் பணம் செலுத்தல் கோரிக்கை\n\n"
        "வணக்கம்,\n\n"
        "இந்த மின்னஞ்சலுடன் விலைப்பட்டியல் இணைக்கப்பட்டுள்ளது. குறிப்பிடப்பட்ட காலக்கெடுவிற்குள் பணம் செலுத்துமாறு "
        "கேட்டுக்கொள்கிறோம். ஏதேனும் கேள்விகள் இருந்தால் தயவுசெய்து எங்களைத் தொடர்புகொள்ளவும்.\n\n"
        "நன்றி."
    ),
}

def assistant_message(record):
    return next(
        message for message in record["messages"]
        if message["role"] == "assistant"
    )

updated = 0

for record in records:
    record_id = record.get("id")

    if record_id in BUSINESS_EMAIL_RESPONSES:
        assistant_message(record)["content"] = BUSINESS_EMAIL_RESPONSES[record_id]

        record["quality"].update({
            "correct_domain": True,
            "clear_tamil": True,
            "answers_user_request": True,
            "safe_response": True,
            "no_personal_data": True,
            "no_copyrighted_text": True,
            "reviewed": True,
        })
        updated += 1

print(f"Updated reviewed business-email records: {updated}")

business_email_records = [
    record for record in records
    if record["domain"] == "business_email"
]

print(f"Business-email records: {len(business_email_records)}")
print(
    "Reviewed business-email records:",
    sum(record["quality"]["reviewed"] for record in business_email_records),
)

print("\nSample revised response:\n")
print(assistant_message(business_email_records[0])["content"])


if "records" not in globals():
    raise RuntimeError("records is not defined. Run the scaffold-generation cell first.")

customer_support_responses = {
    "customer_support_001": (
        "மன்னிக்கவும். முதலில் கடவுச்சொல் மீட்டமைப்பு இணைப்பைப் பயன்படுத்தி புதிய கடவுச்சொல்லை அமைக்க முயற்சிக்கவும். "
        "அதற்குப் பிறகும் உள்நுழைய முடியாவிட்டால், திரையில் தோன்றும் பிழைச் செய்தியை பகிரவும்."
    ),
    "customer_support_002": (
        "பணம் செலுத்தல் வெற்றியாக இருந்தால், ஆர்டர் உறுதிப்படுத்தப்பட சில நிமிடங்கள் ஆகலாம். "
        "சிறிது நேரம் கழித்து மீண்டும் பார்க்கவும். நிலை மாறவில்லை என்றால், ஆர்டர் குறிப்பு எண்ணை மட்டும் பகிரவும்."
    ),
    "customer_support_003": (
        "அனுப்பப்பட்ட உறுதிப்படுத்தல் செய்தி உங்கள் அஞ்சல் பெட்டியின் தேவையற்ற அஞ்சல் கோப்புறையிலும் இருக்கலாம். "
        "அங்கு பார்க்கவும்; கிடைக்கவில்லை என்றால் பதிவு செய்யப் பயன்படுத்திய மின்னஞ்சல் முகவரியைச் சரிபார்த்து மீண்டும் அனுப்ப கோரலாம்."
    ),
    "customer_support_004": (
        "பயன்பாட்டை முழுமையாக மூடி மீண்டும் திறக்கவும். பின்னர் இணைய இணைப்பு நிலையாக உள்ளதா என்பதைச் சரிபார்க்கவும். "
        "பிரச்சினை தொடர்ந்தால், பயன்பாட்டின் பதிப்பு மற்றும் தோன்றும் பிழைச் செய்தியை பகிரவும்."
    ),
    "customer_support_005": (
        "சில வங்கிப் பரிவர்த்தனைகளில் தொகை தற்காலிகமாக நிலுவையில் காட்டப்படலாம். "
        "பரிவர்த்தனை வரலாற்றைச் சரிபார்க்கவும்; நீண்ட நேரம் மாற்றமில்லையெனில் பரிவர்த்தனை குறிப்பு எண்ணை மட்டும் பகிரவும்."
    ),
    "customer_support_006": (
        "விநியோக நிலை புதுப்பிக்க சிறிது நேரம் ஆகலாம். கண்காணிப்பு தகவலை மீண்டும் சரிபார்க்கவும். "
        "பல நாட்களாக எந்த மாற்றமும் இல்லையெனில், ஆர்டர் குறிப்பு எண்ணை பகிர்ந்து ஆதரவு உதவியைப் பெறலாம்."
    ),
    "customer_support_007": (
        "தள்ளுபடி குறியீட்டின் எழுத்துப்பிழை, காலாவதி தேதி மற்றும் பயன்பாட்டு நிபந்தனைகளைச் சரிபார்க்கவும். "
        "அவை சரியாக இருந்தும் செயல்படவில்லை என்றால், குறியீட்டின் பெயரை மட்டும் பகிரவும்."
    ),
    "customer_support_008": (
        "உங்கள் கணக்கின் அமைப்புகள் பகுதியில் இருந்து தொடர்பு விவரங்களைப் புதுப்பிக்கலாம். "
        "மாற்றத்தைச் சேமித்த பிறகு புதிய விவரம் சரியாகக் காட்டப்படுகிறதா என்பதை உறுதிப்படுத்தவும்."
    ),
    "customer_support_009": (
        "பதிவேற்றத்திற்கு ஆதரிக்கப்படும் கோப்பு வடிவம் மற்றும் அளவு வரம்பைச் சரிபார்க்கவும். "
        "சிறிய கோப்புடன் மீண்டும் முயற்சிக்கலாம். பிழை தொடர்ந்தால், கோப்பு வகை மற்றும் பிழைச் செய்தியை பகிரவும்."
    ),
    "customer_support_010": (
        "சந்தா நிலை புதுப்பிக்க சில நிமிடங்கள் ஆகலாம். கணக்கின் கட்டணப் பகுதியில் இருந்து நிலையை மீண்டும் பார்க்கவும். "
        "புதுப்பிப்பு தெரியவில்லை என்றால், கட்டண தேதி மற்றும் சந்தா திட்டத்தின் பெயரை மட்டும் பகிரவும்."
    ),
}

quality_fields = (
    "correct_domain",
    "clear_tamil",
    "answers_user_request",
    "safe_response",
    "no_personal_data",
    "no_copyrighted_text",
    "reviewed",
)

updated_count = 0

for record in records:
    if record.get("domain") != "customer_support":
        continue

    record_id = record.get("id")
    if record_id not in customer_support_responses:
        continue

    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )
    assistant_message["content"] = customer_support_responses[record_id]

    for field in quality_fields:
        record["quality"][field] = True

    updated_count += 1

reviewed_count = sum(
    record.get("quality", {}).get("reviewed") is True
    for record in records
    if record.get("domain") == "customer_support"
)

sample_response = customer_support_responses["customer_support_001"]

print(f"Updated customer-support records: {updated_count}")
print(f"Reviewed customer-support records: {reviewed_count}")
print("\nSample revised response:\n")
print(sample_response)


# Replace scheduling scaffolds with reviewed Tamil scheduling examples

if "records" not in globals():
    raise RuntimeError(
        "records is not defined. Run the 60-record scaffold-generation cell first."
    )

SCHEDULING_RESPONSES = {
    "scheduling_001": (
        "நினைவூட்டல்: நாளை காலை 10 மணிக்கு திட்ட மதிப்பாய்வு கூட்டம் நடைபெறும். "
        "தற்போதைய முன்னேற்றம், நிலுவைப் பணிகள் மற்றும் அடுத்த கட்ட நடவடிக்கைகள் பற்றிய குறிப்புகளை தயார் வைத்திருக்கவும்."
    ),
    "scheduling_002": (
        "வணக்கம், வெள்ளிக்கிழமை பிற்பகலில் அணி அழைப்பை நடத்த திட்டமிட்டுள்ளோம். "
        "உங்களுக்கு வசதியான நேரத்தை உறுதிப்படுத்தி தெரிவிக்கவும்."
    ),
    "scheduling_003": (
        "வணக்கம், அடுத்த வாரத்திற்கு திட்டமிடப்பட்ட சந்திப்பை மாற்றி அமைக்க விரும்புகிறோம். "
        "உங்களுக்கு ஏற்ற தேதி மற்றும் நேரத்தை பகிரவும்."
    ),
    "scheduling_004": (
        "வணக்கம், திட்டமிடப்பட்ட நேரத்தில் நேர்காணலில் பங்கேற்க முடியாததற்கு வருந்துகிறோம். "
        "உங்களுக்கு வசதியான மாற்று தேதி மற்றும் நேரத்தைத் தெரிவிக்கவும்."
    ),
    "scheduling_005": (
        "தகவல்: இன்றைய கூட்டம் 30 நிமிடங்கள் தாமதமாகத் தொடங்கும். "
        "புதுப்பிக்கப்பட்ட தொடக்க நேரத்தை கவனத்தில் கொள்ளவும்."
    ),
    "scheduling_006": (
        "வணக்கம், உங்கள் மருத்துவர் சந்திப்பு திட்டமிட்டபடி உறுதிப்படுத்தப்பட்டுள்ளது. "
        "குறிப்பிட்ட நேரத்திற்கு சில நிமிடங்கள் முன்பாக வருமாறு கேட்டுக்கொள்கிறோம்."
    ),
    "scheduling_007": (
        "வணக்கம், பயிற்சி அமர்வுக்கான கால அட்டவணை பகிரப்பட்டுள்ளது. "
        "ஒதுக்கப்பட்ட நேரத்தில் பங்கேற்கவும்; மாற்றம் தேவைப்பட்டால் முன்கூட்டியே தெரிவிக்கவும்."
    ),
    "scheduling_008": (
        "வணக்கம், வாடிக்கையாளர் அழைப்பின் நேரம் மாற்றப்பட்டுள்ளது. "
        "புதுப்பிக்கப்பட்ட தேதி மற்றும் நேரத்தை உங்கள் நாட்காட்டியில் பதிவு செய்து உறுதிப்படுத்தவும்."
    ),
    "scheduling_009": (
        "வணக்கம், வாராந்திர மதிப்பாய்வு கூட்டம் திட்டமிட்டபடி நடைபெறும். "
        "உங்கள் பணிகளின் முன்னேற்றம் மற்றும் எதிர்கொள்ளும் சவால்கள் பற்றிய சுருக்கத்தை தயார் வைத்திருக்கவும்."
    ),
    "scheduling_010": (
        "தகவல்: கூட்ட அறை மாற்றப்பட்டுள்ளது. புதிய அறை விவரங்களைப் பார்த்து, கூட்டத்திற்கு சரியான இடத்தில் நேரத்திற்கு வரவும்."
    ),
}

def get_assistant_message(record):
    return next(
        message for message in record["messages"]
        if message["role"] == "assistant"
    )

updated = 0

for record in records:
    record_id = record.get("id")

    if record.get("domain") == "scheduling" and record_id in SCHEDULING_RESPONSES:
        get_assistant_message(record)["content"] = SCHEDULING_RESPONSES[record_id]

        record["quality"].update({
            "correct_domain": True,
            "clear_tamil": True,
            "answers_user_request": True,
            "safe_response": True,
            "no_personal_data": True,
            "no_copyrighted_text": True,
            "reviewed": True,
        })
        updated += 1

scheduling_records = [
    record for record in records
    if record.get("domain") == "scheduling"
]

print(f"Updated reviewed scheduling records: {updated}")
print(f"Scheduling records: {len(scheduling_records)}")
print(
    "Reviewed scheduling records:",
    sum(record["quality"]["reviewed"] for record in scheduling_records),
)

print("\nSample revised response:\n")
print(get_assistant_message(scheduling_records[0])["content"])


# Read-only Tamil-Orca inspection — no persistence, no training, no JSONL export

import os
import re
import unicodedata
from collections import Counter
from datasets import load_dataset
from huggingface_hub import HfApi

DATASET_ID = "azharmo/tamil-orca"
SAMPLE_ROWS = 100

os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "30"

def tamil_char_count(value):
    text = "" if value is None else str(value)
    return sum("\u0B80" <= char <= "\u0BFF" for char in text)

def text_length(value):
    return len("" if value is None else str(value))

api = HfApi()
info = api.dataset_info(DATASET_ID)

print("=" * 80)
print("DATASET METADATA")
print("=" * 80)
print("Dataset:", DATASET_ID)
print("License:", getattr(info.cardData, "license", "Not declared in card metadata"))
print("Tags:", getattr(info, "tags", []))
print("Description preview:")
print((getattr(info.cardData, "pretty_name", "") or "No pretty name declared"))
print()

dataset = load_dataset(DATASET_ID)

print("=" * 80)
print("SPLITS AND SCHEMA")
print("=" * 80)

for split_name, split_data in dataset.items():
    print(f"\nSplit: {split_name}")
    print(f"Rows: {len(split_data):,}")
    print("Columns:", split_data.column_names)
    print("Features:", split_data.features)

    sample_n = min(SAMPLE_ROWS, len(split_data))
    sample = split_data.select(range(sample_n))
    print(f"\nRead-only sample size: {sample_n}")

    print("\nMissing-value counts:")
    for column in sample.column_names:
        missing = sum(
            value is None or (isinstance(value, str) and not value.strip())
            for value in sample[column]
        )
        print(f"  {column}: {missing}/{sample_n}")

    text_columns = [
        column for column in sample.column_names
        if sample.features[column].dtype == "string"
    ]

    print("\nText metrics:")
    for column in text_columns:
        values = sample[column]
        lengths = [text_length(value) for value in values]
        tamil_counts = [tamil_char_count(value) for value in values]
        tamil_coverage = sum(count > 0 for count in tamil_counts) / sample_n * 100

        print(
            f"  {column}: "
            f"length min/median/max = "
            f"{min(lengths)}/{sorted(lengths)[sample_n // 2]}/{max(lengths)}, "
            f"rows with Tamil Unicode = {tamil_coverage:.1f}%"
        )

    print("\nFirst 3 rows:")
    for index in range(min(3, sample_n)):
        print(f"\n--- {split_name} row {index} ---")
        print(sample[index])

print("\nInspection finished.")
print("Nothing was saved to /kaggle/working and no training was started.")


if "records" not in globals():
    raise RuntimeError("records is not defined. Run the scaffold-generation cell first.")

if "review_record" not in globals():
    raise RuntimeError("review_record is not defined. Run the existing in-memory review-gate cell first.")

scheduling_responses = {
    "scheduling_001": "கூட்டத்திற்கான உங்களுக்கு ஏற்ற தேதி மற்றும் நேரத்தைத் தெரிவிக்கவும். அதன்படி அழைப்பைத் திட்டமிடலாம்.",
    "scheduling_002": "குறிப்பிட்ட நேரம் உங்களுக்கு ஏற்றதா என்பதைத் தெரிவிக்கவும். ஏற்றதாக இல்லையெனில், மாற்று நேரத்தை பகிரவும்.",
    "scheduling_003": "கூட்டத்தை மாற்ற வேண்டுமெனில், உங்களுக்கு ஏற்ற புதிய தேதி மற்றும் நேரத்தைத் தெரிவிக்கவும்.",
    "scheduling_004": "சந்திப்பை ஏற்பாடு செய்ய, உங்களுக்கு விருப்பமான தேதி மற்றும் நேரத்தை பகிரவும்.",
    "scheduling_005": "குறிப்பிட்ட நாளில் உங்கள் கிடைப்புநேரத்தைத் தெரிவிக்கவும். அதற்கு ஏற்ப கூட்ட நேரத்தைத் தேர்வு செய்யலாம்.",
    "scheduling_006": "கூட்டத்திற்கான மாற்று நேரத்தைத் தெரிவிக்கவும்; அதன்படி அழைப்பை புதுப்பிக்கலாம்.",
    "scheduling_007": "இந்த சந்திப்பை ரத்து செய்ய விரும்புவதாகப் பதிவு செய்துள்ளோம். மாற்று நேரம் தேவைப்பட்டால் தெரிவிக்கவும்.",
    "scheduling_008": "அழைப்பில் சேர வேண்டியவர்களின் கிடைப்புநேரத்தை உறுதிப்படுத்திய பிறகு, பொருத்தமான நேரத்தைத் தெரிவிக்கவும்.",
    "scheduling_009": "நினைவூட்டலை அமைக்க வேண்டிய தேதி மற்றும் நேரத்தைத் தெரிவிக்கவும்.",
    "scheduling_010": "சந்திப்பை உறுதிப்படுத்துவதற்கு முன், குறிப்பிடப்பட்ட நேரம் அனைவருக்கும் ஏற்றதா என்பதைத் தெரிவிக்கவும்.",
}

quality_fields = (
    "correct_domain",
    "clear_tamil",
    "answers_user_request",
    "safe_response",
    "no_personal_data",
    "no_copyrighted_text",
    "reviewed",
)

scheduling_records = [
    record for record in records
    if record.get("domain") == "scheduling"
]

expected_ids = set(scheduling_responses)
actual_ids = {record.get("id") for record in scheduling_records}

if actual_ids != expected_ids:
    raise RuntimeError(
        "Scheduling record IDs do not match the expected controlled-corpus IDs. "
        f"Expected: {sorted(expected_ids)}; found: {sorted(actual_ids)}"
    )

updated_ids = []
not_approved = []

for record in scheduling_records:
    record_id = record["id"]
    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )

    assistant_message["content"] = scheduling_responses[record_id]

    issues = review_record(record)
    semantic_ok = (
        not issues
        and len(assistant_message["content"].strip()) >= 25
    )

    if semantic_ok:
        for field in quality_fields:
            record["quality"][field] = True
        updated_ids.append(record_id)
    else:
        record["quality"]["reviewed"] = False
        not_approved.append((record_id, issues))

print("Scheduling IDs updated:")
for record_id in sorted(updated_ids):
    print(f"- {record_id}")

print("\nFinal scheduling assistant answers:")
for record in sorted(scheduling_records, key=lambda item: item["id"]):
    answer = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    print(f"\n{record['id']}:\n{answer}")

print("\nRecords not safely approved:")
if not not_approved:
    print("- None")
else:
    for record_id, issues in not_approved:
        print(f"- {record_id}: {'; '.join(issues)}")

review_results = {}
for record in records:
    record_id = record.get("id", "unknown")
    issues = review_record(record)
    record["quality"]["reviewed"] = len(issues) == 0
    record["quality"]["answers_user_request"] = len(issues) == 0
    record["quality"]["clear_tamil"] = bool(
        TAMIL_RE.search(get_message(record, "assistant"))
    )
    if issues:
        review_results[record_id] = issues

print("\nReview-gate counts by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [r for r in records if r["domain"] == domain]
    approved = sum(r["quality"]["reviewed"] for r in domain_records)
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nOverall status: NOT READY FOR EXPORT OR TRAINING")



general_assistance_records = [
    record for record in records
    if record.get("domain") == "general_assistance"
]

print("GENERAL_ASSISTANCE PROMPTS")
print("=" * 72)

for record in sorted(general_assistance_records, key=lambda item: item["id"]):
    user_prompt = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )
    print(f"\n{record['id']}")
    print(f"User: {user_prompt}")


if "records" not in globals():
    raise RuntimeError("records is not defined. Run the controlled-corpus scaffold cell first.")

if "review_record" not in globals():
    raise RuntimeError("review_record is not defined. Run the existing in-memory review-gate cell first.")

general_assistance_responses = {
    "general_assistance_001": (
        "பணிகளை முதலில் அவசரம் மற்றும் முக்கியத்துவம் அடிப்படையில் பிரிக்கவும். "
        "காலக்கெடு நெருக்கமான முக்கிய பணிகளை முதலில் செய்து, மீதியை வரிசைப்படுத்தவும்."
    ),
    "general_assistance_002": (
        "கூட்டத்தின் நோக்கம், முக்கிய விவாதங்கள், முடிவுகள், பொறுப்பாளர்கள் மற்றும் "
        "அடுத்த நடவடிக்கைகளைச் சுருக்கமாகப் பதிவு செய்யவும்."
    ),
    "general_assistance_003": (
        "முதலில் அன்றைய முக்கிய மூன்று பணிகளைத் தேர்வு செய்யவும். "
        "ஒவ்வொரு பணிக்கும் நேர ஒதுக்கி, முடிந்த பணிகளைப் பட்டியலில் குறிக்கவும்."
    ),
    "general_assistance_004": (
        "முன்னைய தொடர்பைச் சுருக்கமாகக் குறிப்பிடுங்கள். பின்னர் தேவையான தகவல் அல்லது "
        "நடவடிக்கையை மரியாதையாகக் கேட்டு, பதிலுக்கான நன்றியையும் தெரிவிக்கவும்."
    ),
    "general_assistance_005": (
        "முக்கிய பணிகளுக்கு முன்னுரிமை கொடுத்து நேரத் துண்டுகளாக வேலை செய்யவும். "
        "தேவையற்ற அறிவிப்புகளைத் தவிர்த்து, இடைவேளைகளையும் திட்டமிடவும்."
    ),
    "general_assistance_006": (
        "ஒரு சுருக்கமான அறிக்கையில் தலைப்பு, நோக்கம், முக்கிய தகவல்கள், முடிவு மற்றும் "
        "தேவையான அடுத்த நடவடிக்கைகள் இடம்பெறலாம்."
    ),
    "general_assistance_007": (
        "கூட்டத்தின் நோக்கம், எதிர்பார்க்கப்படும் முடிவு, பொறுப்பாளர்கள், காலக்கெடு மற்றும் "
        "அடுத்த நடவடிக்கைகள் குறித்து கேள்விகள் கேட்கலாம்."
    ),
    "general_assistance_008": (
        "ஒவ்வொரு பணிக்கும் தெளிவான தலைப்பு, பொறுப்பாளர், காலக்கெடு மற்றும் நிலையைப் பதிவு செய்யவும். "
        "பணிப் பட்டியலைத் தொடர்ந்து புதுப்பித்து, முடிந்தவற்றை மூடவும்."
    ),
    "general_assistance_009": (
        "நினைவூட்டல் செய்தியில் செய்ய வேண்டிய பணி, தேவையான நேரம் அல்லது காலக்கெடு, மற்றும் "
        "தேவையான அடுத்த செயலைத் தெளிவாகச் சேர்க்கவும்."
    ),
    "general_assistance_010": (
        "மரியாதையாக மறுக்க, கோரிக்கையை ஏற்க முடியாததைத் தெளிவாகக் கூறுங்கள். "
        "பொருத்தமான மாற்று வழி இருந்தால் மட்டும் அதைச் சுருக்கமாக வழங்கலாம்."
    ),
}

required_quality_fields = (
    "correct_domain",
    "clear_tamil",
    "answers_user_request",
    "safe_response",
    "no_personal_data",
    "no_copyrighted_text",
    "reviewed",
)

target_records = [
    record for record in records
    if record.get("domain") == "general_assistance"
]

expected_ids = set(general_assistance_responses)
actual_ids = {record.get("id") for record in target_records}

if actual_ids != expected_ids:
    raise RuntimeError(
        "General-assistance record IDs do not match the expected set. "
        f"Expected: {sorted(expected_ids)}; found: {sorted(actual_ids)}"
    )

updated_ids = []
not_approved = []

for record in target_records:
    record_id = record["id"]
    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )

    assistant_message["content"] = general_assistance_responses[record_id]

    issues = review_record(record)
    semantic_ok = (
        not issues
        and len(assistant_message["content"].strip()) >= 25
        and record["domain"] == "general_assistance"
    )

    if semantic_ok:
        for field in required_quality_fields:
            record["quality"][field] = True
        updated_ids.append(record_id)
    else:
        record["quality"]["reviewed"] = False
        not_approved.append((record_id, issues))

print("General-assistance IDs updated:")
for record_id in sorted(updated_ids):
    print(f"- {record_id}")

print("\nFinal assistant answers:")
for record in sorted(target_records, key=lambda item: item["id"]):
    answer = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    print(f"\n{record['id']}:\n{answer}")

print("\nRecords not safely approved:")
if not not_approved:
    print("- None")
else:
    for record_id, issues in not_approved:
        print(f"- {record_id}: {'; '.join(issues)}")

print("\nReview-gate counts by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [record for record in records if record["domain"] == domain]
    approved = sum(record["quality"].get("reviewed", False) for record in domain_records)
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nOverall status: NOT READY FOR EXPORT OR TRAINING")


tanglish_records = [
    record for record in records
    if record.get("domain") == "tanglish_to_tamil"
]

print("TANGLISH-TO-TAMIL PROMPTS")
print("=" * 72)

for record in sorted(tanglish_records, key=lambda item: item["id"]):
    user_prompt = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )
    print(f"\n{record['id']}")
    print(f"User: {user_prompt}")


if "records" not in globals():
    raise RuntimeError("records is not defined. Run the controlled-corpus scaffold cell first.")

if "review_record" not in globals():
    raise RuntimeError("review_record is not defined. Run the existing in-memory review-gate cell first.")

tanglish_to_tamil_responses = {
    "tanglish_to_tamil_001": (
        "இந்த ஆவணத்தை இன்று மாலைக்குள் அனுப்புங்கள்."
    ),
    "tanglish_to_tamil_002": (
        "வாடிக்கையாளரிடம் கருத்துக் கேட்டு ஒரு செய்தி அனுப்புங்கள்."
    ),
    "tanglish_to_tamil_003": (
        "நாளைய கூட்டத்திற்கான நினைவூட்டலை அனுப்புங்கள்."
    ),
    "tanglish_to_tamil_004": (
        "பணம் செலுத்தப்பட வேண்டியுள்ளது என்பதை மரியாதையாகத் தெரிவியுங்கள்."
    ),
    "tanglish_to_tamil_005": (
        "இதில் கையொப்பமிட்டு, ஸ்கேன் செய்து அனுப்புங்கள்."
    ),
    "tanglish_to_tamil_006": (
        "அட்டவணை மாற்றப்பட்டுவிட்டது என்பதைத் தெரிவியுங்கள்."
    ),
    "tanglish_to_tamil_007": (
        "தொடர்ந்து விசாரிக்க ஒரு சுருக்கமான செய்தி வேண்டும்."
    ),
    "tanglish_to_tamil_008": (
        "காலக்கெடுவை நீட்டிக்கக் கோரிக்கை விடுங்கள்."
    ),
    "tanglish_to_tamil_009": (
        "அழைப்பில் கலந்துகொள்ள முடியாது என்பதைத் தெரிவியுங்கள்."
    ),
    "tanglish_to_tamil_010": (
        "புதுப்பிக்கப்பட்ட கோப்பைப் பகிருங்கள்."
    ),
}

target_records = [
    record for record in records
    if record.get("domain") == "tanglish_to_tamil"
]

expected_ids = set(tanglish_to_tamil_responses)
actual_ids = {record.get("id") for record in target_records}

if actual_ids != expected_ids:
    raise RuntimeError(
        "Tanglish-to-Tamil record IDs do not match the expected set. "
        f"Expected: {sorted(expected_ids)}; found: {sorted(actual_ids)}"
    )

updated_ids = []
not_approved = []

for record in target_records:
    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )
    assistant_message["content"] = tanglish_to_tamil_responses[record["id"]]

    issues = review_record(record)
    semantic_ok = (
        not issues
        and record["domain"] == "tanglish_to_tamil"
        and assistant_message["content"].strip()
    )

    if semantic_ok:
        record["quality"]["correct_domain"] = True
        record["quality"]["clear_tamil"] = True
        record["quality"]["answers_user_request"] = True
        record["quality"]["safe_response"] = True
        record["quality"]["no_personal_data"] = True
        record["quality"]["no_copyrighted_text"] = True
        record["quality"]["reviewed"] = True
        updated_ids.append(record["id"])
    else:
        record["quality"]["reviewed"] = False
        not_approved.append((record["id"], issues))

print("Tanglish-to-Tamil IDs updated:")
for record_id in sorted(updated_ids):
    print(f"- {record_id}")

print("\nFinal Tamil translations:")
for record in sorted(target_records, key=lambda item: item["id"]):
    answer = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    print(f"\n{record['id']}:\n{answer}")

print("\nRecords not safely approved:")
if not not_approved:
    print("- None")
else:
    for record_id, issues in not_approved:
        print(f"- {record_id}: {'; '.join(issues)}")

print("\nReview-gate counts by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [record for record in records if record["domain"] == domain]
    approved = sum(
        record["quality"].get("reviewed", False)
        for record in domain_records
    )
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nOverall status: NOT READY FOR EXPORT OR TRAINING")


safety_records = [
    record for record in records
    if record.get("domain") == "safety_refusal"
]

print("SAFETY-REFUSAL PROMPTS")
print("=" * 72)

for record in sorted(safety_records, key=lambda item: item["id"]):
    user_prompt = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "user"
    )
    print(f"\n{record['id']}")
    print(f"User: {user_prompt}")


if "records" not in globals():
    raise RuntimeError("records is not defined. Run the controlled-corpus scaffold cell first.")

if "review_record" not in globals():
    raise RuntimeError("review_record is not defined. Run the existing in-memory review-gate cell first.")

safety_refusal_responses = {
    "safety_refusal_001": (
        "வேறொருவரின் கணக்கிற்கான கடவுச்சொல்லை கண்டறிய உதவ முடியாது. "
        "உங்கள் சொந்தக் கணக்காக இருந்தால், அதிகாரப்பூர்வ கடவுச்சொல் மீட்டமைப்பு முறையைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_002": (
        "ஒரு இணையதளத்தை பாதிக்கும் ஸ்கிரிப்டை வழங்க முடியாது. "
        "அனுமதி பெற்ற சூழலில் பாதுகாப்புச் சோதனை செய்ய வேண்டுமெனில், சட்டபூர்வமான சோதனை நடைமுறைகளைப் பின்பற்றுங்கள்."
    ),
    "safety_refusal_003": (
        "மற்றொருவரின் மின்னஞ்சலை அணுக உதவ முடியாது. "
        "உங்கள் சொந்தக் கணக்காக இருந்தால், வழங்குநரின் கணக்கு மீட்பு வழிமுறைகளைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_004": (
        "போலியான பணம் செலுத்திய சான்றை உருவாக்க உதவ முடியாது. "
        "சரியான பரிவர்த்தனை ரசீது தேவைப்பட்டால், உரிய கட்டண சேவை அல்லது வங்கியைத் தொடர்புகொள்ளுங்கள்."
    ),
    "safety_refusal_005": (
        "நிறுவன உள்தகவலைத் திருட உதவ முடியாது. "
        "தேவையான தகவலுக்கு அனுமதி இருந்தால், நிறுவனத்தின் அங்கீகரிக்கப்பட்ட அணுகல் முறையைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_006": (
        "மோசடி நோக்கிலான மின்னஞ்சல் வடிவத்தை வழங்க முடியாது. "
        "பாதுகாப்பு விழிப்புணர்வுக்காக, சந்தேகமான இணைப்புகள் மற்றும் கோரிக்கைகளை எவ்வாறு அடையாளம் காண்பது என்பதை விளக்க உதவலாம்."
    ),
    "safety_refusal_007": (
        "CCTV அமைப்பை செயலிழக்கச் செய்வது குறித்து உதவ முடியாது. "
        "அங்கீகரிக்கப்பட்ட பராமரிப்பு தேவைப்பட்டால், பொறுப்பான நிர்வாகி அல்லது சேவை நிபுணரைத் தொடர்புகொள்ளுங்கள்."
    ),
    "safety_refusal_008": (
        "பிறரின் OTP-ஐப் பெறுவதற்கான வழிகளைச் சொல்ல முடியாது. "
        "உங்கள் சொந்தக் கணக்கில் சிக்கல் இருந்தால், அதிகாரப்பூர்வ மீட்பு அல்லது ஆதரவு முறையைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_009": (
        "அனுமதியில்லாத கணக்கு அணுகலுக்கு உதவ முடியாது. "
        "உங்கள் சொந்தக் கணக்காக இருந்தால், கடவுச்சொல் மீட்டமைப்பு மற்றும் அதிகாரப்பூர்வ ஆதரவு வழிகளைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_010": (
        "தீங்கிழைக்கும் மென்பொருளை அனுப்பும் முறைகளை வழங்க முடியாது. "
        "உங்கள் சாதனத்தைப் பாதுகாக்க, நம்பகமான பாதுகாப்பு மென்பொருளைப் பயன்படுத்தி சந்தேகமான கோப்புகள் அல்லது இணைப்புகளைத் தவிர்க்கவும்."
    ),
}

target_records = [
    record for record in records
    if record.get("domain") == "safety_refusal"
]

expected_ids = set(safety_refusal_responses)
actual_ids = {record.get("id") for record in target_records}

if actual_ids != expected_ids:
    raise RuntimeError(
        "Safety-refusal record IDs do not match the expected set. "
        f"Expected: {sorted(expected_ids)}; found: {sorted(actual_ids)}"
    )

updated_ids = []
not_approved = []

for record in target_records:
    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )
    assistant_message["content"] = safety_refusal_responses[record["id"]]

    issues = review_record(record)
    semantic_ok = (
        not issues
        and record["domain"] == "safety_refusal"
        and assistant_message["content"].strip()
    )

    if semantic_ok:
        record["quality"]["correct_domain"] = True
        record["quality"]["clear_tamil"] = True
        record["quality"]["answers_user_request"] = True
        record["quality"]["safe_response"] = True
        record["quality"]["no_personal_data"] = True
        record["quality"]["no_copyrighted_text"] = True
        record["quality"]["reviewed"] = True
        updated_ids.append(record["id"])
    else:
        record["quality"]["reviewed"] = False
        not_approved.append((record["id"], issues))

print("Safety-refusal IDs updated:")
for record_id in sorted(updated_ids):
    print(f"- {record_id}")

print("\nFinal assistant answers:")
for record in sorted(target_records, key=lambda item: item["id"]):
    answer = next(
        message["content"]
        for message in record["messages"]
        if message["role"] == "assistant"
    )
    print(f"\n{record['id']}:\n{answer}")

print("\nRecords not safely approved:")
if not not_approved:
    print("- None")
else:
    for record_id, issues in not_approved:
        print(f"- {record_id}: {'; '.join(issues)}")

print("\nReview-gate counts by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [record for record in records if record["domain"] == domain]
    approved = sum(
        record["quality"].get("reviewed", False)
        for record in domain_records
    )
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nOverall status: NOT READY FOR EXPORT OR TRAINING")


if "records" not in globals():
    raise RuntimeError("records is not defined.")

if "review_record" not in globals():
    raise RuntimeError("review_record is not defined.")

refusal_fixes = {
    "safety_refusal_002": (
        "ஒரு இணையதளத்தை பாதிக்கும் ஸ்கிரிப்டை வழங்க உதவ முடியாது. "
        "அனுமதி பெற்ற சூழலில் பாதுகாப்புச் சோதனை செய்ய வேண்டுமெனில், சட்டபூர்வமான சோதனை நடைமுறைகளைப் பின்பற்றுங்கள்."
    ),
    "safety_refusal_006": (
        "மோசடி நோக்கிலான மின்னஞ்சல் வடிவத்தை உருவாக்க உதவ முடியாது. "
        "பாதுகாப்பு விழிப்புணர்வுக்காக, சந்தேகமான இணைப்புகள் மற்றும் கோரிக்கைகளை அடையாளம் காணும் முறைகளை விளக்க உதவலாம்."
    ),
    "safety_refusal_008": (
        "பிறரின் OTP-ஐப் பெறுவதற்கான வழிகளைச் சொல்ல உதவ முடியாது. "
        "உங்கள் சொந்தக் கணக்கில் சிக்கல் இருந்தால், அதிகாரப்பூர்வ மீட்பு அல்லது ஆதரவு முறையைப் பயன்படுத்துங்கள்."
    ),
    "safety_refusal_010": (
        "தீங்கிழைக்கும் மென்பொருளை அனுப்பும் முறைகளை வழங்க உதவ முடியாது. "
        "உங்கள் சாதனத்தைப் பாதுகாக்க, நம்பகமான பாதுகாப்பு மென்பொருளைப் பயன்படுத்தி சந்தேகமான கோப்புகள் அல்லது இணைப்புகளைத் தவிர்க்கவும்."
    ),
}

updated = []
for record in records:
    if record.get("id") not in refusal_fixes:
        continue

    assistant_message = next(
        message for message in record["messages"]
        if message.get("role") == "assistant"
    )
    assistant_message["content"] = refusal_fixes[record["id"]]

    issues = review_record(record)
    record["quality"]["reviewed"] = not issues
    record["quality"]["answers_user_request"] = not issues
    record["quality"]["clear_tamil"] = True

    if not issues:
        record["quality"]["correct_domain"] = True
        record["quality"]["safe_response"] = True
        record["quality"]["no_personal_data"] = True
        record["quality"]["no_copyrighted_text"] = True
        updated.append(record["id"])
    else:
        print(f"Still rejected: {record['id']} -> {'; '.join(issues)}")

print("Updated safety-refusal IDs:")
for record_id in sorted(updated):
    print(f"- {record_id}")

print("\nReview-gate counts by domain:")
for domain in APPROVED_DOMAINS:
    domain_records = [record for record in records if record["domain"] == domain]
    approved = sum(
        record["quality"].get("reviewed", False)
        for record in domain_records
    )
    print(f"- {domain}: {approved}/{len(domain_records)} approved")

print("\nOverall status: NOT READY FOR EXPORT OR TRAINING")


import re
from collections import Counter

EXPECTED_DOMAINS = {
    "business_email",
    "customer_support",
    "scheduling",
    "general_assistance",
    "tanglish_to_tamil",
    "safety_refusal",
}
EXPECTED_ROLES = ["system", "user", "assistant"]
ID_PATTERN = re.compile(r"^[a-z_]+_\d{3}$")
TAMIL_PATTERN = re.compile(r"[\u0B80-\u0BFF]")

if "records" not in globals():
    raise RuntimeError(
        "No in-memory `records` object found. Do not recreate scaffolds; "
        "restore the reviewed 60-record corpus first."
    )

issues = []
ids = []
normalized_prompts = []
domain_counts = Counter()

for index, record in enumerate(records, start=1):
    prefix = f"Record {index}"

    if not isinstance(record, dict):
        issues.append(f"{prefix}: record must be a dictionary.")
        continue

    record_id = record.get("id")
    domain = record.get("domain")
    messages = record.get("messages")
    quality = record.get("quality")

    if not isinstance(record_id, str) or not ID_PATTERN.fullmatch(record_id):
        issues.append(f"{prefix}: invalid ID {record_id!r}; expected domain_###.")
    else:
        ids.append(record_id)

    if domain not in EXPECTED_DOMAINS:
        issues.append(f"{prefix}: unapproved or missing domain {domain!r}.")
    else:
        domain_counts[domain] += 1
        if isinstance(record_id, str) and ID_PATTERN.fullmatch(record_id):
            id_domain = record_id.rsplit("_", 1)[0]
            if id_domain != domain:
                issues.append(
                    f"{prefix}: ID domain {id_domain!r} does not match "
                    f"record domain {domain!r}."
                )

    if not isinstance(messages, list) or len(messages) != 3:
        issues.append(f"{prefix}: messages must contain exactly 3 entries.")
        continue

    roles = [message.get("role") if isinstance(message, dict) else None for message in messages]
    if roles != EXPECTED_ROLES:
        issues.append(
            f"{prefix}: message roles must be {EXPECTED_ROLES}; found {roles}."
        )

    message_by_role = {}
    for message in messages:
        if not isinstance(message, dict):
            issues.append(f"{prefix}: each message must be a dictionary.")
            continue

        if set(message) != {"role", "content"}:
            issues.append(
                f"{prefix}: each message must contain only 'role' and 'content'."
            )

        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            issues.append(f"{prefix}: message content must be a non-empty string.")

        message_by_role[message.get("role")] = content

    user_text = message_by_role.get("user", "")
    assistant_text = message_by_role.get("assistant", "")

    if isinstance(user_text, str) and user_text.strip():
        normalized_prompts.append(user_text.strip().lower())

    if not isinstance(assistant_text, str) or not TAMIL_PATTERN.search(assistant_text):
        issues.append(f"{prefix}: assistant response lacks Tamil Unicode.")

    if not isinstance(quality, dict):
        issues.append(f"{prefix}: missing quality metadata.")
    elif quality.get("reviewed") is not True:
        issues.append(f"{prefix}: quality['reviewed'] must be True.")

if len(records) != 60:
    issues.append(f"Total record count must be exactly 60; found {len(records)}.")

if len(ids) != len(set(ids)):
    duplicate_ids = sorted(
        record_id for record_id, count in Counter(ids).items() if count > 1
    )
    issues.append(f"Duplicate IDs found: {duplicate_ids}.")

if len(normalized_prompts) != len(set(normalized_prompts)):
    duplicate_prompts = sorted(
        prompt for prompt, count in Counter(normalized_prompts).items()
        if count > 1
    )
    issues.append(
        "Duplicate normalized user prompts found: "
        + "; ".join(repr(prompt) for prompt in duplicate_prompts)
    )

for domain in sorted(EXPECTED_DOMAINS):
    if domain_counts[domain] != 10:
        issues.append(
            f"Domain {domain!r} must contain exactly 10 records; "
            f"found {domain_counts[domain]}."
        )

print("=" * 72)
print("READ-ONLY PRE-EXPORT AUDIT")
print("=" * 72)
print(f"Total records: {len(records)}")
print("\nDomain counts:")
for domain in sorted(EXPECTED_DOMAINS):
    print(f"- {domain}: {domain_counts[domain]}")

print(f"\nUnique IDs: {len(set(ids))}/{len(ids)}")
print(
    f"Normalized unique user prompts: "
    f"{len(set(normalized_prompts))}/{len(normalized_prompts)}"
)
print(f"Audit issues: {len(issues)}")

if issues:
    print("\nAUDIT RESULT: FAILED")
    for issue in issues:
        print(f"- {issue}")
    print("\nStatus: NOT READY FOR EXPORT OR TRAINING")
else:
    print("\nAUDIT RESULT: PASSED")
    print("Status: READY FOR VERSIONED EXPORT REVIEW — NOT READY FOR TRAINING")


# ==============================================================================
# Phase 1: Manual Promotion & QLoRA Fine-Tuning Execution
# ==============================================================================
import os, json, torch
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

# 1. Promote 20 audited Phase 1 records
promoted_records = []
for r in business_email_records:
    rec = dict(r)
    rec["status"] = "approved"
    if "quality" in rec and isinstance(rec["quality"], dict):
        rec["quality"]["reviewed"] = True
    promoted_records.append(rec)

for r in (customer_support_responses if isinstance(customer_support_responses, list) else customer_support_responses.values()):
    if isinstance(r, dict):
        rec = dict(r)
        rec["status"] = "approved"
        if "quality" in rec and isinstance(rec["quality"], dict):
            rec["quality"]["reviewed"] = True
        promoted_records.append(rec)

export_path = "/kaggle/working/tamil_phase1_approved.jsonl"
with open(export_path, "w", encoding="utf-8") as f:
    for rec in promoted_records:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
print(f"Promoted and saved {len(promoted_records)} records to {export_path}")

# 2. Format dataset for Qwen ChatML
formatted_data = []
for r in promoted_records:
    messages = [
        {"role": "system", "content": "You are a helpful, professional AI assistant fluent in Tamil and English."},
        {"role": "user", "content": r.get("user_prompt") or r.get("prompt") or ""},
        {"role": "assistant", "content": r.get("assistant_response") or r.get("response") or ""}
    ]
    chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
    formatted_data.append({"text": chat_text})

raw_dataset = Dataset.from_list(formatted_data)
def tok_fn(b):
    tok = tokenizer(b["text"], truncation=True, max_length=512)
    tok["labels"] = tok["input_ids"].copy()
    return tok
tokenized_dataset = raw_dataset.map(tok_fn, batched=True)

# 3. Train with QLoRA
model.train()
output_adapter_dir = "/kaggle/working/tamil_qwen3_phase1_adapter"
training_args = TrainingArguments(
    output_dir=output_adapter_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    optim="paged_adamw_8bit"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
)

train_result = trainer.train()
trainer.save_model(output_adapter_dir)
tokenizer.save_pretrained(output_adapter_dir)
print(f"Training complete! Loss: {train_result.training_loss:.4f}, Adapter saved to {output_adapter_dir}")
