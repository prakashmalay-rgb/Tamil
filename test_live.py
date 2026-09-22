import urllib.request
import json
import time
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

prompts = [
    "வணக்கம்! நீங்கள் யார்?",
    "Innaiku weather pathi sollunga, evening rain varuma?",
    "What is photosynthesis?"
]

print("=== Testing Live Inference with Newly Distilled QLoRA Adapter ===")
for p in prompts:
    data = json.dumps({"prompt": p, "max_tokens": 160}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/chat", data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req) as resp:
            res = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            print(f"\n[Prompt]: {p}")
            print(f"[Response]: {res.get('response')}")
            print(f"[Time]: {elapsed:.2f}s | [Status]: {res.get('status')}")
    except Exception as e:
        print(f"Error: {e}")
