import urllib.request
import json
import time
import sys

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def call_chat(messages):
    payload = {
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.3
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8000/chat", data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res.get("response"), time.time() - t0

print("=== Simulating User Multi-Turn Interaction ===")
history = []

# Turn 1
p1 = "write email for leave letter for school"
history.append({"role": "user", "content": p1})
print(f"\nUser [Turn 1]: {p1}")
ans1, t1 = call_chat(history)
print(f"Assistant [Turn 1] ({t1:.2f}s):\n{ans1}")
history.append({"role": "assistant", "content": ans1})

# Turn 2
p2 = "Prakash 2, subbaraya Mudali Street, Royapettah CEO 9840705435 Company name - infygalaxy"
history.append({"role": "user", "content": p2})
print(f"\nUser [Turn 2]: {p2}")
ans2, t2 = call_chat(history)
print(f"Assistant [Turn 2] ({t2:.2f}s):\n{ans2}")
