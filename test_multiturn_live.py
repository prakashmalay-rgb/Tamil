"""
Live Multi-Turn Validation Probe
Tests Turn 1: 'write email for leave letter for school'
Tests Turn 2: 'Prakash 2, subbaraya Mudali Street, Royapettah CEO 9840705435 Company name - infygalaxy'
"""
import urllib.request
import json
import sys

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

URL = "http://127.0.0.1:8000/chat"

def send_chat(messages):
    print(f"--> Sending request to {URL} with {len(messages)} message(s)...", flush=True)
    payload = json.dumps({"messages": messages, "max_tokens": 512}).encode("utf-8")
    req = urllib.request.Request(URL, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        res = json.loads(resp.read())
        print(f"<-- Received response from {URL}!", flush=True)
        return res

print(">>> Testing Turn 1: 'write email for leave letter for school'")
t1_msgs = [{"role": "user", "content": "write email for leave letter for school"}]
r1 = send_chat(t1_msgs)
asst_t1 = r1.get("response", "")
print("\n[TURN 1 RESPONSE]:")
print(asst_t1)
print("\n" + "=" * 60 + "\n")

print(">>> Testing Turn 2: 'Prakash 2, subbaraya Mudali Street, Royapettah CEO 9840705435 Company name - infygalaxy'")
t2_msgs = [
    {"role": "user", "content": "write email for leave letter for school"},
    {"role": "assistant", "content": asst_t1},
    {"role": "user", "content": "Prakash 2, subbaraya Mudali Street, Royapettah CEO 9840705435 Company name - infygalaxy"}
]
r2 = send_chat(t2_msgs)
asst_t2 = r2.get("response", "")
print("\n[TURN 2 RESPONSE]:")
print(asst_t2)
