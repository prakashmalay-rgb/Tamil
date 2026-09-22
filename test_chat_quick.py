import urllib.request
import json
import time

url = "http://127.0.0.1:8000/chat"
body = {
    "prompt": "write email for leave letter for school",
    "max_tokens": 250
}
req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), headers={"Content-Type": "application/json"})
t0 = time.time()
print("Sending request...")
with urllib.request.urlopen(req, timeout=60) as resp:
    res = json.loads(resp.read().decode("utf-8"))
    print(f"Elapsed: {time.time() - t0:.2f}s")
    print("Response:")
    print(res.get("response"))
