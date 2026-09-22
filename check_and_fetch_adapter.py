"""
Polls Kaggle kernel state until idle, checks saved adapter, and pulls checkpoint.
"""
import time
import json
import urllib.request
import asyncio
import sync_kaggle

STATUS_URL = f"https://{sync_kaggle.PROXY_BASE}/api/kernels/{sync_kaggle.KERNEL_ID}"

def get_kernel_state():
    try:
        req = urllib.request.Request(STATUS_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return data.get("execution_state", "unknown")
    except Exception as e:
        return f"err: {e}"

async def main():
    print(f"Monitoring Kaggle Kernel {sync_kaggle.KERNEL_ID}...")
    while True:
        state = get_kernel_state()
        print(f"[{time.strftime('%H:%M:%S')}] Kernel execution state: {state}", flush=True)
        if state == "idle":
            print("\n>>> Kernel is IDLE! Verifying saved adapter directory on Kaggle...")
            check_code = """
import os
adapter_dir = "/kaggle/working/tamil_qwen3_mastery_adapter"
if os.path.exists(adapter_dir):
    files = os.listdir(adapter_dir)
    print("ADAPTER_FOUND:", files)
else:
    print("ADAPTER_NOT_FOUND")
"""
            out = await sync_kaggle.execute_remote(check_code)
            print(out)
            break
        elif "err" in state:
            print(f"Warning: {state}")
        await asyncio.sleep(10)

if __name__ == "__main__":
    asyncio.run(main())
