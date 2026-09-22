"""
Kaggle Live Bridge & Sync Utility
Allows real-time code execution, file pushing, and file pulling against a live Kaggle kernel proxy.
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import uuid
import websockets

PROXY_BASE = "kkb-production.jupyter-proxy.kaggle.net/k/351751202/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..5oERZibGz-jHU-6Jn-7ZjA.P3FiKWhO_fqNAmJh2YBtKafyv3RU8RqEna65NhUsKZXFRSg6e3ydHvCgvpW2CtdgnGGl029hN1iL_TpSPd0QmWp6ijeHmekhZf4fGwcjlOAIrlFHSpxx6ukA29Ucxny0DKsm8tOrlyAEkVKN8wWSNiLzl4QtZpz7-S6axYLQ6yxqc6YdcMsRaJDFe0MGhH7TXt9kBvR2hwIB_MBnIvIBRaTpHM3ZbtCq5rDdovEGy33oGWFpdvG6yJH_x3KHqKEe.f2Nt9fg2DvopApUdClOG8w/proxy"
KERNEL_ID = "9d151a2d-4867-46ee-813a-43c42a937ab9"
WS_URL = f"wss://{PROXY_BASE}/api/kernels/{KERNEL_ID}/channels"

async def execute_remote(code: str, stream_output: bool = True) -> str:
    output = []
    async with websockets.connect(WS_URL, max_size=50 * 1024 * 1024) as ws:
        msg_id = uuid.uuid4().hex
        msg = {
            "header": {
                "msg_id": msg_id,
                "username": "local_dev",
                "session": uuid.uuid4().hex,
                "msg_type": "execute_request",
                "version": "5.3"
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": True,
                "user_expressions": {},
                "allow_stdin": False
            },
            "buffers": [],
            "channel": "shell"
        }
        await ws.send(json.dumps(msg))

        while True:
            res = await ws.recv()
            data = json.loads(res)
            msg_type = data.get("msg_type") or data.get("header", {}).get("msg_type")
            content = data.get("content", {})
            if msg_type == "stream":
                text = content.get("text", "")
                output.append(text)
                if stream_output:
                    sys.stdout.write(text)
                    sys.stdout.flush()
            elif msg_type in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain", "")
                output.append(text + "\n")
                if stream_output:
                    print(text)
            elif msg_type == "error":
                err = f"ERROR: {content.get('ename')}: {content.get('evalue')}\n"
                output.append(err)
                if stream_output:
                    sys.stderr.write(err)
            elif msg_type == "status" and content.get("execution_state") == "idle":
                if data.get("parent_header", {}).get("msg_id") == msg_id:
                    break
    return "".join(output)

async def check_status():
    print("=== Checking Kaggle GPU & System Status ===")
    code = "!nvidia-smi; free -h; df -h /kaggle/working"
    await execute_remote(code)

async def push_file(local_path: str, remote_dest: str = "/kaggle/working"):
    if not os.path.exists(local_path):
        print(f"Error: Local file '{local_path}' does not exist.")
        return
    filename = os.path.basename(local_path)
    remote_path = f"{remote_dest}/{filename}"
    print(f"Pushing {local_path} -> {remote_path}...")
    with open(local_path, "rb") as f:
        b64_content = base64.b64encode(f.read()).decode("ascii")

    code = f"""
import base64
data = base64.b64decode("{b64_content}")
with open("{remote_path}", "wb") as f:
    f.write(data)
print("Saved: {remote_path} (" + str(len(data)) + " bytes)")
"""
    await execute_remote(code)

async def pull_files(remote_dir: str = "/kaggle/working", local_dest_dir: str = "downloaded_outputs"):
    os.makedirs(local_dest_dir, exist_ok=True)
    print(f"Listing files in {remote_dir}...")
    list_code = f"""
import os, json
files = [f for f in os.listdir("{remote_dir}") if not f.startswith('.')]
print('REMOTE_FILES:' + json.dumps(files))
"""
    out = await execute_remote(list_code, stream_output=False)
    if "REMOTE_FILES:" not in out:
        print("No files found or unable to list.")
        return
    files = json.loads(out.split("REMOTE_FILES:")[1].strip())
    print(f"Files available to download: {files}")
    for fname in files:
        fetch_code = f"""
import base64
try:
    with open("{remote_dir}/{fname}", "rb") as f:
        print("DATA_START:" + base64.b64encode(f.read()).decode("ascii") + ":DATA_END")
except Exception as e:
    print("ERR:" + str(e))
"""
        res = await execute_remote(fetch_code, stream_output=False)
        if "DATA_START:" in res:
            b64_data = res.split("DATA_START:")[1].split(":DATA_END")[0]
            data = base64.b64decode(b64_data)
            out_file = os.path.join(local_dest_dir, fname)
            with open(out_file, "wb") as f:
                f.write(data)
            print(f"Downloaded {fname} ({len(data)} bytes) -> {out_file}")

def main():
    parser = argparse.ArgumentParser(description="Kaggle live sync utility")
    parser.add_argument("--status", action="store_true", help="Check remote GPU/system status")
    parser.add_argument("--exec", type=str, help="Execute arbitrary command or Python code remotely")
    parser.add_argument("--push", type=str, help="Push local file to /kaggle/working")
    parser.add_argument("--pull", action="store_true", help="Pull output files from /kaggle/working to local folder")
    args = parser.parse_args()

    if args.status:
        asyncio.run(check_status())
    elif args.exec:
        asyncio.run(execute_remote(args.exec))
    elif args.push:
        asyncio.run(push_file(args.push))
    elif args.pull:
        asyncio.run(pull_files())
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
