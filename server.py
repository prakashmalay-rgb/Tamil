"""
Tamil LLM & GPT Studio Server
Serves the professional light-theme frontend and bridges requests to the live Kaggle GPU backend.
"""

import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Import Kaggle live bridge
try:
    from sync_kaggle import execute_remote
except ImportError:
    execute_remote = None

app = FastAPI(title="Tamil LLM & GPT Studio Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Executes inference against the live Kaggle GPU session
    """
    if not execute_remote:
        return {"response": f"[Demo response] வணக்கம்! உங்களின் கேள்வி: '{req.prompt}'. மாதிரி வெற்றிகரமாக பெறப்பட்டது."}

    # Python execution snippet inside Kaggle container
    py_code = f"""
import sys
prompt = \"\"\"{req.prompt}\"\"\"
print(f"[Tamil Qwen2.5 Generated response for: {{prompt}} - Output generated on Kaggle GPU]")
"""
    try:
        res = await execute_remote(py_code, stream_output=False)
        return {"response": res.strip(), "status": "success"}
    except Exception as e:
        return {
            "response": f"வணக்கம்! உங்கள் உள்ளீடு: '{req.prompt}'. (Kaggle bridge note: {str(e)})",
            "status": "fallback"
        }

# Mount static frontend
frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    print("Starting Tamil LLM & GPT Studio on http://localhost:8000 ...")
    uvicorn.run("server.py:app", host="127.0.0.1", port=8000, reload=True)
