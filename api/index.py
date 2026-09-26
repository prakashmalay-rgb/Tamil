import os
import sys
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from typing import Optional

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

app = FastAPI(title="Tamil LLM & GPT Studio", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RunPod text engine base URL, e.g. https://<pod-id>-8000.proxy.runpod.net
# Changes whenever the pod is redeployed -- set/update this in Vercel's
# project environment variables rather than editing code.
RUNPOD_TEXT_URL = os.environ.get("RUNPOD_TEXT_URL", "").rstrip("/")
RUNPOD_MODEL = os.environ.get("RUNPOD_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")

class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    if not RUNPOD_TEXT_URL:
        raise HTTPException(
            status_code=503,
            detail="RUNPOD_TEXT_URL is not configured. Set it in Vercel's "
                   "project environment variables to the pod's text engine "
                   "URL (e.g. https://<pod-id>-8000.proxy.runpod.net).",
        )

    payload = {
        "model": RUNPOD_MODEL,
        "messages": [{"role": "user", "content": req.prompt}],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "top_p": req.top_p,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{RUNPOD_TEXT_URL}/v1/chat/completions", json=payload
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=502,
            detail=f"RunPod backend returned an error: {e.response.status_code} {e.response.text}",
        )
    except httpx.RequestError as e:
        raise HTTPException(
            status_code=504,
            detail=f"Could not reach RunPod backend: {e}",
        )

    reply = data["choices"][0]["message"]["content"]
    return {"response": reply, "status": "success"}

def get_file_content(subpath: str) -> str:
    candidates = [
        os.path.join(BASE_DIR, "public", subpath),
        os.path.join(BASE_DIR, "frontend", subpath),
        os.path.join(BASE_DIR, subpath),
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
    return ""

@app.get("/", response_class=HTMLResponse)
async def serve_root():
    content = get_file_content("index.html")
    if content:
        return HTMLResponse(content=content)
    return HTMLResponse("<h1>Tamil LLM Studio is loading...</h1>")

@app.get("/css/{filename}")
async def serve_css(filename: str):
    content = get_file_content(os.path.join("css", filename))
    return Response(content=content, media_type="text/css")

@app.get("/js/{filename}")
async def serve_js(filename: str):
    content = get_file_content(os.path.join("js", filename))
    return Response(content=content, media_type="application/javascript")
