import os
import sys
from fastapi import FastAPI
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

class ChatRequest(BaseModel):
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    return {
        "response": f"[Tamil Qwen2.5 Studio - Vercel Edge]\nவணக்கம்! உங்கள் வினவல் பெறப்பட்டது: '{req.prompt}'. மாதிரி பதிலளிக்க தயாராக உள்ளது.",
        "status": "success"
    }

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
