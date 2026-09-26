"""
Tamil LLM & GPT Studio Server
Serves the professional light-theme frontend and bridges requests to the
vLLM text engine running on RunPod.
"""

import os
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# RunPod text engine base URL, e.g. https://<pod-id>-8000.proxy.runpod.net
# Changes whenever the pod is redeployed -- set/update this via the
# RUNPOD_TEXT_URL environment variable rather than editing code.
RUNPOD_TEXT_URL = os.environ.get("RUNPOD_TEXT_URL", "").rstrip("/")
RUNPOD_MODEL = os.environ.get("RUNPOD_MODEL", "Qwen/Qwen2.5-72B-Instruct-AWQ")

app = FastAPI(title="Tamil LLM Qwen 3.6 Studio Server", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[list] = None
    temperature: Optional[float] = 0.3
    max_tokens: Optional[int] = 512
    top_p: Optional[float] = 0.9

SYSTEM_PROMPT = (
    "You are a senior, native Tamil language expert and professional AI assistant. "
    "Regardless of whether the user writes in English, Tanglish, or Tamil, ALWAYS respond in fluent, grammatically accurate, pure Tamil (தமிழ்). "
    "When asked to write a letter, email, or official document, IMMEDIATELY draft the full, formal letter directly in proper Tamil (பொருள், மதிப்பிற்குரிய ஐயா, முழுமையான கடித உள்ளடக்கம், இப்படிக்கு). "
    "When the user provides names, addresses, or contact information, IMMEDIATELY embed them seamlessly into the requested letter or task. "
    "CRITICAL: NEVER output an empty list of bracket placeholders like [நீங்கள் பெயர்] or [உங்கள் முகவரி]. Always write the complete, ready-to-use, professional letter in full.\n\n"
    "Reference Exemplar:\n"
    "User: write email for leave letter for school\n"
    "Assistant:\n"
    "பொருள்: மருத்துவக் காரணங்களுக்காக விடுப்பு விண்ணப்பம்\n\n"
    "மதிப்பிற்குரிய வகுப்பு ஆசிரியர் அவர்களுக்கு,\n\n"
    "வணக்கம். என் பெயர் செல்வன் கவின், பத்தாம் வகுப்பு 'அ' பிரிவில் பயின்று வருகிறேன். எனக்கு உடல்நலக் குறைவு மற்றும் காய்ச்சல் ஏற்பட்டுள்ளதால், மருத்துவரின் அறிவுரைப்படி இரண்டு நாட்கள் ஓய்வெடுக்க வேண்டியுள்ளது.\n\n"
    "எனவே, வரும் 25-09-2026 முதல் 26-09-2026 வரை எனக்கு விடுப்பு அளித்து உதவுமாறு பணிவுடன் கேட்டுக்கொள்கிறேன். பள்ளிக்குத் திரும்பியவுடன் விடுபட்ட பாடங்களை நிறைவு செய்கிறேன்.\n\n"
    "நன்றி.\n\n"
    "இப்படிக்கு,\n"
    "தங்கள் உண்மையுள்ள மாணவன்,\n"
    "கவின் (பத்தாம் வகுப்பு)."
)

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Executes live inference against the Qwen2.5-72B-AWQ model served by
    vLLM on RunPod, with full multi-turn conversation memory support.
    """
    if not RUNPOD_TEXT_URL:
        return {
            "response": f"வணக்கம்! உங்கள் வினவல் பெறப்பட்டது: '{req.prompt or 'உரையாடல்'}'. "
                        "(RUNPOD_TEXT_URL is not configured on the server.)",
            "status": "demo"
        }

    # Extract conversation history
    if req.messages and len(req.messages) > 0:
        raw_turns = req.messages
    elif req.prompt:
        raw_turns = [{"role": "user", "content": req.prompt}]
    else:
        return {"response": "வணக்கம்! வினவல் காலியாக உள்ளது.", "status": "empty"}

    tokens_to_generate = min(max(req.max_tokens or 256, 128), 512)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in raw_turns:
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content"):
            messages.append({"role": m["role"], "content": m["content"]})

    payload = {
        "model": RUNPOD_MODEL,
        "messages": messages,
        "temperature": req.temperature or 0.2,
        "top_p": req.top_p or 0.85,
        "max_tokens": tokens_to_generate,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(f"{RUNPOD_TEXT_URL}/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()
        ans = data["choices"][0]["message"]["content"].strip()

        # 1. Thought-leak fail-safe: Strip internal reasoning tokens cleanly
        if "</think>" in ans:
            ans = ans.split("</think>")[1].strip()
        elif "<think>" in ans:
            ans = ans.replace("<think>", "").strip()

        # 2. Neuro-symbolic grammar, Sandhi, and Subject-Verb agreement fail-safe
        try:
            from grammar_validator import validator
            ans = validator.correct_sandhi(ans)
        except Exception:
            pass

        # 3. Truncation fail-safe: Ensure graceful termination
        if not ans:
            ans = "வணக்கம்! உங்களுக்கு நான் எவ்வாறு உதவ முடியும்?"

        return {"response": ans, "status": "success"}
    except httpx.HTTPStatusError as e:
        return {
            "response": f"RunPod GPU Runtime: {e.response.status_code} {e.response.text}",
            "status": "runtime_note",
        }
    except httpx.RequestError as e:
        friendly_msg = (
            "⚠️ **RunPod GPU backend unreachable**\n\n"
            f"Could not reach the RunPod text engine at {RUNPOD_TEXT_URL}. "
            "Check that the pod is running and RUNPOD_TEXT_URL still matches "
            f"its current proxy URL.\n\nDetails: {e}"
        )
        return {"response": friendly_msg, "status": "disconnected"}
    except Exception as e:
        return {
            "response": f"Bridge notice: {e}",
            "status": "error"
        }

class ScansionRequest(BaseModel):
    poem_text: str

class TVAFeedRequest(BaseModel):
    raw_text: str
    title: Optional[str] = "TVA பொது நூல்"
    author: Optional[str] = "தமிழ் இணையக் கல்விக்கழகம்"
    era: Optional[str] = "சங்க காலம்"
    category: Optional[str] = "இலக்கியம் / இலக்கணம்"

@app.post("/api/pedagogy/scansion")
async def pedagogy_scansion_endpoint(req: ScansionRequest):
    """
    Scans classical Tamil poetry (Venba, Kural) into metrical seers (அசைகள் & வாய்பாடுகள்).
    """
    try:
        from tamil_pedagogy_engine import pedagogy_engine
        res = pedagogy_engine.scan_kural(req.poem_text)
        return {"status": "success", "data": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/tva/ingest_text")
async def tva_ingest_endpoint(req: TVAFeedRequest):
    """
    Ingests TVA book passages into both RAG memory and SFT training curriculum.
    """
    try:
        from tva_ingestor import ingestor
        meta = {
            "title": req.title,
            "author": req.author,
            "era": req.era,
            "category": req.category
        }
        res = ingestor.process_and_save_book(req.raw_text, meta)
        return {"status": "success", "data": res}
    except Exception as e:
        return {"status": "error", "message": str(e)}

base_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(base_dir, "public")
if not os.path.exists(public_dir):
    public_dir = os.path.join(base_dir, "frontend")

# Mount static CSS & JS
css_dir = os.path.join(public_dir, "css")
js_dir = os.path.join(public_dir, "js")

if os.path.exists(css_dir):
    app.mount("/css", StaticFiles(directory=css_dir), name="css")

if os.path.exists(js_dir):
    app.mount("/js", StaticFiles(directory=js_dir), name="js")

@app.get("/")
async def root_index():
    index_file = os.path.join(public_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Tamil LLM Studio API is running"}

if os.path.exists(public_dir):
    app.mount("/", StaticFiles(directory=public_dir, html=True), name="static_root")

if __name__ == "__main__":
    import uvicorn
    print("Starting Tamil LLM Qwen 3.6 Studio on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
