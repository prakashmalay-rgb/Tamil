"""
Tamil LLM & GPT Studio Server
Serves the professional light-theme frontend and bridges requests to the live Kaggle GPU backend.
"""

import os
import sys
import re
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

# Import Kaggle live bridge
try:
    from sync_kaggle import execute_remote
except ImportError:
    execute_remote = None

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

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Executes live inference directly against the active Qwen 3.6 model in Kaggle GPU memory
    with full multi-turn conversation memory support.
    """
    if not execute_remote:
        return {
            "response": f"வணக்கம்! உங்கள் வினவல் பெறப்பட்டது: '{req.prompt or 'உரையாடல்'}'.",
            "status": "demo"
        }

    # Extract conversation history
    if req.messages and len(req.messages) > 0:
        raw_turns = req.messages
    elif req.prompt:
        raw_turns = [{"role": "user", "content": req.prompt}]
    else:
        return {"response": "வணக்கம்! வினவல் காலியாக உள்ளது.", "status": "empty"}

    tokens_to_generate = min(max(req.max_tokens or 512, 256), 768)

    # Python execution snippet inside Kaggle container using chat template
    py_code = f"""
import sys, torch
try:
    if 'tokenizer' in globals() and 'model' in globals():
        # Ensure mastery adapter is active if available on disk
        import os
        adapter_path = '/kaggle/working/tamil_qwen3_mastery_adapter'
        if os.path.exists(adapter_path) and hasattr(model, 'load_adapter'):
            try:
                if 'mastery' not in getattr(model, 'peft_config', dict()):
                    model.load_adapter(adapter_path, adapter_name='mastery')
                model.set_adapter('mastery')
            except Exception:
                pass

        system_content = (
            "You are a senior, native Tamil language expert and professional AI assistant. "
            "Regardless of whether the user writes in English, Tanglish, or Tamil, ALWAYS respond in fluent, grammatically accurate, pure Tamil (தமிழ்). "
            "When asked to write a letter, email, or official document, IMMEDIATELY draft the full, formal letter directly in proper Tamil (பொருள், மதிப்பிற்குரிய ஐயா, முழுமையான கடித உள்ளடக்கம், இப்படிக்கு). "
            "When the user provides names, addresses, or contact information, IMMEDIATELY embed them seamlessly into the requested letter or task. "
            "CRITICAL: NEVER output an empty list of bracket placeholders like [நீங்கள் பெயர்] or [உங்கள் முகவரி]. Always write the complete, ready-to-use, professional letter in full.\\n\\n"
            "Reference Exemplar:\\n"
            "User: write email for leave letter for school\\n"
            "Assistant:\\n"
            "பொருள்: மருத்துவக் காரணங்களுக்காக விடுப்பு விண்ணப்பம்\\n\\n"
            "மதிப்பிற்குரிய வகுப்பு ஆசிரியர் அவர்களுக்கு,\\n\\n"
            "வணக்கம். என் பெயர் செல்வன் கவின், பத்தாம் வகுப்பு 'அ' பிரிவில் பயின்று வருகிறேன். எனக்கு உடல்நலக் குறைவு மற்றும் காய்ச்சல் ஏற்பட்டுள்ளதால், மருத்துவரின் அறிவுரைப்படி இரண்டு நாட்கள் ஓய்வெடுக்க வேண்டியுள்ளது.\\n\\n"
            "எனவே, வரும் 25-09-2026 முதல் 26-09-2026 வரை எனக்கு விடுப்பு அளித்து உதவுமாறு பணிவுடன் கேட்டுக்கொள்கிறேன். பள்ளிக்குத் திரும்பியவுடன் விடுபட்ட பாடங்களை நிறைவு செய்கிறேன்.\\n\\n"
            "நன்றி.\\n\\n"
            "இப்படிக்கு,\\n"
            "தங்கள் உண்மையுள்ள மாணவன்,\\n"
            "கவின் (பத்தாம் வகுப்பு)."
        )
        raw_turns = {repr(raw_turns)}
        messages = [{{'role': 'system', 'content': system_content}}]
        for m in raw_turns:
            if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and m.get('content'):
                messages.append({{'role': m['role'], 'content': m['content']}})

        try:
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + '<think>\\n\\n</think>\\n'
        except Exception:
            formatted_input = str(raw_turns[-1].get('content', ''))

        inputs = tokenizer(formatted_input, return_tensors='pt').to(model.device)
        gen_kwargs = {{
            'max_new_tokens': {tokens_to_generate},
            'do_sample': True,
            'temperature': 0.2,
            'top_p': 0.85,
            'repetition_penalty': 1.1,
            'pad_token_id': tokenizer.eos_token_id
        }}
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_kwargs)
        res_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print('INFERENCE_OUTPUT_START:' + res_text + ':INFERENCE_OUTPUT_END')
    else:
        print('ERR: model/tokenizer not yet loaded in Kaggle globals.')
except Exception as e:
    print('ERR:' + str(e))
"""
    try:
        raw_res = await execute_remote(py_code, stream_output=False)
        if "INFERENCE_OUTPUT_START:" in raw_res:
            ans = raw_res.split("INFERENCE_OUTPUT_START:")[1].split(":INFERENCE_OUTPUT_END")[0].strip()
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
        elif "ERR:" in raw_res:
            err_msg = raw_res.split("ERR:")[1].strip()
            return {"response": f"Kaggle GPU Runtime: {err_msg}", "status": "runtime_note"}
        else:
            return {"response": raw_res.strip(), "status": "success"}
    except Exception as e:
        err_str = str(e)
        if "timed out during opening handshake" in err_str or "ConnectTimeout" in err_str or "TimeoutError" in err_str:
            friendly_msg = "⚠️ **Kaggle GPU Session Disconnected/Asleep**\n\nThe temporary Kaggle GPU session has timed out due to inactivity. Please open your Kaggle notebook tab, ensure the session is active/running, and refresh."
            return {"response": friendly_msg, "status": "disconnected"}
        return {
            "response": f"Bridge notice: {err_str}",
            "status": "error"
        }

# Determine public/frontend path
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
