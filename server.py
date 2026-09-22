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
    prompt: str
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 0.9

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Executes live inference directly against the active Qwen 3.6 model in Kaggle GPU memory
    """
    if not execute_remote:
        return {
            "response": f"வணக்கம்! உங்கள் வினவல் பெறப்பட்டது: '{req.prompt}'.",
            "status": "demo"
        }

    tokens_to_generate = max(req.max_tokens or 256, 256)

    # Python execution snippet inside Kaggle container using chat template
    py_code = f"""
import sys, torch
try:
    if 'tokenizer' in globals() and 'model' in globals():
        # Ensure mastery adapter is active if available
        if 'mastery_loaded' not in globals():
            import os
            adapter_path = '/kaggle/working/tamil_qwen3_mastery_adapter'
            if os.path.exists(adapter_path) and hasattr(model, 'load_adapter'):
                try:
                    model.load_adapter(adapter_path, adapter_name='mastery')
                    model.set_adapter('mastery')
                except Exception:
                    pass
            globals()['mastery_loaded'] = True

        p = {repr(req.prompt)}
        messages = [
            {{'role': 'system', 'content': 'You are an intelligent, articulate, and knowledgeable AI assistant. Answer helpfully, clearly, and thoughtfully in fluent, grammatically correct Tamil (தமிழ்). Regardless of the input language, formulate a complete and intelligent response in Tamil.'}},
            {{'role': 'user', 'content': p}}
        ]
        try:
            formatted_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            formatted_input = p

        inputs = tokenizer(formatted_input, return_tensors='pt').to(model.device)
        gen_kwargs = {{
            'max_new_tokens': {tokens_to_generate},
            'do_sample': True,
            'temperature': {req.temperature},
            'top_p': {req.top_p},
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
            # Strip internal thought tags cleanly so user gets a direct, intelligent response
            if "</think>" in ans:
                ans = ans.split("</think>")[1].strip()
            elif "<think>" in ans:
                ans = ans.replace("<think>", "").strip()

            # Apply symbolic grammar & Sandhi correction
            try:
                from grammar_validator import validator
                ans = validator.correct_sandhi(ans)
            except Exception:
                pass

            return {"response": ans if ans else "வணக்கம்! உங்களுக்கு எப்படி உதவ முடியும்? (Hello! How can I help you?)", "status": "success"}
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
