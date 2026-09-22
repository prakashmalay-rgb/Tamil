"""
Tamil LLM & GPT Studio Server
Serves the professional light-theme frontend and bridges requests to the live Kaggle GPU backend.
"""

import os
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
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
    max_tokens: Optional[int] = 256
    top_p: Optional[float] = 0.9

@app.post("/chat")
@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Executes live inference directly against the active Qwen 3 model in Kaggle GPU memory
    """
    if not execute_remote:
        return {
            "response": f"[Kaggle Live Bridge Demo]\nவணக்கம்! உங்கள் வினவல் பெறப்பட்டது: '{req.prompt}'.",
            "status": "demo"
        }

    # Python execution snippet inside Kaggle container using loaded model and tokenizer
    py_code = f"""
import sys
try:
    p = {repr(req.prompt)}
    if 'tokenizer' in globals() and 'model' in globals():
        inputs = tokenizer(p, return_tensors='pt').to(model.device)
        gen_kwargs = {{
            'max_new_tokens': {req.max_tokens},
            'do_sample': True,
            'temperature': {req.temperature},
            'top_p': {req.top_p},
            'pad_token_id': tokenizer.eos_token_id
        }}
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
            return {"response": ans if ans else "[Model generated empty text]", "status": "success"}
        elif "ERR:" in raw_res:
            err_msg = raw_res.split("ERR:")[1].strip()
            return {"response": f"Kaggle Runtime: {err_msg}", "status": "runtime_note"}
        else:
            return {"response": raw_res.strip(), "status": "success"}
    except Exception as e:
        return {
            "response": f"Bridge notice: {str(e)}",
            "status": "error"
        }

# Determine public/frontend path
base_dir = os.path.dirname(os.path.abspath(__file__))
public_dir = os.path.join(base_dir, "public")
if not os.path.exists(public_dir):
    public_dir = os.path.join(base_dir, "frontend")

@app.get("/")
async def root_index():
    index_file = os.path.join(public_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Tamil LLM Studio API is running"}

if os.path.exists(public_dir):
    app.mount("/static", StaticFiles(directory=public_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Tamil LLM & GPT Studio on http://localhost:8000 ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
