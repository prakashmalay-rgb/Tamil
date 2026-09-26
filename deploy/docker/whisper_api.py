import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI(title="Tamil Whisper ASR")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

print("Loading Whisper Large v3...")
model = WhisperModel("large-v3", device="cuda", compute_type="float16")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    audio_bytes = await file.read()
    temp_file = "/tmp/audio_upload"
    with open(temp_file, "wb") as f:
        f.write(audio_bytes)
    segments, info = model.transcribe(temp_file, beam_size=5, language="ta")
    text = " ".join([segment.text for segment in segments])
    return {"text": text.strip(), "language": info.language, "language_probability": info.language_probability}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
