from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import torchaudio as ta
import torch
import tempfile
import os
import time
from chatterbox.tts_turbo import ChatterboxTurboTTS
from pathlib import Path
import uuid


app = FastAPI(title="Chatterbox Turbo TTS API")

# Load model once at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

VOICE_DIR = Path("voices")
VOICE_DIR.mkdir(exist_ok=True)

@app.post("/voices")
async def register_voice(reference_audio: UploadFile = File(...)):
    start_time = time.time()
    
    voice_id = str(uuid.uuid4())
    wav_path = VOICE_DIR / f"{voice_id}.wav"
    cond_path = VOICE_DIR / f"{voice_id}.pt"

    # Save wav
    with open(wav_path, "wb") as f:
        f.write(await reference_audio.read())

    # Build conditionals
    cond_start = time.time()
    model.prepare_conditionals(str(wav_path))
    cond_time = time.time() - cond_start

    # Save conditionals
    model.conds.save(cond_path)

    total_time = time.time() - start_time
    print(f"[VOICE REGISTRATION] voice_id={voice_id}")
    print(f"  - Conditional preparation: {cond_time:.3f}s")
    print(f"  - Total time: {total_time:.3f}s")

    return {
        "voice_id": voice_id,
        "inference_time": {
            "conditional_preparation": f"{cond_time:.3f}s",
            "total": f"{total_time:.3f}s"
        }
    }

from chatterbox.tts_turbo import Conditionals

@app.post("/tts")
async def tts(
    text: str = Form(...),
    voice_id: str = Form(...)
):
    start_time = time.time()
    
    cond_path = VOICE_DIR / f"{voice_id}.pt"
    if not cond_path.exists():
        return {"error": "Unknown voice_id"}

    # Load cached conditionals
    load_start = time.time()
    model.conds = Conditionals.load(
        cond_path,
        map_location=model.device
    ).to(model.device)
    load_time = time.time() - load_start

    # Generate audio
    gen_start = time.time()
    wav = model.generate(text)
    gen_time = time.time() - gen_start

    # Save audio
    save_start = time.time()
    out_path = VOICE_DIR / f"{voice_id}_out.wav"
    ta.save(out_path, wav, model.sr)
    save_time = time.time() - save_start

    total_time = time.time() - start_time
    
    print(f"[TTS GENERATION] voice_id={voice_id}, text_length={len(text)}")
    print(f"  - Load conditionals: {load_time:.3f}s")
    print(f"  - Generate audio: {gen_time:.3f}s")
    print(f"  - Save audio: {save_time:.3f}s")
    print(f"  - Total time: {total_time:.3f}s")

    return FileResponse(out_path, media_type="audio/wav")


# Optional health check
@app.get("/health")
def health():
    return {"status": "ok", "device": DEVICE}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

