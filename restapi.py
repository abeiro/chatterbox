from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import torchaudio as ta
import torch
import os
import time
from chatterbox.tts_turbo import ChatterboxTurboTTS, Conditionals
from pathlib import Path
from typing import List, Dict
import uuid
import io


app = FastAPI(title="Chatterbox TTS API", version="0.1.0")

# Load model once at startup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = ChatterboxTurboTTS.from_pretrained(device=DEVICE)

VOICE_DIR = Path("voices")
VOICE_DIR.mkdir(exist_ok=True)


# Pydantic models for request/response
class SynthesisRequest(BaseModel):
    text: str
    speaker_wav: str
    language: str
    exaggeration: float = 0.5
    repetition_penalty: float = 1.2
    min_p: float = 0.00
    top_p: float = 0.95
    cfg_weight: float = 0.0
    temperature: float = 0.8
    top_k: int = 1000


# ============= GET ENDPOINTS =============

@app.get("/speakers_list")
async def get_speakers_list():
    """Get list of available speaker names (simplified format)."""
    speaker_names = []
    
    if VOICE_DIR.exists():
        # Get all .wav files in the voices directory
        wav_files = list(VOICE_DIR.glob("*.wav"))
        
        for wav_file in wav_files:
            # Skip output files
            if "_out.wav" in wav_file.name:
                continue
                
            voice_id = wav_file.stem  # filename without extension
            speaker_names.append(voice_id)
    
    return speaker_names


@app.get("/speakers_list_extended")
async def get_speakers_list_extended():
    """Get detailed list of available speaker voice files."""
    speakers = []
    
    if VOICE_DIR.exists():
        # Get all .wav files in the voices directory
        wav_files = list(VOICE_DIR.glob("*.wav"))
        
        for wav_file in wav_files:
            # Skip output files
            if "_out.wav" in wav_file.name:
                continue
                
            voice_id = wav_file.stem  # filename without extension
            cond_file = VOICE_DIR / f"{voice_id}.pt"
            
            speakers.append({
                "voice_id": voice_id,
                "wav_file": wav_file.name,
                "has_conditionals": cond_file.exists()
            })
    
    return {"speakers": speakers, "count": len(speakers)}


@app.get("/speakers")
async def get_speakers_alt():
    """Alternative endpoint to get list of available speakers."""
    return await get_speakers_list()


@app.get("/sample/{file_name}")
async def get_sample(file_name: str):
    """Get a speaker sample audio file."""
    file_path = VOICE_DIR / file_name
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file '{file_name}' not found")
    
    if not file_path.suffix == ".wav":
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    return FileResponse(file_path, media_type="audio/wav", filename=file_name)


# ============= POST ENDPOINTS =============

@app.post("/upload_sample")
async def upload_sample(wavFile: UploadFile = File(...)):
    """Upload a voice sample to be used as a speaker reference."""
    start_time = time.time()
    
    # Validate file type
    if not wavFile.filename.endswith('.wav'):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
    
    # Use original filename (without .wav extension) as voice_id
    voice_id = wavFile.filename[:-4] if wavFile.filename.endswith('.wav') else wavFile.filename
    wav_path = VOICE_DIR / f"{voice_id}.wav"
    cond_path = VOICE_DIR / f"{voice_id}.pt"
    
    # Check if voice already exists
    if wav_path.exists():
        raise HTTPException(status_code=400, detail=f"Voice '{voice_id}' already exists. Please use a different filename.")

    try:
        # Save wav file
        with open(wav_path, "wb") as f:
            content = await wavFile.read()
            f.write(content)

        # Build conditionals
        cond_start = time.time()
        model.prepare_conditionals(str(wav_path))
        cond_time = time.time() - cond_start

        # Save conditionals
        model.conds.save(cond_path)

        total_time = time.time() - start_time
        
        print(f"[VOICE UPLOAD] voice_id={voice_id}, filename={wavFile.filename}")
        print(f"  - Conditional preparation: {cond_time:.3f}s")
        print(f"  - Total time: {total_time:.3f}s")

        return {
            "status": "success",
            "voice_id": voice_id,
            "wav_file": wav_path.name,
            "original_filename": wavFile.filename,
            "inference_time": {
                "conditional_preparation": f"{cond_time:.3f}s",
                "total": f"{total_time:.3f}s"
            }
        }
    
    except Exception as e:
        # Cleanup on error
        if wav_path.exists():
            os.remove(wav_path)
        if cond_path.exists():
            os.remove(cond_path)
        
        raise HTTPException(status_code=500, detail=f"Error processing upload: {str(e)}")


@app.post("/tts_to_audio/")
async def tts_to_audio(request: SynthesisRequest):
    """
    Generate TTS audio from text using a speaker voice.
    
    - **text**: The text to synthesize
    - **speaker_wav**: The voice ID or filename (without extension) of the speaker
    - **language**: Language code (currently not used by model but kept for API compatibility)
    - **exaggeration**: Emotion exaggeration level (default: 0.5)
    - **repetition_penalty**: Penalty for repetition (default: 1.2)
    - **min_p**: Minimum probability threshold (default: 0.0)
    - **top_p**: Top-p sampling threshold (default: 0.95)
    - **cfg_weight**: Classifier-free guidance weight (default: 0.0)
    - **temperature**: Sampling temperature (default: 0.8)
    - **top_k**: Top-k sampling (default: 1000)
    
    Note: CFG weight, min_p and exaggeration are not supported by Turbo version and will be ignored.
    """
    start_time = time.time()
    
    # Parse speaker_wav - could be voice_id or filename
    voice_id = request.speaker_wav
    if voice_id.endswith('.wav'):
        voice_id = voice_id[:-4]  # Remove .wav extension
    
    wav_path = VOICE_DIR / f"{voice_id}.wav"
    
    if not wav_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Voice audio file for '{voice_id}' not found"
        )

    try:
        # Prepare conditionals (compute on the fly, exaggeration can be changed per request)
        cond_start = time.time()
        model.prepare_conditionals(str(wav_path), exaggeration=request.exaggeration)
        cond_time = time.time() - cond_start

        # Generate audio
        gen_start = time.time()
        wav = model.generate(
            request.text,
            repetition_penalty=request.repetition_penalty,
            min_p=request.min_p,
            top_p=request.top_p,
            exaggeration=request.exaggeration,
            cfg_weight=request.cfg_weight,
            temperature=request.temperature,
            top_k=request.top_k
        )
        gen_time = time.time() - gen_start

        # Convert to bytes for streaming
        save_start = time.time()
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        save_time = time.time() - save_start

        total_time = time.time() - start_time
        
        print(f"[TTS GENERATION] voice_id={voice_id}, text_length={len(request.text)}, language={request.language}")
        print(f"  - Prepare conditionals: {cond_time:.3f}s")
        print(f"  - Generate audio: {gen_time:.3f}s")
        print(f"  - Prepare output: {save_time:.3f}s")
        print(f"  - Total time: {total_time:.3f}s")

        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": f"{gen_time:.3f}s",
                "X-Total-Time": f"{total_time:.3f}s"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating audio: {str(e)}")


# ============= HEALTH CHECK =============

@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": DEVICE,
        "model_loaded": model is not None,
        "voice_directory": str(VOICE_DIR)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8020)
