import os
import logging
import uuid
import asyncio
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import aiofiles

from tts_engine import engine, VOICES, SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen TTS Web UI", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_path = Path(__file__).parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

class TTSRequest(BaseModel):
    text: str
    voice: str = "Vivian"
    language: str = "Russian"
    response_format: str = "mp3"
    stream: bool = False

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(engine.load_model))

@app.get("/")
async def root():
    return {"message": "Qwen TTS Web UI backend. Visit /static/index.html for UI."}

@app.get("/voices")
async def get_voices():
    return {"voices": VOICES}

@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    if request.voice not in VOICES:
        raise HTTPException(status_code=400, detail="Invalid voice")
    try:
        audio_bytes = await asyncio.to_thread(
            engine.generate_audio,
            request.text,
            request.voice,
            request.language
        )
        media_type = "audio/mpeg" if request.response_format == "mp3" else "audio/wav"
        return StreamingResponse(
            iter([audio_bytes]),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename=speech.{request.response_format}"}
        )
    except Exception as e:
        logger.exception("Error generating speech")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/audio/stream")
async def stream_speech(request: TTSRequest):
    if not request.stream:
        return await generate_speech(request)
    if request.voice not in VOICES:
        raise HTTPException(status_code=400, detail="Invalid voice")
    async def generate():
        try:
            async for chunk in engine.stream_generate(request.text, request.voice, request.language):
                yield chunk
        except Exception as e:
            logger.exception("Streaming error")
    return StreamingResponse(
        generate(),
        media_type="audio/L16",
        headers={
            "Content-Type": "audio/L16",
            "X-Sample-Rate": str(SAMPLE_RATE),
            "X-Channels": "1"
        }
    )

BOOK_UPLOAD_DIR = Path("/tmp/book_uploads")
BOOK_UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload_text")
async def upload_text(file: UploadFile = File(...)):
    if not file.filename.endswith(".txt"):
        raise HTTPException(400, "Only .txt files are supported for now")
    content = await file.read()
    text = content.decode("utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    job_id = str(uuid.uuid4())
    job_path = BOOK_UPLOAD_DIR / job_id
    job_path.mkdir()
    (job_path / "original.txt").write_bytes(content)
    return {
        "job_id": job_id,
        "segments": [{"id": i, "text": p[:500]} for i, p in enumerate(paragraphs)],
        "total": len(paragraphs)
    }

@app.post("/generate_book/{job_id}")
async def generate_book(job_id: str, background_tasks: BackgroundTasks):
    job_path = BOOK_UPLOAD_DIR / job_id
    if not job_path.exists():
        raise HTTPException(404, "Job not found")
    content = (job_path / "original.txt").read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    background_tasks.add_task(generate_book_worker, job_id, paragraphs)
    return {"status": "started", "job_id": job_id}

async def generate_book_worker(job_id: str, paragraphs: list):
    output_dir = BOOK_UPLOAD_DIR / job_id / "audio"
    output_dir.mkdir(exist_ok=True)
    for idx, para in enumerate(paragraphs):
        try:
            audio_bytes = await asyncio.to_thread(
                engine.generate_audio,
                para,
                "Vivian",
                "Russian"
            )
            out_file = output_dir / f"part_{idx:04d}.mp3"
            async with aiofiles.open(out_file, "wb") as f:
                await f.write(audio_bytes)
        except Exception as e:
            logger.error(f"Error generating part {idx}: {e}")
    logger.info(f"Book generation finished for {job_id}")

@app.post("/train_voice")
async def train_voice(
    voice_name: str = Form(...),
    audio_files: list[UploadFile] = File(...),
    transcriptions: list[str] = Form(...)
):
    raise HTTPException(501, "Voice training is not implemented in this demo. Use external scripts.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)