import asyncio
import logging
import uuid
from pathlib import Path
from typing import Literal, Optional

import aiofiles
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import DEFAULT_LANGUAGE, DEFAULT_VOICE
from tts_engine import engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Qwen TTS Web UI", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_candidates = [
    Path("/frontend"),
    Path(__file__).resolve().parent.parent / "frontend",
]
frontend_path = next((p for p in frontend_candidates if p.exists()), None)
if frontend_path:
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    language: str = DEFAULT_LANGUAGE
    response_format: Literal["mp3", "wav", "pcm"] = "mp3"
    stream: bool = False
    instruct: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(asyncio.to_thread(engine.load_model))


@app.get("/")
async def root():
    if frontend_path:
        return RedirectResponse(url="/static/index.html")
    return JSONResponse({"message": "Qwen TTS Web UI backend is running"})


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": engine.model is not None,
        "voices": engine.get_voices(),
    }


@app.get("/voices")
@app.get("/v1/voices")
async def get_voices():
    return {"voices": engine.get_voices()}


@app.post("/v1/audio/speech")
async def generate_speech(request: TTSRequest):
    voices = engine.get_voices()
    if request.voice not in voices:
        raise HTTPException(status_code=400, detail=f"Invalid voice. Available: {voices}")

    try:
        audio_bytes, media_type, sample_rate, ext = await asyncio.to_thread(
            engine.generate_audio,
            request.text,
            request.voice,
            request.language,
            request.response_format,
            request.instruct,
        )
        return StreamingResponse(
            iter([audio_bytes]),
            media_type=media_type,
            headers={
                "Content-Disposition": f'inline; filename="speech.{ext}"',
                "X-Sample-Rate": str(sample_rate),
                "X-Channels": "1",
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Error generating speech")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/v1/audio/stream")
async def stream_speech(request: TTSRequest):
    voices = engine.get_voices()
    if request.voice not in voices:
        raise HTTPException(status_code=400, detail=f"Invalid voice. Available: {voices}")

    async def generate():
        try:
            async for chunk in engine.stream_generate(
                request.text,
                request.voice,
                request.language,
                request.instruct,
            ):
                yield chunk
        except Exception:
            logger.exception("Streaming error")
            raise

    return StreamingResponse(
        generate(),
        media_type="audio/L16",
        headers={
            "Content-Type": "audio/L16",
            "X-Sample-Rate": "24000",
            "X-Channels": "1",
        },
    )


BOOK_UPLOAD_DIR = Path("/tmp/book_uploads")
BOOK_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def decode_text_bytes(content: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1251"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    return content.decode("latin-1", errors="replace")


@app.post("/upload_text")
async def upload_text(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only .txt files are supported for now")

    content = await file.read()
    text = decode_text_bytes(content)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    job_id = str(uuid.uuid4())
    job_path = BOOK_UPLOAD_DIR / job_id
    job_path.mkdir(parents=True, exist_ok=True)
    (job_path / "original.txt").write_text(text, encoding="utf-8")

    return {
        "job_id": job_id,
        "segments": [{"id": i, "text": p[:500]} for i, p in enumerate(paragraphs[:100])],
        "total": len(paragraphs),
    }


async def generate_book_worker(job_id: str, paragraphs: list[str], voice: str, language: str):
    output_dir = BOOK_UPLOAD_DIR / job_id / "audio"
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, para in enumerate(paragraphs):
        try:
            audio_bytes, _, _, _ = await asyncio.to_thread(
                engine.generate_audio,
                para,
                voice,
                language,
                "mp3",
                None,
            )
            out_file = output_dir / f"part_{idx:04d}.mp3"
            async with aiofiles.open(out_file, "wb") as f:
                await f.write(audio_bytes)
        except Exception as e:
            logger.error("Error generating part %s: %s", idx, e)

    logger.info("Book generation finished for %s", job_id)


@app.post("/generate_book/{job_id}")
async def generate_book(
    job_id: str,
    background_tasks: BackgroundTasks,
    voice: str = DEFAULT_VOICE,
    language: str = DEFAULT_LANGUAGE,
):
    job_path = BOOK_UPLOAD_DIR / job_id
    if not job_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    content = (job_path / "original.txt").read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    background_tasks.add_task(generate_book_worker, job_id, paragraphs, voice, language)
    return {"status": "started", "job_id": job_id, "parts": len(paragraphs)}


@app.post("/train_voice")
async def train_voice(
    voice_name: str = Form(...),
    audio_files: list[UploadFile] = File(...),
    transcriptions: list[str] = Form(...),
):
    raise HTTPException(
        status_code=501,
        detail="Voice training is not implemented in this demo. Use external scripts.",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
