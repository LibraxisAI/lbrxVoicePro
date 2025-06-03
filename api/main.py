"""Main API server for lbrxVoicePro"""

from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from pathlib import Path
import asyncio
from typing import Optional

from ..core import VoicePipeline, AudioRecorder
from ..dataset import DatasetCollector
from ..models.csm_mlx import CSMVoiceSynthesizer
from ..models.rag import RAGEngine

app = FastAPI(title="lbrxVoicePro API", version="1.0.0")

# Initialize components
pipeline = VoicePipeline()
synthesizer = CSMVoiceSynthesizer()
collector = DatasetCollector()
rag_engine = RAGEngine()


@app.on_event("startup")
async def startup():
    """Initialize models on startup"""
    await pipeline.initialize()
    await synthesizer.initialize()
    print("✅ lbrxVoicePro API ready")


@app.get("/")
async def root():
    return {
        "name": "lbrxVoicePro",
        "version": "1.0.0",
        "endpoints": {
            "transcription": "/api/v1/transcribe",
            "synthesis": "/api/v1/synthesize",
            "dataset": "/api/v1/dataset/collect",
            "rag": "/api/v1/rag/query"
        }
    }


@app.post("/api/v1/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = "pl"
):
    """Transcribe audio file"""
    
    # Save uploaded file
    temp_path = Path(f"/tmp/{file.filename}")
    with open(temp_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Transcribe
    result = await pipeline.transcribe_file(temp_path, language)
    
    # Clean up
    temp_path.unlink()
    
    return JSONResponse(content=result)


@app.websocket("/api/v1/transcribe/stream")
async def transcribe_stream(websocket: WebSocket):
    """Real-time transcription via WebSocket"""
    
    await websocket.accept()
    
    try:
        async def audio_generator():
            while True:
                data = await websocket.receive_bytes()
                if not data:
                    break
                yield data
        
        # Process audio stream
        async for result in pipeline.process_audio_stream(audio_generator()):
            await websocket.send_json(result)
            
    except Exception as e:
        await websocket.send_json({"error": str(e)})
    finally:
        await websocket.close()


@app.post("/api/v1/synthesize")
async def synthesize_speech(
    text: str,
    speaker_id: Optional[str] = None,
    temperature: float = 0.7
):
    """Synthesize speech from text"""
    
    audio = await synthesizer.synthesize(text, speaker_id, temperature)
    
    # Return audio as streaming response
    return StreamingResponse(
        io.BytesIO(audio.tobytes()),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )


@app.post("/api/v1/dataset/collect")
async def collect_dataset_sample(
    file: UploadFile = File(...),
    speaker_id: str = "default",
    language: str = "pl"
):
    """Collect audio sample for dataset"""
    
    # Save uploaded file
    audio_path = Path(f"/tmp/{file.filename}")
    with open(audio_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Collect sample
    sample = await collector.collect_sample(audio_path, speaker_id, language)
    
    # Clean up
    audio_path.unlink()
    
    return JSONResponse(content=sample)


@app.post("/api/v1/rag/query")
async def query_rag(
    query: str,
    top_k: int = 5
):
    """Query RAG knowledge base"""
    
    results = await rag_engine.query(query, top_k)
    
    return JSONResponse(content={
        "query": query,
        "results": results
    })


@app.post("/api/v1/rag/index")
async def index_documents(
    documents: list[str]
):
    """Index documents in RAG"""
    
    await rag_engine.index_documents(documents)
    
    return JSONResponse(content={
        "status": "success",
        "indexed": len(documents)
    })


if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )