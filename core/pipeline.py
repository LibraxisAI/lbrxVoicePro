"""Core voice processing pipeline"""

import asyncio
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator
import numpy as np

import mlx_whisper
from .audio_recorder import AudioRecorder
from .vad import VoiceActivityDetector


class VoicePipeline:
    """Production-ready voice processing pipeline"""
    
    def __init__(self, model_name: str = "mlx-community/whisper-medium-mlx"):
        self.model_name = model_name
        self.model = None
        self.recorder = AudioRecorder()
        self.vad = VoiceActivityDetector()
        
    async def initialize(self):
        """Load models and initialize components"""
        if not self.model:
            self.model = await asyncio.to_thread(mlx_whisper.load_model, self.model_name)
    
    async def process_audio_stream(self, 
                                 audio_stream: AsyncGenerator[bytes, None],
                                 language: str = "pl") -> AsyncGenerator[Dict[str, Any], None]:
        """Process audio stream with VAD and transcription"""
        
        await self.initialize()
        
        audio_buffer = []
        
        async for chunk in audio_stream:
            # Convert bytes to numpy array
            audio_data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Voice activity detection
            if self.vad.is_speech(audio_data):
                audio_buffer.append(audio_data)
            elif audio_buffer:
                # Process accumulated audio
                full_audio = np.concatenate(audio_buffer)
                
                # Transcribe
                result = await asyncio.to_thread(
                    mlx_whisper.transcribe,
                    full_audio,
                    path_or_hf_repo=self.model_name,
                    language=language,
                    word_timestamps=True
                )
                
                yield {
                    "text": result["text"],
                    "segments": result.get("segments", []),
                    "language": result.get("language", language),
                    "duration": len(full_audio) / 16000
                }
                
                audio_buffer = []
    
    async def transcribe_file(self, 
                            file_path: Path,
                            language: str = "pl") -> Dict[str, Any]:
        """Transcribe audio file"""
        
        await self.initialize()
        
        result = await asyncio.to_thread(
            mlx_whisper.transcribe,
            str(file_path),
            path_or_hf_repo=self.model_name,
            language=language,
            word_timestamps=True
        )
        
        return {
            "text": result["text"],
            "segments": result.get("segments", []),
            "language": result.get("language", language),
            "file": str(file_path)
        }