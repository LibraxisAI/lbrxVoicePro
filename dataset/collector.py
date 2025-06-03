"""Dataset collector for MOSHI/MIMI format"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import soundfile as sf

from ..core import VoicePipeline


class DatasetCollector:
    """Collects voice-text pairs in MOSHI/MIMI format"""
    
    def __init__(self, output_dir: Path = Path("./dataset")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = VoicePipeline()
        self.metadata = []
        
    async def collect_sample(self, 
                           audio_path: Path,
                           speaker_id: str = "default",
                           language: str = "pl") -> Dict[str, Any]:
        """Collect a single audio-text sample"""
        
        # Transcribe audio
        result = await self.pipeline.transcribe_file(audio_path, language)
        
        # Load audio for duration and sample rate
        audio_data, sample_rate = sf.read(audio_path)
        duration = len(audio_data) / sample_rate
        
        # Create sample metadata
        sample = {
            "id": f"{speaker_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "audio_file": str(audio_path.name),
            "text": result["text"],
            "duration": duration,
            "sample_rate": sample_rate,
            "speaker_id": speaker_id,
            "language": language,
            "segments": result.get("segments", []),
            "timestamp": datetime.now().isoformat()
        }
        
        # Save audio to dataset directory
        output_audio = self.output_dir / "audio" / audio_path.name
        output_audio.parent.mkdir(exist_ok=True)
        
        # Copy audio file
        import shutil
        shutil.copy2(audio_path, output_audio)
        
        # Update metadata
        self.metadata.append(sample)
        
        return sample
    
    async def collect_batch(self, 
                          audio_files: List[Path],
                          speaker_id: str = "default",
                          language: str = "pl") -> List[Dict[str, Any]]:
        """Collect multiple samples"""
        
        tasks = [
            self.collect_sample(audio_file, speaker_id, language)
            for audio_file in audio_files
        ]
        
        return await asyncio.gather(*tasks)
    
    def save_metadata(self, format: str = "jsonl"):
        """Save metadata in MOSHI/MIMI compatible format"""
        
        output_file = self.output_dir / f"metadata.{format}"
        
        if format == "jsonl":
            # JSONL format (one JSON object per line)
            with open(output_file, "w", encoding="utf-8") as f:
                for sample in self.metadata:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        elif format == "json":
            # Standard JSON format
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)
        
        # Also save in MOSHI/MIMI specific format
        moshi_format = {
            "version": "1.0",
            "dataset_name": "lbrxVoicePro",
            "language": "pl",
            "total_duration": sum(s["duration"] for s in self.metadata),
            "total_samples": len(self.metadata),
            "speakers": list(set(s["speaker_id"] for s in self.metadata)),
            "samples": self.metadata
        }
        
        with open(self.output_dir / "dataset_moshi.json", "w", encoding="utf-8") as f:
            json.dump(moshi_format, f, ensure_ascii=False, indent=2)
            
        print(f"✅ Saved {len(self.metadata)} samples to {output_file}")