"""Test core components"""

import pytest


def test_imports():
    """Test that core imports work"""
    from core import AudioRecorder, VoicePipeline, VoiceActivityDetector
    
    assert AudioRecorder is not None
    assert VoicePipeline is not None
    assert VoiceActivityDetector is not None


def test_audio_recorder():
    """Test audio recorder initialization"""
    from core.audio_recorder import AudioRecorder
    
    recorder = AudioRecorder()
    assert recorder.sample_rate == 16000
    assert recorder.channels == 1


@pytest.mark.asyncio
async def test_pipeline_init():
    """Test pipeline initialization"""
    from core.pipeline import VoicePipeline
    
    pipeline = VoicePipeline()
    assert pipeline.model_name == "mlx-community/whisper-medium-mlx"
    assert pipeline.model is None  # Not loaded yet


def test_dataset_collector():
    """Test dataset collector"""
    from dataset.collector import DatasetCollector
    
    collector = DatasetCollector()
    assert collector.output_dir.exists()
    assert len(collector.metadata) == 0


def test_formatter():
    """Test MOSHI/MIMI formatter"""
    from dataset.formatter import MoshiMimiFormatter
    
    samples = [{
        "id": "test_001",
        "audio_file": "test.wav",
        "text": "Test transcription",
        "duration": 2.5,
        "speaker_id": "test_speaker",
        "language": "pl"
    }]
    
    moshi_data = MoshiMimiFormatter.to_moshi_format(samples)
    assert moshi_data["version"] == "1.0"
    assert len(moshi_data["utterances"]) == 1
    
    mimi_data = MoshiMimiFormatter.to_mimi_format(samples)
    assert mimi_data["codec"] == "mimi"
    assert len(mimi_data["data"]) == 1