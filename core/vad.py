"""Voice Activity Detection using Silero VAD"""

import torch
import numpy as np
from typing import Tuple


class VoiceActivityDetector:
    """Production VAD using Silero"""
    
    def __init__(self, threshold: float = 0.5, sampling_rate: int = 16000):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.model = None
        self._load_model()
        
    def _load_model(self):
        """Load Silero VAD model"""
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        self.model.eval()
        
    def is_speech(self, audio: np.ndarray) -> bool:
        """Check if audio contains speech"""
        if len(audio) < 512:  # Minimum window size
            return False
            
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech probability
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sampling_rate).item()
            
        return speech_prob >= self.threshold
    
    def get_speech_segments(self, audio: np.ndarray, 
                          window_size_ms: int = 30) -> list[Tuple[int, int]]:
        """Get speech segments from audio"""
        window_size = int(self.sampling_rate * window_size_ms / 1000)
        segments = []
        
        current_segment = None
        
        for i in range(0, len(audio) - window_size, window_size):
            window = audio[i:i + window_size]
            
            if self.is_speech(window):
                if current_segment is None:
                    current_segment = [i, i + window_size]
                else:
                    current_segment[1] = i + window_size
            else:
                if current_segment is not None:
                    segments.append(tuple(current_segment))
                    current_segment = None
        
        if current_segment is not None:
            segments.append(tuple(current_segment))
            
        return segments