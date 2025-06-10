# lbrxVoicePro - Universal Voice AI Pipeline

## Architecture Overview

Production-ready voice processing pipeline for dataset collection and conversational AI, built on Apple Silicon MLX.

### Core Components

```
lbrxVoicePro/
├── core/                 # Core voice processing engine
│   ├── whisper.py       # MLX Whisper ASR
│   ├── vad.py           # Voice Activity Detection
│   └── pipeline.py      # Audio processing pipeline
├── dataset/             # MOSHI/MIMI format dataset tools
│   ├── collector.py     # Audio-text pair collection
│   ├── formatter.py     # MOSHI/MIMI format conversion
│   └── validator.py     # Dataset quality checks
├── models/              # Model integrations
│   ├── csm_mlx/        # CSM-MLX TTS
│   ├── whisper_mlx/    # Whisper models
│   └── rag/            # Universal RAG engine
├── api/                 # REST/WebSocket APIs
│   ├── transcription/  # ASR endpoints
│   ├── synthesis/      # TTS endpoints
│   └── conversation/   # Conversational AI
└── ui/                  # User interfaces
    ├── tui/            # Terminal UI
    └── web/            # Web interface

```

### Key Features

1. **Voice Processing Pipeline**
   - Real-time ASR with MLX Whisper
   - Voice Activity Detection
   - Audio preprocessing & normalization

2. **Dataset Collection** 
   - MOSHI/MIMI compatible format
   - Automatic quality validation
   - Batch processing capabilities

3. **Universal RAG System**
   - Domain-agnostic knowledge base
   - Plug any corpus (technical docs, literature, etc.)
   - ChromaDB vector storage

4. **TTS Integration**
   - CSM-1B model support
   - Real-time synthesis
   - Voice cloning ready

5. **Conversational AI**
   - Local LLM integration (LM Studio)
   - Context-aware responses
   - Multi-turn conversations

### Technical Stack

- **Framework**: Python 3.11+ with UV package manager
- **ML Backend**: MLX (Apple Silicon optimized)
- **ASR**: MLX Whisper (Polish optimized)
- **TTS**: CSM-MLX (Sesame Labs)
- **LLM**: Local models via LM Studio
- **Vector DB**: ChromaDB
- **APIs**: FastAPI + WebSockets
- **UI**: Textual (TUI) + Next.js (Web)

### Performance Metrics

- ASR Latency: <100ms (on M1 Pro)
- TTS Latency: <200ms 
- RAG Query: <50ms
- Concurrent users: 10+

### Use Cases

1. **Dataset Creation**: Collect voice-text pairs for TTS training
2. **Domain Expert AI**: Load any knowledge base → instant expert
3. **Voice Assistants**: Build conversational interfaces
4. **Transcription Services**: Batch/real-time transcription

---

## Credits & Acknowledgments

This project integrates and builds upon several excellent open-source models:

### Speech Recognition
- **[mlx-community/whisper](https://huggingface.co/mlx-community)** - MLX-optimized Whisper models for ASR
  - Apache 2.0 License
  - Optimized for Apple Silicon by the MLX Community

### Text-to-Speech
- **[senstella/csm-mlx](https://github.com/senstella/csm-mlx)** - CSM-1B with native MLX support SoTATTS model
  - MIT License
  - State-of-the-art speech synthesis
  
- **[nari-labs/Dia-1.6B](https://huggingface.co/nari-labs/Dia-1.6B)** - Multilingual TTS model
  - Apache 2.0 License
  - Supports multiple languages including 
  
- **[coqui-ai/XTTS-v2](https://github.com/coqui-ai/TTS)** - Multi-lingual TTS model
  - Mozilla Public License 2.0
  - Voice cloning capabilities

### Voice Activity Detection
- **[Silero VAD](https://github.com/snakers4/silero-vad)** - Pre-trained VAD
  - MIT License

---

Built by LIBRAXIS Team | Optimized for Sesame AI Labs Integration
