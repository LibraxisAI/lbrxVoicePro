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
│   ├── csm_mlx/        # CSM-1B Polish TTS
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

4. **Polish TTS Integration**
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

Built by LIBRAXIS Team | Optimized for Sesame AI Labs Integration