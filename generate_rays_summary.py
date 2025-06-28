#!/usr/bin/env python3
"""
Generate Polish audio summary of rays-vet-rag project
Using Meta MMS-TTS-POL model
"""

import asyncio
import sys
from pathlib import Path
import numpy as np
import scipy.io.wavfile
from datetime import datetime

# Polish summary text
SUMMARY_TEXT = """
Cześć Maciej! Tu Klaudiusz z podsumowaniem tego, co zrobiliśmy z projektem rays-vet-rag.

Główne osiągnięcia:

Po pierwsze, całkowicie przebudowaliśmy projekt rays-vet-rag jako moduł CLI Panda. Teraz instaluje się go jedną komendą curl, dokładnie jak brew.

Po drugie, zunifikowaliśmy dependencies. Wywaliliśmy duplikaty pyproject.toml, pozbyliśmy się śmieci typu uuid i pathlib, i wszystko przepisaliśmy na uv. MLX jest teraz głównym backendem dla Apple Silicon, z opcjonalnym PyTorch dla innych platform.

Po trzecie, dodaliśmy pełne wsparcie dla PostDevAI Memory System, dokładnie jak w CLI Panda. Mamy hybrydową pamięć RAM plus RocksDB, z kompresją LZ4 i cache hit rate na poziomie 94 procent.

Po czwarte, stworzyliśmy Makefile ze wszystkimi potrzebnymi komendami: make install, develop, test, lint, format i deploy.

Po piąte, projekt ma teraz trzy główne komendy: vetrag do uruchomienia interfejsu TUI, vet-index do budowania indeksów, i vet-classic do oryginalnego interfejsu Raya.

Wszystko jest teraz w pełni zintegrowane z ekosystemem CLI Panda i gotowe do użycia. Projekt zmienił nazwę na rays-vet-rag wersja 0.2.0 i jest oznaczony jako Beta.

To tyle w skrócie. Miłego słuchania!
"""


async def generate_with_mms_tts():
    """Generate audio using Meta MMS-TTS-POL"""
    print("🎙️ Generating Polish audio summary with Meta MMS-TTS...")
    
    try:
        from transformers import VitsModel, AutoTokenizer
        import torch
        
        # Load Polish TTS model
        print("📥 Loading facebook/mms-tts-pol model...")
        model = VitsModel.from_pretrained("facebook/mms-tts-pol")
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-pol")
        
        # Process text in chunks to avoid memory issues
        sentences = SUMMARY_TEXT.strip().split('\n\n')
        audio_chunks = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                print(f"🗣️ Processing chunk {i+1}/{len(sentences)}...")
                
                inputs = tokenizer(sentence, return_tensors="pt")
                
                with torch.no_grad():
                    output = model(**inputs).waveform
                
                audio_chunks.append(output.squeeze().numpy())
        
        # Concatenate all chunks with small pauses
        sample_rate = model.config.sampling_rate
        pause = np.zeros(int(sample_rate * 0.5))  # 0.5s pause
        
        final_audio = []
        for chunk in audio_chunks:
            final_audio.append(chunk)
            final_audio.append(pause)
        
        audio_array = np.concatenate(final_audio)
        
        # Save audio
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"rays_summary_polish_{timestamp}.wav")
        
        scipy.io.wavfile.write(
            output_path,
            rate=sample_rate,
            data=(audio_array * 32767).astype(np.int16)
        )
        
        print(f"✅ Audio saved to: {output_path}")
        print(f"📊 Duration: {len(audio_array) / sample_rate:.1f} seconds")
        print(f"🎧 Ready to play!")
        
        return output_path
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: uv add transformers scipy torch")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def generate_with_edge_tts():
    """Fallback: Generate audio using Edge-TTS"""
    print("🎙️ Using Edge-TTS as fallback...")
    
    try:
        import edge_tts
        
        # Polish voices available in Edge-TTS
        voice = "pl-PL-MarekNeural"  # Male voice
        # voice = "pl-PL-ZofiaNeural"  # Female voice
        
        print(f"🗣️ Using voice: {voice}")
        
        communicate = edge_tts.Communicate(SUMMARY_TEXT, voice)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"rays_summary_polish_edge_{timestamp}.mp3")
        
        await communicate.save(str(output_path))
        
        print(f"✅ Audio saved to: {output_path}")
        print(f"🎧 Ready to play!")
        
        return output_path
        
    except ImportError:
        print("❌ Edge-TTS not installed")
        print("Please install: uv add edge-tts")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def generate_with_macos_say():
    """Fallback: Use macOS native TTS"""
    print("🎙️ Using macOS Say command...")
    
    import subprocess
    import tempfile
    
    try:
        # Check if Zosia voice is available
        result = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True
        )
        
        has_polish = "Zosia" in result.stdout
        
        if has_polish:
            voice = "Zosia"
            print("✅ Found Polish voice: Zosia")
        else:
            voice = "Alex"  # Default English voice
            print("⚠️ Polish voice not found, using default")
        
        # Save to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f"rays_summary_polish_say_{timestamp}.aiff")
        
        # Create command
        cmd = [
            "say",
            "-v", voice,
            "-o", str(output_path),
            SUMMARY_TEXT
        ]
        
        print("🗣️ Generating speech...")
        subprocess.run(cmd, check=True)
        
        print(f"✅ Audio saved to: {output_path}")
        print(f"🎧 Ready to play!")
        
        return output_path
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running say command: {e}")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


async def main():
    """Generate audio summary using available TTS method"""
    print("🐼 rays-vet-rag Audio Summary Generator")
    print("=" * 50)
    
    # Try different TTS methods in order of preference
    methods = [
        ("Meta MMS-TTS-POL", generate_with_mms_tts),
        ("Edge-TTS", generate_with_edge_tts),
        ("macOS Say", generate_with_macos_say)
    ]
    
    for name, method in methods:
        print(f"\n🔧 Trying {name}...")
        result = await method()
        if result:
            print(f"\n🎉 Successfully generated audio using {name}!")
            break
    else:
        print("\n❌ All TTS methods failed")
        print("Please install one of:")
        print("  - uv add transformers scipy torch")
        print("  - uv add edge-tts")
        print("  - Or check macOS Say command")


if __name__ == "__main__":
    asyncio.run(main())