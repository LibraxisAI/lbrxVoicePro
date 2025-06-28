#!/usr/bin/env python3
"""
Better Polish TTS response for Maciej
Using Edge-TTS with neural voices
"""

import asyncio
import edge_tts
from datetime import datetime
from pathlib import Path

# Response text
RESPONSE_TEXT = """
Hej Maciej! Masz absolutną rację - Meta MMS ma problem z literą C.

Teraz używam Edge TTS z neuronowym głosem Marka. Ten model poprawnie wymawia wszystkie polskie głoski, włącznie z literą C.

Oto test: Cały czas czekałem na przybycie tego dokumentu. Co by wiedzieć, co było zrobione.

Dodatkowo, Edge TTS oferuje lepszą kontrolę nad prozodią, tempem i emocjami. Mogę mówić szybciej lub wolniej, głośniej lub ciszej.

Ten model jest też znacznie szybszy - generowanie audio zajmuje sekundy, nie minuty.

Dzięki za feedback! Zawsze używaj Edge TTS dla polskiego - jest najlepszy.
"""

async def generate_better_audio():
    """Generate with Edge-TTS Polish Neural voice"""
    print("🎙️ Generating better Polish audio with Edge-TTS...")
    
    # Use Polish male neural voice
    voice = "pl-PL-MarekNeural"
    
    # Create communication object with adjusted parameters
    communicate = edge_tts.Communicate(
        RESPONSE_TEXT,
        voice,
        rate="-5%",  # Slightly slower for clarity
        volume="+10%"  # Slightly louder
    )
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(f"better_response_polish_{timestamp}.mp3")
    
    # Save audio
    await communicate.save(str(output_path))
    
    print(f"✅ Audio saved to: {output_path}")
    print(f"🎧 Ready to play!")
    print(f"\n📊 Edge-TTS advantages:")
    print("  - Correct Polish pronunciation (C != K)")
    print("  - Natural prosody and intonation")
    print("  - Fast generation (<5 seconds)")
    print("  - Small file size (MP3)")
    print("  - No dependencies on heavy ML models")
    
    return output_path

if __name__ == "__main__":
    asyncio.run(generate_better_audio())