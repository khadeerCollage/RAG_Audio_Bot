"""
============================================
MODULE 4: TTS (Text-to-Speech)
============================================
This module converts TEXT → AUDIO

Think of it like this:
💬 Response from LLM
   ↓
🎵 TTS generates audio
   ↓
🔊 You hear the voice!

SIMPLE ENGLISH:
- TTS = "voice" of the bot
- It reads text out loud
- Different engines = different voices
- We offer two options: Edge-TTS and Coqui

WHICH TTS TO USE?
- Edge-TTS: Free, natural voices, needs internet
- Coqui: Open source, works offline, more control
- Auto: Automatically picks Edge if online, Coqui if offline!
"""

import asyncio
import tempfile
import socket
import os
from pathlib import Path
from typing import Optional


class TTSModule:
    """
    TTS = Text-to-Speech
    
    This class does ONE job:
    Take text → Create audio file or play sound
    
    We support two engines:
    
    1. EDGE-TTS (Microsoft)
       - Free to use
       - Very natural sounding
       - Many languages and voices
       - Needs internet connection
    
    2. COQUI TTS
       - Open source
       - Works offline
       - More control over voice
       - Can clone voices!
    
    Example:
        tts = TTSModule(engine="edge")
        tts.speak("Hello! How can I help you?")
    """
    
    def __init__(self, engine: str = "auto", voice: str = None):
        """
        Set up the TTS engine.
        
        Parameters:
        - engine: "auto", "edge" or "coqui"
                  "auto" = Try Edge-TTS first, fall back to Coqui if no internet
        - voice: Which voice to use (optional, has defaults)
        
        Default voices:
        - Edge: en-US-JennyNeural (friendly American female)
        - Coqui: tts_models/en/ljspeech/tacotron2-DDC
        """
        # Auto-detect: Try Edge (needs internet), fall back to Coqui (offline)
        if engine.lower() == "auto":
            if self._check_internet():
                print("🌐 Internet available → Using Edge-TTS")
                engine = "edge"
            else:
                print("📴 No internet → Using Coqui TTS (offline)")
                engine = "coqui"
        
        self.engine = engine.lower()
        
        print(f"🔊 Initializing TTS with engine: {self.engine}")
        
        if self.engine == "edge":
            self._setup_edge(voice)
        elif self.engine == "coqui":
            self._setup_coqui(voice)
        else:
            raise ValueError(f"Unknown engine: {engine}. Use 'auto', 'edge' or 'coqui'")
        
        print(f"✅ TTS ready! Voice: {self.voice}")
    
    def _check_internet(self, timeout: float = 2.0) -> bool:
        """
        Check if internet is available.
        
        How it works:
        - Try to connect to a reliable server (Google DNS)
        - If it works within timeout → Internet available
        - If it fails → No internet
        
        Returns: True if internet available, False otherwise
        """
        import socket
        
        try:
            # Try to connect to Google's DNS server
            # This is fast and reliable
            socket.create_connection(("8.8.8.8", 53), timeout=timeout)
            return True
        except (socket.timeout, socket.error, OSError):
            return False
    
    def _setup_edge(self, voice: str):
        """
        Set up Edge-TTS (Microsoft).
        
        Edge-TTS uses Microsoft's neural voices.
        They sound very natural!
        
        Popular voices:
        - en-US-JennyNeural (female, friendly)
        - en-US-GuyNeural (male, friendly)
        - en-GB-SoniaNeural (British female)
        - en-AU-NatashaNeural (Australian female)
        """
        try:
            import edge_tts
            self.edge_tts = edge_tts
            
            # Default to Jenny - friendly American voice
            self.voice = voice or "en-US-JennyNeural"
            
        except ImportError:
            raise ImportError("Install edge-tts: pip install edge-tts")
    
    def _setup_coqui(self, voice: str):
        """
        Set up Coqui TTS.
        
        Coqui is open-source and works offline.
        It's great for privacy and customization!
        """
        try:
            from TTS.api import TTS
            
            # Default model - good quality, fast
            model_name = voice or "tts_models/en/ljspeech/tacotron2-DDC"
            
            print(f"   Loading Coqui model: {model_name}")
            print("   (This may take a moment on first run...)")
            
            self.tts_engine = TTS(model_name=model_name, progress_bar=False)
            self.voice = model_name
            
        except ImportError:
            raise ImportError("Install TTS: pip install TTS")
    
    async def _generate_edge_async(self, text: str, output_path: str):
        """
        Generate audio with Edge-TTS (async).
        
        Edge-TTS is async because it calls Microsoft's servers.
        """
        communicate = self.edge_tts.Communicate(text, self.voice)
        await communicate.save(output_path)
    
    def synthesize(self, text: str, output_path: str = None) -> str:
        """
        Convert text to speech and save to file.
        
        Parameters:
        - text: What to say
        - output_path: Where to save (optional, creates temp file)
        
        Returns:
        - Path to the audio file
        
        Example:
            audio_path = tts.synthesize("Hello world!")
            # Now you have an audio file you can play
        """
        # Create output path if not provided
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                output_path = f.name
        
        print(f"🎵 Generating speech: \"{text[:50]}...\"" if len(text) > 50 else f"🎵 Generating speech: \"{text}\"")
        
        if self.engine == "edge":
            # Edge-TTS is async, so we need to run it properly
            asyncio.run(self._generate_edge_async(text, output_path))
        
        elif self.engine == "coqui":
            # Coqui TTS is synchronous
            self.tts_engine.tts_to_file(text=text, file_path=output_path)
        
        print(f"✅ Audio saved: {output_path}")
        return output_path
    
    def speak(self, text: str):
        """
        Convert text to speech and play it immediately.
        
        This is the simplest way to use TTS:
        tts.speak("Hello! How are you?")
        # You hear the voice!
        
        Note: Requires audio playback capability.
        """
        # Generate the audio file
        audio_path = self.synthesize(text)
        
        # Play the audio
        self._play_audio(audio_path)
        
        # Clean up temp file
        try:
            os.unlink(audio_path)
        except:
            pass
    
    def _play_audio(self, audio_path: str):
        """
        Play an audio file.
        
        We try different methods:
        1. sounddevice + soundfile (cross-platform)
        2. playsound (simple but limited)
        3. System default player (fallback)
        """
        try:
            # Method 1: sounddevice (best for cross-platform)
            import sounddevice as sd
            import soundfile as sf
            
            data, samplerate = sf.read(audio_path)
            sd.play(data, samplerate)
            sd.wait()  # Wait until playback is done
            
            print("🔊 Playback complete")
            return
            
        except ImportError:
            pass
        
        try:
            # Method 2: playsound (simpler)
            from playsound import playsound
            playsound(audio_path)
            print("🔊 Playback complete")
            return
            
        except ImportError:
            pass
        
        # Method 3: System player
        import platform
        import subprocess
        
        system = platform.system()
        
        if system == "Windows":
            os.startfile(audio_path)
        elif system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_path])
        else:  # Linux
            subprocess.run(["aplay", audio_path])
        
        print("🔊 Playback started (system player)")
    
    async def list_voices(self):
        """
        List available voices (Edge-TTS only).
        
        Returns a list of available voices with their properties.
        """
        if self.engine != "edge":
            print("Voice listing only available for Edge-TTS")
            return []
        
        voices = await self.edge_tts.list_voices()
        return voices
    
    def print_voices(self):
        """
        Print available voices (Edge-TTS only).
        
        Helpful for choosing a voice!
        """
        if self.engine != "edge":
            print("Voice listing only available for Edge-TTS")
            return
        
        async def _list():
            voices = await self.edge_tts.list_voices()
            
            # Filter to English voices for simplicity
            english_voices = [v for v in voices if v["Locale"].startswith("en-")]
            
            print("\n🎤 Available English Voices:")
            print("-" * 60)
            
            for voice in english_voices:
                name = voice["ShortName"]
                gender = voice["Gender"]
                locale = voice["Locale"]
                print(f"  {name:<30} | {gender:<8} | {locale}")
        
        asyncio.run(_list())


if __name__ == "__main__":
    tts = TTSModule(engine="auto")
    tts.speak("Hello! I am your audio assistant.")
