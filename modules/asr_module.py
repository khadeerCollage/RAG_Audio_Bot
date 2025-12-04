"""
============================================
MODULE 1: ASR (Automatic Speech Recognition)
============================================
This module converts AUDIO → TEXT

Think of it like this:
🎤 Microphone captures your voice
   ↓
📊 Audio waves become numbers
   ↓
🧠 Whisper AI understands the numbers
   ↓
📝 You get text!

SIMPLE ENGLISH:
- ASR = "ears" of the bot
- It listens and writes down what you say
- We use OpenAI's Whisper (free, works offline)
"""

import whisper
import numpy as np
import sounddevice as sd
import tempfile
import soundfile as sf
from pathlib import Path


class ASRModule:
    """
    ASR = Automatic Speech Recognition
    
    This class does ONE job:
    Take audio → Give back text
    
    Example:
        asr = ASRModule(model_size="base")
        text = asr.transcribe_audio(audio_data)
        print(text)  # "Hello, how are you?"
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Set up the ASR engine.
        
        model_size options:
        - "tiny"   → Fastest, least accurate (39M parameters)
        - "base"   → Good balance (74M parameters)
        - "small"  → Better accuracy (244M parameters)
        - "medium" → High accuracy (769M parameters)
        - "large"  → Best accuracy (1550M parameters)
        - "turbo"  → Fast + accurate (809M parameters)
        
        For learning, start with "base" - it's fast enough
        to experiment but accurate enough to understand you.
        """
        print(f"🎤 Loading Whisper model: {model_size}")
        print("   (This may take a moment on first run...)")
        
        # Load the Whisper model
        # The model is downloaded automatically if not present
        self.model = whisper.load_model(model_size)
        
        # Store the model size for reference
        self.model_size = model_size
        
        print(f"✅ ASR ready! Using '{model_size}' model")
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Convert audio data to text.
        
        Parameters:
        - audio_data: NumPy array of audio samples
        - sample_rate: How many samples per second (16000 is standard)
        
        Returns:
        - The text that was spoken
        
        How it works (simple):
        1. Audio comes in as numbers (wave heights)
        2. Whisper looks at patterns in the numbers
        3. It matches patterns to words it learned
        4. Returns the words as text
        """
        # Whisper needs audio in a specific format
        # We save to a temp file, then load it
        # (There are faster ways, but this is clearest for learning)
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_path = f.name
            sf.write(temp_path, audio_data, sample_rate)
        
        try:
            # The magic happens here!
            # Whisper analyzes the audio and returns text
            result = self.model.transcribe(
                temp_path,
                fp16=False  # Use fp32 for CPU (more compatible)
            )
            
            # Get just the text from the result
            text = result["text"].strip()
            
            return text
            
        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)
    
    def transcribe_file(self, audio_path: str) -> str:
        """
        Transcribe an audio file directly.
        
        Supported formats: mp3, wav, m4a, flac, etc.
        
        Example:
            text = asr.transcribe_file("recording.wav")
        """
        print(f"🎧 Transcribing: {audio_path}")
        
        result = self.model.transcribe(audio_path, fp16=False)
        text = result["text"].strip()
        
        print(f"📝 Result: {text}")
        return text
    
    def record_and_transcribe(self, duration: float = 5.0) -> str:
        """
        Record from microphone for fixed duration and transcribe.
        """
        sample_rate = 16000
        
        print(f"🎙️ Recording for {duration} seconds...")
        print("   Speak now!")
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        print("✅ Recording complete!")
        audio_data = audio_data.flatten()
        text = self.transcribe_audio(audio_data, sample_rate)
        
        return text
    
    def record_with_vad(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Record audio with Voice Activity Detection (VAD).
        
        Automatically stops when user stops speaking!
        
        Parameters:
        - silence_threshold: Audio level below this = silence (0.01 works well)
        - silence_duration: Seconds of silence before stopping (1.5s default)
        - max_duration: Maximum recording time (safety limit)
        - sample_rate: Audio sample rate
        
        Returns:
        - NumPy array of recorded audio
        """
        import queue
        import threading
        
        chunk_duration = 0.1
        chunk_samples = int(sample_rate * chunk_duration)
        
        audio_queue = queue.Queue()
        recording = []
        is_speaking = False
        silence_chunks = 0
        silence_chunks_needed = int(silence_duration / chunk_duration)
        max_chunks = int(max_duration / chunk_duration)
        
        stop_flag = threading.Event()
        
        def audio_callback(indata, frames, time, status):
            if not stop_flag.is_set():
                audio_queue.put(indata.copy())
        
        print("🎙️ Listening... (speak now, I'll stop when you're done)")
        
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            blocksize=chunk_samples,
            callback=audio_callback
        )
        
        with stream:
            chunk_count = 0
            
            while chunk_count < max_chunks:
                try:
                    chunk = audio_queue.get(timeout=0.5)
                    recording.append(chunk)
                    chunk_count += 1
                    
                    energy = np.sqrt(np.mean(chunk ** 2))
                    
                    if energy > silence_threshold:
                        if not is_speaking:
                            print("   🗣️ Speech detected...")
                            is_speaking = True
                        silence_chunks = 0
                    else:
                        if is_speaking:
                            silence_chunks += 1
                            
                            if silence_chunks >= silence_chunks_needed:
                                print("   ✅ Speech ended, processing...")
                                break
                    
                except queue.Empty:
                    continue
        
        stop_flag.set()
        
        if not recording:
            return np.array([], dtype=np.float32)
        
        audio_data = np.concatenate(recording).flatten()
        
        duration_recorded = len(audio_data) / sample_rate
        print(f"   📊 Recorded {duration_recorded:.1f} seconds")
        
        return audio_data
    
    def listen_and_transcribe(
        self,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.5,
        max_duration: float = 30.0
    ) -> str:
        """
        Listen with VAD and transcribe - stops automatically when you stop talking!
        
        Parameters:
        - silence_threshold: Audio level below this = silence
        - silence_duration: Seconds of silence before stopping
        - max_duration: Maximum recording time
        
        Returns:
        - Transcribed text
        """
        sample_rate = 16000
        
        audio_data = self.record_with_vad(
            silence_threshold=silence_threshold,
            silence_duration=silence_duration,
            max_duration=max_duration,
            sample_rate=sample_rate
        )
        
        if len(audio_data) == 0:
            return ""
        
        text = self.transcribe_audio(audio_data, sample_rate)
        return text


if __name__ == "__main__":
    asr = ASRModule(model_size="base")
    input("Press Enter when ready to record...")
    text = asr.record_and_transcribe(duration=5)
    print(f"📝 YOU SAID: {text}")
