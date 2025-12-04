"""
============================================
🎤 AUDIO BOT - FULL PIPELINE
============================================
This is the MAIN file that brings everything together!

THE COMPLETE FLOW:
==================
1. 🎤 You speak into microphone
2. 🧠 VAD detects when you stop speaking
3. ✨ ASR converts your speech to text
4. 🔍 RAG searches for relevant information
5. 💬 LLM generates a smart response
6. 🔊 TTS converts response to speech
7. 🎧 You hear the answer!

SIMPLE ENGLISH:
===============
This is like building a robot that can:
- Listen to you (ASR)
- Remember your documents (RAG)
- Think about what you said (LLM)
- Talk back to you (TTS)

LEARNING PATH:
==============
1. First, understand each module separately
2. Then see how they connect here
3. Finally, customize for your needs!
"""

import os
import time
import numpy as np
import sounddevice as sd
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Import our modules
from modules.asr_module import ASRModule
from modules.rag_module import RAGModule
from modules.llm_module import LLMModule
from modules.tts_module import TTSModule


class AudioBot:
    """
    🤖 The Complete Audio Bot
    
    This class orchestrates all modules to create
    a working voice assistant!
    
    PIPELINE VISUALIZATION:
    
    ┌─────────────────────────────────────────────────────┐
    │                    USER SPEAKS                       │
    │                        🎤                            │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              ASR (Whisper)                           │
    │         "What's the return policy?"                  │
    │                   TEXT                               │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              RAG (Retrieval)                         │
    │      Finds: "Return Policy: 30 days..."              │
    │                 CONTEXT                              │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              LLM (GPT/Llama)                         │
    │     "You can return items within 30 days..."         │
    │                 RESPONSE                             │
    └───────────────────────┬─────────────────────────────┘
                            │
                            ▼
    ┌─────────────────────────────────────────────────────┐
    │              TTS (Edge/Coqui)                        │
    │                   🔊                                 │
    │                 AUDIO OUT                            │
    └─────────────────────────────────────────────────────┘
    """
    
    def __init__(
        self,
        documents_path: str = None,
        whisper_model: str = "base",
        llm_provider: str = "groq",
        tts_engine: str = "edge"
    ):
        """
        Initialize the complete audio bot.
        
        Parameters:
        - documents_path: Path to your RAG documents (JSON)
        - whisper_model: ASR model size (tiny/base/small/medium/large)
        - llm_provider: LLM provider (groq/openai)
        - tts_engine: TTS engine (edge/coqui)
        """
        print("=" * 60)
        print("🤖 INITIALIZING AUDIO BOT")
        print("=" * 60)
        
        # Load environment variables
        load_dotenv()
        
        # Track initialization time
        start_time = time.time()
        
        # ---- Module 1: ASR (Speech to Text) ----
        print("\n📦 Module 1: ASR (Automatic Speech Recognition)")
        print("-" * 40)
        self.asr = ASRModule(model_size=whisper_model)
        
        # ---- Module 2: RAG (Information Retrieval) ----
        print("\n📦 Module 2: RAG (Retrieval Augmented Generation)")
        print("-" * 40)
        if documents_path:
            self.rag = RAGModule(documents_path=documents_path)
        else:
            # Use default sample documents
            default_docs = Path(__file__).parent / "rag_data" / "sample_documents.json"
            self.rag = RAGModule(documents_path=str(default_docs))
        
        # ---- Module 3: LLM (Language Model) ----
        print("\n📦 Module 3: LLM (Large Language Model)")
        print("-" * 40)
        self.llm = LLMModule(provider=llm_provider)
        
        # ---- Module 4: TTS (Text to Speech) ----
        print("\n📦 Module 4: TTS (Text to Speech)")
        print("-" * 40)
        # "auto" mode: Uses Edge-TTS if internet available, Coqui if offline
        self.tts = TTSModule(engine=tts_engine)
        
        # Calculate total init time
        init_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print(f"✅ AUDIO BOT READY! (initialized in {init_time:.1f}s)")
        print("=" * 60)
    
    def process_text(self, user_text: str) -> str:
        """
        Process text input through RAG + LLM.
        
        Use this to test without microphone:
            response = bot.process_text("What's your return policy?")
        """
        print(f"\n💭 Processing: \"{user_text}\"")
        
        # Step 1: Get relevant context from RAG
        print("   🔍 Searching knowledge base...")
        context = self.rag.get_context(user_text)
        
        # Step 2: Get LLM response
        print("   🧠 Generating response...")
        response = self.llm.chat(user_text, context=context)
        
        print(f"   ✅ Response: \"{response[:50]}...\"" if len(response) > 50 else f"   ✅ Response: \"{response}\"")
        
        return response
    
    def process_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Process audio through the complete pipeline.
        
        ASR → RAG → LLM → TTS
        
        Returns the spoken response (and plays audio).
        """
        print("\n🎤 Processing audio...")
        
        # Step 1: ASR - Convert speech to text
        print("   ✨ Transcribing speech...")
        user_text = self.asr.transcribe_audio(audio_data, sample_rate)
        print(f"   📝 You said: \"{user_text}\"")
        
        if not user_text.strip():
            response = "I didn't catch that. Could you please repeat?"
            self.tts.speak(response)
            return response
        
        # Step 2 & 3: RAG + LLM
        response = self.process_text(user_text)
        
        # Step 4: TTS - Convert response to speech
        print("   🔊 Speaking response...")
        self.tts.speak(response)
        
        return response
    
    def record_and_respond(self, duration: float = 5.0):
        """
        Record from microphone (fixed duration) and respond.
        """
        sample_rate = 16000
        
        print(f"\n🎙️ Recording for {duration} seconds...")
        print("   Speak now!")
        
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        print("   ✅ Recording complete!")
        
        audio_data = audio_data.flatten()
        response = self.process_audio(audio_data, sample_rate)
        
        return response
    
    def listen_and_respond(self):
        """
        Listen with VAD (Voice Activity Detection) and respond.
        
        Automatically stops when you stop speaking!
        No fixed duration - just speak and pause when done.
        """
        sample_rate = 16000
        
        audio_data = self.asr.record_with_vad(
            silence_threshold=0.01,
            silence_duration=1.5,
            max_duration=30.0,
            sample_rate=sample_rate
        )
        
        if len(audio_data) == 0:
            print("   ⚠️ No speech detected")
            return ""
        
        response = self.process_audio(audio_data, sample_rate)
        return response
    
    def chat_loop(self, use_voice: bool = True):
        """
        Start an interactive chat loop.
        
        Parameters:
        - use_voice: If True, use voice input/output
                    If False, use text input only
        
        Commands:
        - "quit" or "exit": Stop the chat
        - "clear": Clear conversation history
        """
        print("\n" + "=" * 60)
        print("🎯 CHAT MODE STARTED")
        print("=" * 60)
        print("Commands: 'quit' to exit, 'clear' to reset")
        
        if use_voice:
            print("\nVoice mode: Press Enter, then speak. I'll detect when you stop!")
        else:
            print("\nText mode: Type your message and press Enter")
        
        print("-" * 60)
        
        while True:
            try:
                if use_voice:
                    input("\n[Press Enter to speak...]")
                    self.listen_and_respond()
                else:
                    user_input = input("\n👤 You: ").strip()
                    
                    if user_input.lower() in ["quit", "exit", "bye"]:
                        print("👋 Goodbye!")
                        break
                    
                    if user_input.lower() == "clear":
                        self.llm.clear_history()
                        continue
                    
                    if not user_input:
                        continue
                    
                    response = self.process_text(user_input)
                    print(f"🤖 Bot: {response}")
                    
                    # Optionally speak the response
                    speak_it = input("   [Press Enter to hear it, or 's' to skip]: ")
                    if speak_it.lower() != 's':
                        self.tts.speak(response)
                        
            except KeyboardInterrupt:
                print("\n\n👋 Chat ended by user")
                break


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🎤 AUDIO BOT - Voice Assistant with RAG                 ║
    ║                                                           ║
    ║   Pipeline: ASR → RAG → LLM → TTS                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check for API key
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("⚠️  No API key found!")
        print("")
        print("To use this bot, you need an LLM API key:")
        print("")
        print("Option 1: Groq (FREE tier available)")
        print("  1. Go to: https://console.groq.com/keys")
        print("  2. Create an account and get a key")
        print("  3. Create .env file with: GROQ_API_KEY=your_key")
        print("")
        print("Option 2: OpenAI")
        print("  1. Go to: https://platform.openai.com/api-keys")
        print("  2. Get your API key")
        print("  3. Create .env file with: OPENAI_API_KEY=your_key")
        print("")
        exit(1)
    
    # Determine which provider to use
    provider = "groq" if os.getenv("GROQ_API_KEY") else "openai"
    
    # Create the bot
    bot = AudioBot(
        whisper_model="base",    # ASR model
        llm_provider=provider,    # LLM provider
        tts_engine="auto"         # TTS engine: auto = Edge if online, Coqui if offline
    )
    
    # Choose mode
    print("\n🎯 Choose input mode:")
    print("   1. Voice (speak into microphone)")
    print("   2. Text (type your messages)")
    
    choice = input("\nEnter 1 or 2: ").strip()
    
    use_voice = choice == "1"
    
    # Start chatting!
    bot.chat_loop(use_voice=use_voice)
