"""
============================================
🧪 QUICK TEST - Test the Bot Without Setup
============================================
This file lets you test individual modules
without needing all the dependencies.

Great for learning one piece at a time!
"""

import sys
from pathlib import Path


def test_rag():
    """Test the RAG module alone."""
    print("\n" + "=" * 50)
    print("🔍 TESTING RAG MODULE")
    print("=" * 50)
    
    from modules.rag_module import RAGModule
    
    # Use sample documents
    docs_path = Path(__file__).parent / "rag_data" / "sample_documents.json"
    
    rag = RAGModule(
        documents_path=str(docs_path),
        use_simple_search=True  # No need for embeddings model
    )
    
    # Test queries
    queries = [
        "How do I return something?",
        "What payment do you accept?",
        "When can I call support?",
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        context = rag.get_context(query, top_k=2)
        print(f"📄 Context:\n{context}")


def test_tts():
    """Test the TTS module alone."""
    print("\n" + "=" * 50)
    print("🔊 TESTING TTS MODULE")
    print("=" * 50)
    
    from modules.tts_module import TTSModule
    
    tts = TTSModule(engine="edge")
    
    # Test speech
    test_texts = [
        "Hello! This is a test of the text to speech system.",
        "I can help you answer questions about your products.",
    ]
    
    for text in test_texts:
        print(f"\n🎵 Saying: \"{text}\"")
        tts.speak(text)
        print("✅ Done!")


def test_asr():
    """Test the ASR module alone."""
    print("\n" + "=" * 50)
    print("🎤 TESTING ASR MODULE")
    print("=" * 50)
    
    from modules.asr_module import ASRModule
    
    asr = ASRModule(model_size="base")
    
    print("\n🎙️ Recording 5 seconds of speech...")
    input("Press Enter when ready to speak...")
    
    text = asr.record_and_transcribe(duration=5)
    
    print(f"\n📝 You said: \"{text}\"")


def test_llm():
    """Test the LLM module alone."""
    print("\n" + "=" * 50)
    print("🧠 TESTING LLM MODULE")
    print("=" * 50)
    
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    if not os.getenv("GROQ_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("❌ No API key found! Set GROQ_API_KEY or OPENAI_API_KEY")
        return
    
    from modules.llm_module import LLMModule
    
    provider = "groq" if os.getenv("GROQ_API_KEY") else "openai"
    llm = LLMModule(provider=provider)
    
    # Test chat
    queries = [
        "Hello! What can you help me with?",
        "What is 2 + 2?",
    ]
    
    for query in queries:
        print(f"\n❓ Query: {query}")
        response = llm.chat(query)
        print(f"🤖 Response: {response}")


def main():
    """Main test menu."""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🧪 AUDIO BOT - Module Tester                            ║
    ║                                                           ║
    ║   Test individual modules to learn how they work!         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    print("Choose a module to test:")
    print("  1. RAG   - Test document search (no API needed)")
    print("  2. TTS   - Test text-to-speech (needs internet)")
    print("  3. ASR   - Test speech-to-text (needs microphone)")
    print("  4. LLM   - Test language model (needs API key)")
    print("  5. All   - Test everything")
    print("  0. Exit")
    
    choice = input("\nEnter choice (0-5): ").strip()
    
    if choice == "1":
        test_rag()
    elif choice == "2":
        test_tts()
    elif choice == "3":
        test_asr()
    elif choice == "4":
        test_llm()
    elif choice == "5":
        test_rag()
        test_tts()
        test_asr()
        test_llm()
    elif choice == "0":
        print("👋 Bye!")
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
