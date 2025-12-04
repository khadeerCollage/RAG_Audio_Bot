# 🎤 Audio Bot - Voice Assistant with RAG

A complete **voice assistant pipeline** that listens to you, searches your knowledge base, thinks with AI, and talks back!

## 🎯 What This Project Does

```
🎤 You Speak
    ↓
✨ ASR (Whisper) → Converts speech to text
    ↓
🔍 RAG → Searches your documents for relevant info
    ↓
🧠 LLM (GPT/Llama) → Generates smart response
    ↓
🔊 TTS (Edge) → Speaks the response back to you!
```

## 📁 Project Structure

```
audio_bot/
│
├── main.py              # 🚀 Main entry point - runs the full bot
├── test_modules.py      # 🧪 Test individual modules
├── requirements.txt     # 📦 Python dependencies
├── .env.example         # 🔐 Example environment variables
│
├── modules/             # 🧩 The building blocks
│   ├── asr_module.py    # 🎤 Speech to Text (Whisper)
│   ├── rag_module.py    # 🔍 Document Search
│   ├── llm_module.py    # 🧠 AI Brain (GPT/Llama)
│   └── tts_module.py    # 🔊 Text to Speech
│
└── rag_data/            # 📚 Your knowledge base
    └── sample_documents.json  # Sample FAQ data
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get an API Key (Free!)

Get a **free** Groq API key:
1. Go to https://console.groq.com/keys
2. Sign up and create a key
3. Create `.env` file:

```
GROQ_API_KEY=your_key_here
```

### 3. Run the Bot!

```bash
python main.py
```

## 🎓 Learning Path

Each module has **detailed comments** explaining:
- How it works
- Why we do it this way
- Key concepts to understand

### Recommended order:

1. **RAG Module** (`modules/rag_module.py`)
   - Learn how we search documents
   - Understand embeddings
   - No API key needed to test!

2. **ASR Module** (`modules/asr_module.py`)
   - Learn speech recognition
   - Understand audio processing
   - Needs microphone

3. **LLM Module** (`modules/llm_module.py`)
   - Learn about AI language models
   - Understand prompts and context
   - Needs API key

4. **TTS Module** (`modules/tts_module.py`)
   - Learn text-to-speech
   - Understand voice synthesis
   - Needs internet

5. **Main Pipeline** (`main.py`)
   - See how everything connects
   - Understand the flow

## 🧪 Test Individual Modules

```bash
python test_modules.py
```

This lets you test each module separately!

## 📚 Customize Your Knowledge Base

Edit `rag_data/sample_documents.json`:

```json
[
  {
    "id": "doc_001",
    "title": "Your Topic",
    "content": "Your information here...",
    "category": "category_name"
  }
]
```

The bot will answer questions based on YOUR data!

## 🔧 Configuration

Edit `.env` to customize:

```bash
# LLM Provider: "groq" (free) or "openai"
LLM_PROVIDER=groq

# Whisper model: tiny/base/small/medium/large
WHISPER_MODEL=base

# TTS engine: "edge" (natural) or "coqui" (offline)
TTS_PROVIDER=edge
```

## 📖 Key Concepts Explained

### ASR (Automatic Speech Recognition)
- **Whisper**: OpenAI's speech-to-text model
- Works offline after downloading
- Supports 99+ languages

### RAG (Retrieval Augmented Generation)
- **Retrieval**: Find relevant documents
- **Augmented**: Add context to the question
- **Generation**: LLM answers with real info

### LLM (Large Language Model)
- **Groq**: Free, super fast (uses Llama)
- **OpenAI**: Paid, very capable (GPT)

### TTS (Text-to-Speech)
- **Edge-TTS**: Microsoft voices (free, natural)
- **Coqui**: Open source (offline, customizable)

## 🚧 Next Steps (Advanced)

- [ ] Add VAD (Voice Activity Detection)
- [ ] Implement streaming responses
- [ ] Add wake word detection
- [ ] Build WebSocket server for web UI
- [ ] Add barge-in (interrupt mid-response)

## 🤝 Need Help?

Each file has a `# TEACHING SECTION` at the bottom with:
- Detailed explanations
- Visual diagrams
- Tips and best practices

Read those sections to understand deeply!

---

Built for learning. Comments in simple English. 🎓
