# 🎤 Audio Bot - API Documentation

> **Version:** 1.0.0  
> **Last Updated:** December 2024  
> **License:** MIT  

---

## Table of Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation & Setup](#3-installation--setup)
4. [Module Reference](#4-module-reference)
   - [4.1 ASRModule (Speech-to-Text)](#41-asrmodule-speech-to-text)
   - [4.2 RAGModule (Knowledge Retrieval)](#42-ragmodule-knowledge-retrieval)
   - [4.3 LLMModule (Language Model)](#43-llmmodule-language-model)
   - [4.4 TTSModule (Text-to-Speech)](#44-ttsmodule-text-to-speech)
   - [4.5 AudioBot (Main Orchestrator)](#45-audiobot-main-orchestrator)
5. [Data Formats](#5-data-formats)
6. [Configuration](#6-configuration)
7. [Error Handling](#7-error-handling)
8. [Production Deployment](#8-production-deployment)
9. [Performance Optimization](#9-performance-optimization)

---

## 1. Overview

### 1.1 What is Audio Bot?

Audio Bot is a production-ready voice assistant pipeline that processes spoken queries, retrieves relevant information from a knowledge base, generates intelligent responses, and speaks them back to the user.

### 1.2 Core Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│     ASR      │───▶│     RAG      │───▶│     LLM      │───▶│     TTS      │
│  (Whisper)   │    │  (Search)    │    │ (Groq/GPT)   │    │(Edge/Coqui)  │
│              │    │              │    │              │    │              │
│ Audio → Text │    │ Text → Ctx   │    │ Ctx → Reply  │    │ Reply → Audio│
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| **VAD (Voice Activity Detection)** | Auto-detects speech start/end - no fixed recording duration |
| **RAG Integration** | Grounds responses in your custom knowledge base |
| **Multi-Provider LLM** | Supports Groq (free) and OpenAI |
| **Adaptive TTS** | Auto-switches between Edge-TTS (online) and Coqui (offline) |
| **Conversation Memory** | Maintains context across multiple turns |

---

## 2. Architecture

### 2.1 System Architecture Diagram

```
                            ┌─────────────────────────────────────────┐
                            │              AUDIO BOT                   │
                            │            (Orchestrator)                │
                            └─────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    │                         │                         │
                    ▼                         ▼                         ▼
        ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐
        │    ASR Module     │    │    RAG Module     │    │    TTS Module     │
        │                   │    │                   │    │                   │
        │ • Whisper Model   │    │ • Document Store  │    │ • Edge-TTS        │
        │ • VAD Detection   │    │ • Embeddings      │    │ • Coqui TTS       │
        │ • Audio Recording │    │ • Search Engine   │    │ • Audio Playback  │
        └───────────────────┘    └───────────────────┘    └───────────────────┘
                                              │
                                              ▼
                                 ┌───────────────────┐
                                 │    LLM Module     │
                                 │                   │
                                 │ • Groq API        │
                                 │ • OpenAI API      │
                                 │ • Chat History    │
                                 └───────────────────┘
```

### 2.2 Data Flow

```
User Speech
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 1: AUDIO CAPTURE (ASRModule.record_with_vad)                       │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  Microphone stream                                               │
│ Output: numpy.ndarray (float32, 16kHz, mono)                            │
│ Process: VAD monitors energy levels, stops on 1.5s silence              │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 2: TRANSCRIPTION (ASRModule.transcribe_audio)                      │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  numpy.ndarray audio data                                        │
│ Output: str (transcribed text)                                          │
│ Model:  OpenAI Whisper (base/small/medium/large)                        │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 3: CONTEXT RETRIEVAL (RAGModule.get_context)                       │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  str (user query)                                                │
│ Output: str (formatted context from knowledge base)                     │
│ Method: Keyword search or semantic embeddings                           │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 4: RESPONSE GENERATION (LLMModule.chat)                            │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  str (query) + str (context) + List[Message] (history)           │
│ Output: str (AI response)                                               │
│ API:    Groq or OpenAI                                                  │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ STEP 5: SPEECH SYNTHESIS (TTSModule.speak)                              │
│ ─────────────────────────────────────────────────────────────────────── │
│ Input:  str (response text)                                             │
│ Output: Audio playback (speakers)                                       │
│ Engine: Edge-TTS (online) or Coqui (offline)                            │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
User Hears Response
```

---

## 3. Installation & Setup

### 3.1 Requirements

```txt
# requirements.txt
openai-whisper>=20231117
sounddevice>=0.4.6
soundfile>=0.12.1
numpy>=1.24.0
groq>=0.4.0
openai>=1.0.0
edge-tts>=6.1.0
python-dotenv>=1.0.0
TTS>=0.22.0  # Optional: for offline TTS
sentence-transformers>=2.2.0  # Optional: for semantic search
```

### 3.2 Installation

```bash
# Clone or create project
mkdir audio_bot && cd audio_bot

# Install dependencies
pip install -r requirements.txt

# Create environment file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
```

### 3.3 Quick Start

```python
from main import AudioBot

# Initialize bot
bot = AudioBot(
    whisper_model="base",
    llm_provider="groq",
    tts_engine="auto"
)

# Start interactive chat
bot.chat_loop(use_voice=True)
```

---

## 4. Module Reference

---

### 4.1 ASRModule (Speech-to-Text)

**Location:** `modules/asr_module.py`

**Purpose:** Converts audio input to text using OpenAI Whisper with Voice Activity Detection.

#### Class: `ASRModule`

##### Constructor

```python
ASRModule(model_size: str = "base")
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | `str` | `"base"` | Whisper model size |

**Available Model Sizes:**

| Model | Parameters | Speed | Accuracy | VRAM |
|-------|------------|-------|----------|------|
| `tiny` | 39M | Fastest | Low | ~1GB |
| `base` | 74M | Fast | Medium | ~1GB |
| `small` | 244M | Medium | Good | ~2GB |
| `medium` | 769M | Slow | High | ~5GB |
| `large` | 1550M | Slowest | Highest | ~10GB |
| `turbo` | 809M | Fast | High | ~6GB |

##### Methods

---

###### `transcribe_audio()`

Converts audio numpy array to text.

```python
def transcribe_audio(
    self,
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> str
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `audio_data` | `np.ndarray` | required | Audio samples (float32) |
| `sample_rate` | `int` | `16000` | Sample rate in Hz |

**Returns:** `str` - Transcribed text

**Example:**

```python
# Request
asr = ASRModule(model_size="base")

# Simulate audio data (in practice, from microphone)
import numpy as np
audio_data = np.random.randn(16000 * 3).astype(np.float32)  # 3 seconds

# Transcribe
text = asr.transcribe_audio(audio_data, sample_rate=16000)

# Response
print(text)  # "Hello, how are you today?"
```

---

###### `transcribe_file()`

Transcribes an audio file directly.

```python
def transcribe_file(self, audio_path: str) -> str
```

**Parameters:**

| Name | Type | Description |
|------|------|-------------|
| `audio_path` | `str` | Path to audio file (mp3, wav, m4a, flac) |

**Returns:** `str` - Transcribed text

**Example:**

```python
# Request
text = asr.transcribe_file("recording.wav")

# Response
print(text)  # "What is your return policy?"
```

---

###### `record_with_vad()`

Records audio with Voice Activity Detection - automatically stops when speech ends.

```python
def record_with_vad(
    self,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0,
    sample_rate: int = 16000
) -> np.ndarray
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `silence_threshold` | `float` | `0.01` | RMS energy below this = silence |
| `silence_duration` | `float` | `1.5` | Seconds of silence to trigger stop |
| `max_duration` | `float` | `30.0` | Maximum recording length (safety) |
| `sample_rate` | `int` | `16000` | Audio sample rate |

**Returns:** `np.ndarray` - Recorded audio data

**Console Output:**
```
🎙️ Listening... (speak now, I'll stop when you're done)
   🗣️ Speech detected...
   ✅ Speech ended, processing...
   📊 Recorded 3.2 seconds
```

**Example:**

```python
# Request
audio = asr.record_with_vad(
    silence_threshold=0.01,
    silence_duration=1.5
)

# Response
print(f"Recorded {len(audio)/16000:.1f} seconds")
# Recorded 4.2 seconds
```

---

###### `listen_and_transcribe()`

Complete pipeline: Record with VAD → Transcribe → Return text.

```python
def listen_and_transcribe(
    self,
    silence_threshold: float = 0.01,
    silence_duration: float = 1.5,
    max_duration: float = 30.0
) -> str
```

**Returns:** `str` - Transcribed text from speech

**Example:**

```python
# Request
text = asr.listen_and_transcribe()

# Console Output
# 🎙️ Listening... (speak now, I'll stop when you're done)
#    🗣️ Speech detected...
#    ✅ Speech ended, processing...
#    📊 Recorded 2.8 seconds

# Response
print(text)  # "What products do you sell?"
```

---

### 4.2 RAGModule (Knowledge Retrieval)

**Location:** `modules/rag_module.py`

**Purpose:** Retrieves relevant context from a knowledge base to ground LLM responses.

#### Class: `RAGModule`

##### Constructor

```python
RAGModule(
    documents_path: str = None,
    use_simple_search: bool = True
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents_path` | `str` | `None` | Path to JSON document file |
| `use_simple_search` | `bool` | `True` | Use keyword search (vs embeddings) |

##### Methods

---

###### `load_documents()`

Loads documents from a JSON file.

```python
def load_documents(self, path: str) -> None
```

**Expected JSON Format:**

```json
[
    {
        "id": "doc_001",
        "title": "Return Policy",
        "content": "Items can be returned within 30 days of purchase with original receipt.",
        "category": "policy"
    },
    {
        "id": "doc_002",
        "title": "Shipping Information",
        "content": "Free shipping on orders over $50. Standard delivery takes 3-5 business days.",
        "category": "shipping"
    }
]
```

**Example:**

```python
# Request
rag = RAGModule()
rag.load_documents("rag_data/sample_documents.json")

# Console Output
# 📂 Loading documents from: rag_data/sample_documents.json
#    ✅ Loaded 8 documents
```

---

###### `search()`

Finds the most relevant documents for a query.

```python
def search(
    self,
    query: str,
    top_k: int = 3
) -> List[Tuple[Document, float]]
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `query` | `str` | required | User's question |
| `top_k` | `int` | `3` | Number of documents to return |

**Returns:** `List[Tuple[Document, float]]` - List of (document, relevance_score) tuples

**Example:**

```python
# Request
results = rag.search("How do I return an item?", top_k=3)

# Response
for doc, score in results:
    print(f"{doc.title}: {score:.2f}")

# Output:
# Return Policy: 0.67
# Order Cancellation: 0.33
# Shipping Information: 0.17
```

---

###### `get_context()`

Returns formatted context string for LLM consumption.

```python
def get_context(self, query: str, top_k: int = 3) -> str
```

**Returns:** `str` - Formatted context string

**Example:**

```python
# Request
context = rag.get_context("What's the return policy?")

# Response
print(context)
```

**Output:**
```
Based on the following information:

[1] Return Policy
    Items can be returned within 30 days of purchase. Items must be unused 
    and in original packaging. Refunds are processed within 5-7 business days.

[2] Order Cancellation
    Orders can be cancelled within 24 hours of placement for a full refund.
```

---

###### `add_document()`

Adds a new document to the knowledge base at runtime.

```python
def add_document(
    self,
    id: str,
    title: str,
    content: str,
    category: str = ""
) -> None
```

**Example:**

```python
# Request
rag.add_document(
    id="doc_new_001",
    title="Holiday Hours",
    content="We are closed on Christmas Day and New Year's Day.",
    category="hours"
)

# Console Output
# ✅ Added document: Holiday Hours
```

---

### 4.3 LLMModule (Language Model)

**Location:** `modules/llm_module.py`

**Purpose:** Generates intelligent responses using Groq or OpenAI APIs.

#### Class: `LLMModule`

##### Constructor

```python
LLMModule(
    provider: str = "groq",
    model: str = None,
    api_key: str = None,
    system_prompt: str = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"groq"` | `"groq"` or `"openai"` |
| `model` | `str` | `None` | Model name (uses default if None) |
| `api_key` | `str` | `None` | API key (or use env variable) |
| `system_prompt` | `str` | `None` | Custom system instructions |

**Default Models:**

| Provider | Default Model | Notes |
|----------|---------------|-------|
| Groq | `llama-3.3-70b-versatile` | Free tier, very fast |
| OpenAI | `gpt-3.5-turbo` | Paid, high quality |

##### Methods

---

###### `chat()`

Sends a message and gets an AI response.

```python
def chat(
    self,
    user_message: str,
    context: str = None,
    use_history: bool = True
) -> str
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `user_message` | `str` | required | User's question/message |
| `context` | `str` | `None` | RAG context (optional) |
| `use_history` | `bool` | `True` | Include conversation history |

**Returns:** `str` - AI response

**Example 1: Simple Chat**

```python
# Request
llm = LLMModule(provider="groq")
response = llm.chat("What is machine learning?")

# Response
print(response)
# "Machine learning is a type of artificial intelligence where computers 
#  learn patterns from data without being explicitly programmed."
```

**Example 2: Chat with RAG Context**

```python
# Request
context = """Based on the following information:

[1] Return Policy
    Items can be returned within 30 days of purchase."""

response = llm.chat(
    user_message="Can I return something I bought last week?",
    context=context
)

# Response
print(response)
# "Yes, you can return your item. According to our policy, items can be 
#  returned within 30 days of purchase, so you're well within that window."
```

**Example 3: Multi-turn Conversation**

```python
# Turn 1
response1 = llm.chat("My name is John")
print(response1)  # "Nice to meet you, John! How can I help you today?"

# Turn 2 (remembers context)
response2 = llm.chat("What's my name?")
print(response2)  # "Your name is John."
```

---

###### `clear_history()`

Clears conversation memory.

```python
def clear_history(self) -> None
```

**Example:**

```python
# Request
llm.clear_history()

# Console Output
# 🗑️ Conversation history cleared
```

---

###### `set_system_prompt()`

Changes the AI's personality/instructions.

```python
def set_system_prompt(self, prompt: str) -> None
```

**Example:**

```python
# Request
llm.set_system_prompt("""You are a friendly customer service agent for TechStore.
Always be helpful and recommend products when appropriate.
Keep responses under 3 sentences.""")

# Console Output
# 📝 System prompt updated
```

**Default System Prompt:**
```
You are a helpful voice assistant. Your responses will be spoken aloud, so:

1. Keep responses concise (1-3 sentences when possible)
2. Use natural, conversational language
3. Avoid bullet points, numbered lists, or formatting
4. Don't use special characters or emojis
5. If you don't know something, say so honestly

When given context from a knowledge base, use that information to answer accurately.
If the context doesn't contain the answer, say you don't have that information.
```

---

### 4.4 TTSModule (Text-to-Speech)

**Location:** `modules/tts_module.py`

**Purpose:** Converts text responses to spoken audio.

#### Class: `TTSModule`

##### Constructor

```python
TTSModule(
    engine: str = "auto",
    voice: str = None
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | `str` | `"auto"` | `"auto"`, `"edge"`, or `"coqui"` |
| `voice` | `str` | `None` | Voice identifier (engine-specific) |

**Engine Comparison:**

| Engine | Requires Internet | Quality | Speed | Offline |
|--------|------------------|---------|-------|---------|
| Edge-TTS | ✅ Yes | High | Fast | ❌ No |
| Coqui | ❌ No | Medium | Medium | ✅ Yes |
| Auto | Adapts | - | - | ✅ Fallback |

**Available Voices (Edge-TTS):**

| Voice ID | Gender | Accent |
|----------|--------|--------|
| `en-US-JennyNeural` | Female | American |
| `en-US-GuyNeural` | Male | American |
| `en-GB-SoniaNeural` | Female | British |
| `en-AU-NatashaNeural` | Female | Australian |
| `en-IN-NeerjaNeural` | Female | Indian |

##### Methods

---

###### `speak()`

Converts text to speech and plays immediately.

```python
def speak(self, text: str) -> None
```

**Example:**

```python
# Request
tts = TTSModule(engine="edge", voice="en-US-JennyNeural")
tts.speak("Hello! Welcome to our store. How can I help you today?")

# Console Output
# 🎵 Generating speech: "Hello! Welcome to our store..."
# ✅ Audio saved: C:\Users\...\tmp123.mp3
# 🔊 Playback complete
```

---

###### `synthesize()`

Converts text to speech and saves to file.

```python
def synthesize(
    self,
    text: str,
    output_path: str = None
) -> str
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `text` | `str` | required | Text to convert |
| `output_path` | `str` | `None` | Save path (auto-generates if None) |

**Returns:** `str` - Path to saved audio file

**Example:**

```python
# Request
audio_path = tts.synthesize(
    text="Your order has been shipped!",
    output_path="notifications/order_shipped.mp3"
)

# Response
print(audio_path)  # "notifications/order_shipped.mp3"
```

---

###### `print_voices()`

Lists available voices (Edge-TTS only).

```python
def print_voices(self) -> None
```

**Example:**

```python
# Request
tts.print_voices()

# Output
# 🎤 Available English Voices:
# ------------------------------------------------------------
#   en-US-JennyNeural              | Female   | en-US
#   en-US-GuyNeural                | Male     | en-US
#   en-GB-SoniaNeural              | Female   | en-GB
#   en-AU-NatashaNeural            | Female   | en-AU
```

---

### 4.5 AudioBot (Main Orchestrator)

**Location:** `main.py`

**Purpose:** Orchestrates all modules into a complete voice assistant.

#### Class: `AudioBot`

##### Constructor

```python
AudioBot(
    documents_path: str = None,
    whisper_model: str = "base",
    llm_provider: str = "groq",
    tts_engine: str = "edge"
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `documents_path` | `str` | `None` | Path to RAG documents JSON |
| `whisper_model` | `str` | `"base"` | Whisper model size |
| `llm_provider` | `str` | `"groq"` | `"groq"` or `"openai"` |
| `tts_engine` | `str` | `"edge"` | `"auto"`, `"edge"`, or `"coqui"` |

##### Methods

---

###### `process_text()`

Processes text input through RAG → LLM pipeline.

```python
def process_text(self, user_text: str) -> str
```

**Example:**

```python
# Request
bot = AudioBot()
response = bot.process_text("What's your return policy?")

# Console Output
# 💭 Processing: "What's your return policy?"
#    🔍 Searching knowledge base...
#    🧠 Generating response...
#    ✅ Response: "Items can be returned within 30 days..."

# Response
print(response)
# "You can return items within 30 days of purchase. Items must be unused 
#  and in original packaging. Refunds are processed within 5-7 business days."
```

---

###### `process_audio()`

Full pipeline: Audio → Text → Context → Response → Speech.

```python
def process_audio(
    self,
    audio_data: np.ndarray,
    sample_rate: int = 16000
) -> str
```

**Example:**

```python
# Request (with audio data from microphone)
response = bot.process_audio(audio_data, sample_rate=16000)

# Console Output
# 🎤 Processing audio...
#    ✨ Transcribing speech...
#    📝 You said: "Do you offer free shipping?"
#    🔍 Searching knowledge base...
#    🧠 Generating response...
#    🔊 Speaking response...
# 🎵 Generating speech: "Yes, we offer free shipping..."
# 🔊 Playback complete

# Response
print(response)
# "Yes, we offer free shipping on orders over fifty dollars."
```

---

###### `listen_and_respond()`

Complete voice interaction with VAD.

```python
def listen_and_respond(self) -> str
```

**Example:**

```python
# Request
response = bot.listen_and_respond()

# Console Output
# 🎙️ Listening... (speak now, I'll stop when you're done)
#    🗣️ Speech detected...
#    ✅ Speech ended, processing...
#    📊 Recorded 3.2 seconds
# 🎤 Processing audio...
#    ✨ Transcribing speech...
#    📝 You said: "What payment methods do you accept?"
#    🔍 Searching knowledge base...
#    🧠 Generating response...
#    🔊 Speaking response...
# 🔊 Playback complete

# Response
print(response)
# "We accept all major credit cards, PayPal, and Apple Pay."
```

---

###### `chat_loop()`

Starts an interactive conversation loop.

```python
def chat_loop(self, use_voice: bool = True) -> None
```

**Parameters:**

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `use_voice` | `bool` | `True` | Voice input/output mode |

**Commands:**
- `quit` or `exit` - End the conversation
- `clear` - Clear conversation history

**Example (Voice Mode):**

```python
# Request
bot.chat_loop(use_voice=True)

# Console Output
# ============================================================
# 🎯 CHAT MODE STARTED
# ============================================================
# Commands: 'quit' to exit, 'clear' to reset
#
# Voice mode: Press Enter, then speak. I'll detect when you stop!
# ------------------------------------------------------------
#
# [Press Enter to speak...]
# 🎙️ Listening... (speak now, I'll stop when you're done)
#    🗣️ Speech detected...
#    ✅ Speech ended, processing...
# ...
```

**Example (Text Mode):**

```python
# Request
bot.chat_loop(use_voice=False)

# Console Output
# 🎯 CHAT MODE STARTED
# Text mode: Type your message and press Enter
#
# 👤 You: What products do you sell?
# 🤖 Bot: We sell a variety of electronics including laptops, phones, and accessories.
#    [Press Enter to hear it, or 's' to skip]: 
```

---

## 5. Data Formats

### 5.1 RAG Documents JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "array",
  "items": {
    "type": "object",
    "required": ["id", "title", "content"],
    "properties": {
      "id": {
        "type": "string",
        "description": "Unique document identifier"
      },
      "title": {
        "type": "string",
        "description": "Document title (used in search)"
      },
      "content": {
        "type": "string",
        "description": "Full document content"
      },
      "category": {
        "type": "string",
        "description": "Optional category for grouping"
      }
    }
  }
}
```

### 5.2 Sample Documents

```json
[
  {
    "id": "doc_001",
    "title": "Return Policy",
    "content": "Items can be returned within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days after we receive the returned item.",
    "category": "policy"
  },
  {
    "id": "doc_002",
    "title": "Shipping Information",
    "content": "We offer free standard shipping on orders over $50. Standard delivery takes 3-5 business days. Express shipping is available for $9.99 with 1-2 day delivery.",
    "category": "shipping"
  },
  {
    "id": "doc_003",
    "title": "Payment Methods",
    "content": "We accept Visa, MasterCard, American Express, Discover, PayPal, and Apple Pay. All transactions are secured with 256-bit SSL encryption.",
    "category": "payment"
  }
]
```

### 5.3 Audio Format Requirements

| Property | Required Value |
|----------|----------------|
| Sample Rate | 16000 Hz |
| Channels | 1 (Mono) |
| Data Type | float32 |
| Format | WAV (internal), MP3 (TTS output) |

---

## 6. Configuration

### 6.1 Environment Variables

Create a `.env` file in the project root:

```env
# LLM API Keys (at least one required)
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxx

# Optional: Custom settings
WHISPER_MODEL=base
TTS_ENGINE=auto
```

### 6.2 VAD Configuration

Adjust Voice Activity Detection sensitivity:

```python
bot.asr.record_with_vad(
    silence_threshold=0.01,   # Lower = more sensitive (0.005-0.02)
    silence_duration=1.5,     # Seconds of silence before stop (1.0-3.0)
    max_duration=30.0         # Safety limit in seconds
)
```

**Tuning Guide:**

| Environment | `silence_threshold` | `silence_duration` |
|-------------|--------------------|--------------------|
| Quiet room | 0.005 | 1.0 |
| Normal room | 0.01 | 1.5 |
| Noisy environment | 0.02 | 2.0 |

### 6.3 LLM Parameters

```python
# In LLMModule._call_api()
completion = self.client.chat.completions.create(
    model=self.model,
    messages=messages,
    temperature=0.7,    # 0.0-2.0: Lower = focused, Higher = creative
    max_tokens=500      # Maximum response length
)
```

---

## 7. Error Handling

### 7.1 Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `ValueError: Groq API key required!` | Missing API key | Add `GROQ_API_KEY` to `.env` |
| `ImportError: Install edge-tts` | Missing dependency | `pip install edge-tts` |
| `socket.error` during TTS | No internet | Use `tts_engine="coqui"` |
| Empty transcription | No speech detected | Speak louder, check microphone |

### 7.2 Error Response Example

```python
# LLM API Error Handling
try:
    response = llm.chat("Hello")
except Exception as e:
    print(f"Error: {e}")
    # Returns: "I'm sorry, I encountered an error. Please try again."
```

---

## 8. Production Deployment

### 8.1 FastAPI Integration Example

```python
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import numpy as np
import soundfile as sf
import io

app = FastAPI(title="Audio Bot API")

# Initialize bot once at startup
bot = None

@app.on_event("startup")
async def startup():
    global bot
    bot = AudioBot(
        whisper_model="base",
        llm_provider="groq",
        tts_engine="edge"
    )

class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    response: str
    
@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """Process text input and return response."""
    response = bot.process_text(request.text)
    return TextResponse(response=response)

@app.post("/chat/audio")
async def chat_audio(file: UploadFile = File(...)):
    """Process audio file and return response."""
    # Read audio file
    audio_bytes = await file.read()
    audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
    
    # Process (without TTS playback)
    text = bot.asr.transcribe_audio(audio_data.astype(np.float32), sample_rate)
    context = bot.rag.get_context(text)
    response = bot.llm.chat(text, context=context)
    
    return {
        "transcription": text,
        "response": response
    }
```

### 8.2 Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Run API
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 9. Performance Optimization

### 9.1 Latency Breakdown

| Component | Typical Latency | Optimization |
|-----------|----------------|--------------|
| VAD Recording | Variable (user speech) | Reduce `silence_duration` |
| ASR (Whisper) | 1-3s | Use `tiny` or `base` model |
| RAG Search | 10-100ms | Use simple search for small datasets |
| LLM (Groq) | 0.5-2s | Groq is already optimized |
| LLM (OpenAI) | 1-5s | Use `gpt-3.5-turbo` |
| TTS (Edge) | 0.5-1s | Already fast |

### 9.2 Memory Optimization

```python
# Use smaller Whisper model
bot = AudioBot(whisper_model="tiny")  # 39M params vs 74M (base)

# Disable embedding search (saves ~400MB)
rag = RAGModule(use_simple_search=True)
```

### 9.3 Concurrent Processing

```python
import asyncio

async def process_multiple(queries: List[str]):
    """Process multiple queries concurrently."""
    tasks = [
        asyncio.to_thread(bot.process_text, query)
        for query in queries
    ]
    return await asyncio.gather(*tasks)
```

---

## Appendix A: Quick Reference Card

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    AUDIO BOT - QUICK REFERENCE                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INITIALIZATION:                                                        │
│  ───────────────                                                        │
│  from main import AudioBot                                              │
│  bot = AudioBot(whisper_model="base", llm_provider="groq")              │
│                                                                         │
│  MAIN METHODS:                                                          │
│  ─────────────                                                          │
│  bot.process_text("query")      → RAG + LLM (text in, text out)         │
│  bot.listen_and_respond()       → Full voice pipeline with VAD          │
│  bot.chat_loop(use_voice=True)  → Interactive conversation              │
│                                                                         │
│  INDIVIDUAL MODULES:                                                    │
│  ───────────────────                                                    │
│  asr.listen_and_transcribe()    → Voice → Text                          │
│  rag.get_context("query")       → Query → Context                       │
│  llm.chat("msg", context=ctx)   → Generate response                     │
│  tts.speak("text")              → Text → Voice                          │
│                                                                         │
│  ENVIRONMENT:                                                           │
│  ────────────                                                           │
│  GROQ_API_KEY=gsk_xxx           → Free LLM API                          │
│  OPENAI_API_KEY=sk-xxx          → Paid LLM API                          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**Document Version:** 1.0.0  
**Generated:** December 2024  
**Maintained by:** Audio Bot Development Team
