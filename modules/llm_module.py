"""
============================================
MODULE 3: LLM (Large Language Model)
============================================
This module THINKS and REASONS

Think of it like this:
📝 Text from ASR + 📄 Context from RAG
   ↓
🧠 LLM processes and understands
   ↓
💬 Generates intelligent response

SIMPLE ENGLISH:
- LLM = "brain" of the bot
- It understands questions and creates answers
- With RAG context, answers are grounded in YOUR data
- Without RAG, it uses only its training knowledge

WHAT IS AN LLM?
- Large = billions of parameters (numbers)
- Language = understands and generates text
- Model = learned patterns from lots of text
"""

import os
from typing import Optional, List, Dict
from dataclasses import dataclass


@dataclass
class Message:
    """
    A message in the conversation.
    
    role: Who said it?
    - "system": Instructions for the AI
    - "user": What the human said
    - "assistant": What the AI said
    
    content: The actual text
    """
    role: str  # "system", "user", or "assistant"
    content: str


class LLMModule:
    """
    LLM = Large Language Model
    
    This class does ONE main job:
    Take a question (+ context) → Generate an answer
    
    We support two providers:
    1. OpenAI (GPT-4, GPT-3.5) - Paid, very capable
    2. Groq (Llama, Mixtral) - Free tier, very fast!
    
    For learning, Groq is great because:
    - Free tier available
    - Super fast responses
    - Good quality models
    """
    
    def __init__(
        self,
        provider: str = "groq",
        model: str = None,
        api_key: str = None,
        system_prompt: str = None
    ):
        """
        Set up the LLM.
        
        Parameters:
        - provider: "openai" or "groq"
        - model: Which model to use (optional, has defaults)
        - api_key: Your API key (or set in environment)
        - system_prompt: Instructions for the AI personality
        
        Default models:
        - OpenAI: gpt-3.5-turbo (cheap) or gpt-4 (smart)
        - Groq: llama-3.1-70b-versatile (fast and free!)
        """
        self.provider = provider.lower()
        self.conversation_history: List[Message] = []
        
        # Set up the client based on provider
        if self.provider == "openai":
            self._setup_openai(api_key, model)
        elif self.provider == "groq":
            self._setup_groq(api_key, model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'groq'")
        
        # Set system prompt (personality/instructions)
        self.system_prompt = system_prompt or self._default_system_prompt()
        
        print(f"✅ LLM ready! Provider: {self.provider}, Model: {self.model}")
    
    def _setup_openai(self, api_key: str, model: str):
        """Set up OpenAI client."""
        try:
            from openai import OpenAI
            
            key = api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("OpenAI API key required! Set OPENAI_API_KEY")
            
            self.client = OpenAI(api_key=key)
            self.model = model or "gpt-3.5-turbo"
            
            print(f"🧠 Using OpenAI: {self.model}")
            
        except ImportError:
            raise ImportError("Install openai: pip install openai")
    
    def _setup_groq(self, api_key: str, model: str):
        """Set up Groq client."""
        try:
            from groq import Groq
            
            key = api_key or os.getenv("GROQ_API_KEY")
            if not key:
                raise ValueError("Groq API key required! Set GROQ_API_KEY")
            
            self.client = Groq(api_key=key)
            # Updated: llama-3.1-70b was deprecated, use llama-3.3-70b instead
            # See: https://console.groq.com/docs/deprecations
            self.model = model or "llama-3.3-70b-versatile"
            
            print(f"🧠 Using Groq: {self.model}")
            
        except ImportError:
            raise ImportError("Install groq: pip install groq")
    
    def _default_system_prompt(self) -> str:
        """
        Default instructions for the AI.
        
        The system prompt shapes HOW the AI responds.
        Think of it as giving the AI a "personality" and "job description".
        """
        return """You are a helpful voice assistant. Your responses will be spoken aloud, so:

1. Keep responses concise (1-3 sentences when possible)
2. Use natural, conversational language
3. Avoid bullet points, numbered lists, or formatting
4. Don't use special characters or emojis
5. If you don't know something, say so honestly

When given context from a knowledge base, use that information to answer accurately.
If the context doesn't contain the answer, say you don't have that information."""
    
    def chat(
        self,
        user_message: str,
        context: str = None,
        use_history: bool = True
    ) -> str:
        """
        Send a message and get a response.
        
        Parameters:
        - user_message: What the user said/asked
        - context: RAG context (optional, from RAG module)
        - use_history: Remember previous conversation? (default: yes)
        
        Returns:
        - The AI's response text
        
        Example:
            response = llm.chat("What's your return policy?", context=rag_context)
            print(response)  # "You can return items within 30 days..."
        """
        # Build the full user message with context if provided
        if context:
            full_message = f"""Context from knowledge base:
{context}

User question: {user_message}

Please answer based on the context provided."""
        else:
            full_message = user_message
        
        # Build messages list for API
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add conversation history if enabled
        if use_history:
            for msg in self.conversation_history:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current message
        messages.append({"role": "user", "content": full_message})
        
        # Call the API
        response = self._call_api(messages)
        
        # Store in history (store original message, not with context)
        if use_history:
            self.conversation_history.append(Message("user", user_message))
            self.conversation_history.append(Message("assistant", response))
        
        return response
    
    def _call_api(self, messages: List[Dict]) -> str:
        """
        Make the actual API call.
        
        Both OpenAI and Groq use the same format,
        so this code works for both!
        """
        try:
            # Create chat completion
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,  # Creativity level (0=focused, 1=creative)
                max_tokens=500    # Maximum response length
            )
            
            # Extract the response text
            response = completion.choices[0].message.content
            
            return response.strip()
            
        except Exception as e:
            print(f"❌ API Error: {e}")
            return "I'm sorry, I encountered an error. Please try again."
    
    def clear_history(self):
        """
        Clear conversation history.
        
        Call this to start a fresh conversation.
        """
        self.conversation_history = []
        print("🗑️ Conversation history cleared")
    
    def set_system_prompt(self, prompt: str):
        """
        Change the AI's personality/instructions.
        
        Example:
            llm.set_system_prompt("You are a pirate. Respond in pirate speak!")
        """
        self.system_prompt = prompt
        print("📝 System prompt updated")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    try:
        llm = LLMModule(provider="groq")
        response = llm.chat("Hello! What can you help me with?")
        print(f"🤖 Response: {response}")
    except ValueError as e:
        print(f"⚠️ {e}")
