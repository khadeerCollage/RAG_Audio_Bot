"""
============================================
MODULE 2: RAG (Retrieval Augmented Generation)
============================================
This module FINDS relevant information from your data

Think of it like this:
❓ User asks a question
   ↓
🔍 RAG searches your documents
   ↓
📄 Finds the most relevant pieces
   ↓
📦 Sends them to LLM for answering

SIMPLE ENGLISH:
- RAG = "memory" of the bot
- It remembers your data and finds what's useful
- Without RAG, bot only knows what the LLM was trained on
- With RAG, bot knows YOUR specific information!

WHY RAG?
- LLMs can "hallucinate" (make up facts)
- RAG grounds answers in YOUR real data
- User asks about return policy → RAG finds your return policy
- Now the LLM answers with REAL information!
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Document:
    """
    A simple container for a document.
    
    Each document has:
    - id: Unique identifier
    - title: What the document is about
    - content: The actual text
    - category: Optional grouping
    - embedding: Number representation (filled later)
    """
    id: str
    title: str
    content: str
    category: str = ""
    embedding: Optional[np.ndarray] = None


class RAGModule:
    """
    RAG = Retrieval Augmented Generation
    
    This class does THREE jobs:
    1. Load your documents
    2. Create embeddings (number representations)
    3. Find relevant documents for a question
    
    WHAT ARE EMBEDDINGS?
    ====================
    Imagine every sentence as a point in space:
    - "I love dogs" → point at [0.8, 0.2, 0.5, ...]
    - "I adore puppies" → point at [0.79, 0.21, 0.49, ...]
    - "The car is red" → point at [0.1, 0.9, 0.3, ...]
    
    Similar sentences = close points!
    "love dogs" and "adore puppies" are CLOSE
    "love dogs" and "car is red" are FAR
    
    This is how we find relevant documents!
    """
    
    def __init__(self, documents_path: str = None, use_simple_search: bool = True):
        """
        Set up the RAG system.
        
        Parameters:
        - documents_path: Path to your JSON file with documents
        - use_simple_search: If True, use basic keyword search
                            If False, use embeddings (needs more setup)
        
        For learning, we start with simple search.
        It's easier to understand and doesn't need extra models!
        """
        self.documents: List[Document] = []
        self.use_simple_search = use_simple_search
        self.embedder = None
        
        print("🔍 Initializing RAG Module...")
        
        if documents_path:
            self.load_documents(documents_path)
        
        if not use_simple_search:
            self._setup_embedder()
        
        print("✅ RAG ready!")
    
    def load_documents(self, path: str):
        """
        Load documents from a JSON file.
        
        Expected format:
        [
            {
                "id": "doc_001",
                "title": "Return Policy",
                "content": "You can return items within 30 days...",
                "category": "policy"
            },
            ...
        ]
        """
        print(f"📂 Loading documents from: {path}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.documents = []
        for item in data:
            doc = Document(
                id=item.get("id", ""),
                title=item.get("title", ""),
                content=item.get("content", ""),
                category=item.get("category", "")
            )
            self.documents.append(doc)
        
        print(f"   ✅ Loaded {len(self.documents)} documents")
        
        # If using embeddings, compute them now
        if not self.use_simple_search and self.embedder:
            self._compute_embeddings()
    
    def _setup_embedder(self):
        """
        Set up the embedding model.
        
        We use sentence-transformers, which creates good embeddings
        for finding similar text.
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            print("🧠 Loading embedding model...")
            # all-MiniLM-L6-v2 is small and fast
            # Good enough for most use cases!
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            print("   ✅ Embedding model ready")
            
        except ImportError:
            print("⚠️ sentence-transformers not installed")
            print("   Falling back to simple search")
            self.use_simple_search = True
    
    def _compute_embeddings(self):
        """
        Compute embeddings for all documents.
        
        This turns each document's text into numbers
        that can be compared for similarity.
        """
        if not self.embedder:
            return
        
        print("🔢 Computing embeddings...")
        
        # Combine title and content for better matching
        texts = [f"{doc.title} {doc.content}" for doc in self.documents]
        
        # Get embeddings from the model
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Store embeddings in documents
        for doc, emb in zip(self.documents, embeddings):
            doc.embedding = emb
        
        print("   ✅ Embeddings computed")
    
    def search(self, query: str, top_k: int = 3) -> List[Tuple[Document, float]]:
        """
        Find the most relevant documents for a query.
        
        Parameters:
        - query: The user's question
        - top_k: How many documents to return
        
        Returns:
        - List of (document, score) tuples
        - Higher score = more relevant
        
        Example:
            results = rag.search("How do I return an item?")
            for doc, score in results:
                print(f"{doc.title}: {score:.2f}")
        """
        if self.use_simple_search:
            return self._simple_search(query, top_k)
        else:
            return self._embedding_search(query, top_k)
    
    def _simple_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """
        Simple keyword-based search.
        
        HOW IT WORKS:
        1. Split query into words
        2. Count how many query words appear in each document
        3. Documents with more matches = higher score
        
        This is basic but works well for exact matches!
        """
        # Convert query to lowercase words
        query_words = set(query.lower().split())
        
        scores = []
        
        for doc in self.documents:
            # Combine title and content, lowercase
            doc_text = f"{doc.title} {doc.content}".lower()
            doc_words = set(doc_text.split())
            
            # Count matching words
            matches = len(query_words & doc_words)  # & = intersection
            
            # Normalize by query length
            if len(query_words) > 0:
                score = matches / len(query_words)
            else:
                score = 0
            
            scores.append((doc, score))
        
        # Sort by score (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top_k results
        return scores[:top_k]
    
    def _embedding_search(self, query: str, top_k: int) -> List[Tuple[Document, float]]:
        """
        Embedding-based semantic search.
        
        HOW IT WORKS:
        1. Convert query to embedding (numbers)
        2. Compare to all document embeddings
        3. Use "cosine similarity" to measure closeness
        4. Return closest documents
        
        This finds SEMANTICALLY similar content!
        "return policy" can match "refund process"
        even though the words are different.
        """
        if not self.embedder:
            print("⚠️ No embedder, using simple search")
            return self._simple_search(query, top_k)
        
        # Get query embedding
        query_embedding = self.embedder.encode([query])[0]
        
        scores = []
        
        for doc in self.documents:
            if doc.embedding is None:
                continue
            
            # Cosine similarity: how aligned are the vectors?
            # 1.0 = perfectly similar, 0.0 = unrelated
            similarity = self._cosine_similarity(query_embedding, doc.embedding)
            scores.append((doc, similarity))
        
        # Sort by similarity (highest first)
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        VISUAL EXPLANATION:
        Imagine two arrows from the origin.
        - If they point the same direction: similarity = 1
        - If they're perpendicular: similarity = 0
        - If they point opposite: similarity = -1
        
        Formula: (A · B) / (|A| * |B|)
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get relevant context as a formatted string.
        
        This is what we send to the LLM along with the question.
        The LLM can then answer based on this real information!
        
        Example:
            context = rag.get_context("What's the return policy?")
            # Returns: "Based on your documents:\n1. Return Policy: ..."
        """
        results = self.search(query, top_k)
        
        if not results or results[0][1] == 0:
            return "No relevant information found in the knowledge base."
        
        context_parts = ["Based on the following information:\n"]
        
        for i, (doc, score) in enumerate(results, 1):
            if score > 0:  # Only include if there's some match
                context_parts.append(f"\n[{i}] {doc.title}")
                context_parts.append(f"    {doc.content}")
        
        return "\n".join(context_parts)
    
    def add_document(self, id: str, title: str, content: str, category: str = ""):
        """
        Add a new document to the knowledge base.
        
        Example:
            rag.add_document(
                id="doc_new",
                title="New Feature",
                content="We just launched a new feature..."
            )
        """
        doc = Document(id=id, title=title, content=content, category=category)
        
        # Compute embedding if using embeddings
        if not self.use_simple_search and self.embedder:
            text = f"{title} {content}"
            doc.embedding = self.embedder.encode([text])[0]
        
        self.documents.append(doc)
        print(f"✅ Added document: {title}")


if __name__ == "__main__":
    docs_path = Path(__file__).parent.parent / "rag_data" / "sample_documents.json"
    rag = RAGModule(documents_path=str(docs_path), use_simple_search=True)
    
    query = "What's the return policy?"
    context = rag.get_context(query)
    print(context)
