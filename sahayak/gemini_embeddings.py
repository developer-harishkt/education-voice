# sahayak/gemini_embeddings.py

import os
import time
from typing import List, Union
from langchain_google_vertexai import VertexAIEmbeddings

class GeminiEmbeddings:
    """Gemini embeddings wrapper for the RAG system."""
    
    def __init__(self, model_name: str = "text-embedding-005", max_retries: int = 3):
        """
        Initialize Gemini embeddings.
        
        Args:
            model_name: Gemini embedding model name
            max_retries: Maximum number of retry attempts
        """
        self.model_name = model_name
        self.max_retries = max_retries
        self.embeddings = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embeddings model."""
        try:
            self.embeddings = VertexAIEmbeddings(model_name=self.model_name)
        except Exception as e:
            print(f"⚠️ Failed to initialize Gemini embeddings: {e}")
            self.embeddings = None
    
    def _retry_operation(self, operation, operation_name: str):
        """Retry an operation with exponential backoff."""
        for attempt in range(self.max_retries):
            try:
                return operation()
            except Exception as e:
                print(f"⚠️ {operation_name} attempt {attempt + 1}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = min(2 ** attempt, 10)  # Exponential backoff, max 10 seconds
                    print(f"⏳ Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ {operation_name} failed after {self.max_retries} attempts")
                    raise
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.
        
        Args:
            texts: List of text documents to embed
            
        Returns:
            List of embedding vectors
        """
        if self.embeddings is None:
            raise ValueError("Gemini embeddings not initialized")
        
        def _embed_operation():
            return self.embeddings.embed_documents(texts)
        
        return self._retry_operation(_embed_operation, "Gemini document embedding")
    
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding vector
        """
        if self.embeddings is None:
            raise ValueError("Gemini embeddings not initialized")
        
        def _embed_operation():
            return self.embeddings.embed_query(text)
        
        return self._retry_operation(_embed_operation, "Gemini query embedding")
    
    def is_available(self) -> bool:
        """Check if Gemini embeddings are available."""
        if self.embeddings is None:
            return False
        
        # Test with a simple query to ensure it's actually working
        try:
            test_result = self.embed_query("test")
            return len(test_result) > 0
        except Exception as e:
            print(f"⚠️ Gemini embeddings availability check failed: {e}")
            return False

# Create a global instance with better error handling
try:
    gemini_embeddings = GeminiEmbeddings()
    if not gemini_embeddings.is_available():
        print("⚠️ Gemini embeddings not available, will use fallback")
        gemini_embeddings = None
except Exception as e:
    print(f"⚠️ Failed to create Gemini embeddings instance: {e}")
    gemini_embeddings = None 