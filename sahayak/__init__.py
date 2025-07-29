# sahayak/__init__.py

"""
Sahayak - Educational Voice Assistant
Initialization module to prevent duplicate setup messages.
"""

import os
import sys
from typing import Optional

# Global flags to prevent duplicate initialization
_initialized = False
_initialization_messages = []

def initialize_once():
    """Initialize components only once to prevent duplicate messages."""
    global _initialized, _initialization_messages
    
    if _initialized:
        return _initialization_messages
    
    messages = []
    
    # Suppress warnings during initialization
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Initialize Gemini embeddings
        try:
            from .gemini_embeddings import gemini_embeddings
            if gemini_embeddings is not None and gemini_embeddings.is_available():
                messages.append("✅ Using Gemini embeddings for RAG")
            else:
                messages.append("⚠️ Gemini embeddings not available")
        except Exception as e:
            messages.append(f"⚠️ Gemini embeddings error: {e}")
        
        # Initialize Firestore
        try:
            from .firestore_db import FirestoreDatabase
            firestore_db = FirestoreDatabase()
            messages.append("✅ Firestore database initialized")
        except Exception as e:
            messages.append(f"⚠️ Firestore error: {e}")
        
        # Initialize Vertex AI Search
        try:
            from .vertex_search import vertex_search_tool
            if vertex_search_tool is not None:
                messages.append("✅ Vertex AI Search initialized")
            else:
                messages.append("⚠️ Vertex AI Search not available")
        except Exception as e:
            messages.append(f"⚠️ Vertex AI Search error: {e}")
        
        # Initialize image generation
        try:
            from langchain_google_vertexai import VertexAIImageGeneratorChat
            image_tool = VertexAIImageGeneratorChat(model="imagegeneration@006")
            messages.append("✅ Vertex AI Image Generation initialized")
        except Exception as e:
            messages.append(f"⚠️ Image generation error: {e}")
    
    _initialized = True
    _initialization_messages = messages
    return messages

def print_initialization_status():
    """Print initialization status only once."""
    if not _initialized:
        messages = initialize_once()
        for message in messages:
            print(message)
        print("✅ API graph built successfully.")

# Initialize when module is imported
if __name__ != "__main__":
    # Only initialize if this is the main process (not reloader)
    if not os.environ.get("UVICORN_RELOAD_PROCESS"):
        print_initialization_status()
