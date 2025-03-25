"""Test embeddings functionality with Ollama models."""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import EmbeddingsManager

def test_embeddings():
    """Test the embeddings functionality"""
    print("Testing embeddings with nomic-embed-text model...")
    
    embeddings_manager = EmbeddingsManager()
    embed = embeddings_manager.get_working_embeddings(
        base_url='localhost:11434', 
        model='nomic-embed-text'
    )
    
    if embed:
        input_text = "The meaning of life is 42"
        vector = embed.embed_query(input_text)
        print(f"Generated embedding with dimension: {len(vector)}")
        print(f"Sample values: {vector[:3]}")
        print("Embedding test successful!")
        return True
    else:
        print("Failed to initialize embeddings model")
        return False

if __name__ == "__main__":
    test_embeddings()