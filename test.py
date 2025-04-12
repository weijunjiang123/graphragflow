"""Test embeddings functionality with Ollama models."""

from pathlib import Path
import sys
import os

PROJECT_ROOT = Path(__file__).resolve().parent

from src.core.graph_retrieval import test_llm_retrieval
from src.config import MODEL
# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.embeddings import EmbeddingsManager

def test_embeddings():
    """Test the embeddings functionality"""
    print("Testing embeddings with nomic-embed-text model...")
    
    embeddings_manager = EmbeddingsManager()
    embed = embeddings_manager.get_working_embeddings(
        provider=MODEL.MODEL_PROVIDER,
        base_url=MODEL.OPENAI_EMBEDDING_API_BASE,  # 修正拼写错误，移除末尾的L
        model_name=MODEL.OPENAI_EMBEDDINGS_MODEL,
        api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
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
    # test_embeddings()
    test_llm_retrieval()