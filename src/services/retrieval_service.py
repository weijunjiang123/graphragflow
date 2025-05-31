"""
Retrieval Service - Adapter module for GraphRetriever
Provides backward compatibility for existing code using retrieval_service imports
"""
import logging
from typing import List, Dict, Any, Optional
import asyncio

from src.core.text2cypher import GraphRetriever
from src.config import DATABASE, MODEL
from src.core.model_provider import ModelProvider

logger = logging.getLogger(__name__)

# Alias RetrievalService to GraphRetriever for backward compatibility
RetrievalService = GraphRetriever

# Global service instance
_retrieval_service_instance = None

def get_retrieval_service() -> RetrievalService:
    """
    Get or create a retrieval service instance.
    Uses singleton pattern to avoid re-initialization.
    
    Returns:
        RetrievalService: Configured retrieval service instance
    """
    global _retrieval_service_instance
    
    if _retrieval_service_instance is None:
        logger.info("Initializing retrieval service")
        try:
            model_provider = ModelProvider()
            llm = model_provider.get_llm()
            
            _retrieval_service_instance = RetrievalService(
                connection_uri=DATABASE.URI,
                username=DATABASE.USERNAME,
                password=DATABASE.PASSWORD,
                database_name=DATABASE.DATABASE_NAME,
                llm=llm
            )
            
            # Initialize vector index
            if hasattr(_retrieval_service_instance, 'ensure_fulltext_index'):
                _retrieval_service_instance.ensure_fulltext_index()
            
            if hasattr(_retrieval_service_instance, 'ensure_vector_index'):
                _retrieval_service_instance.ensure_vector_index()
            
            logger.info("Retrieval service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize retrieval service: {str(e)}")
            raise
    
    return _retrieval_service_instance

# Add async support for GraphRetriever
async def execute_cypher_query_async(self, query: str, params: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Execute a Cypher query asynchronously
    
    Args:
        query: Cypher query string
        params: Query parameters (currently ignored as execute_cypher doesn't support params)
        
    Returns:
        Query results
    """
    # The execute_cypher method doesn't accept a params argument
    return await asyncio.to_thread(self.execute_cypher, query)

# Add the async method to GraphRetriever
if not hasattr(GraphRetriever, 'execute_cypher_query_async'):
    GraphRetriever.execute_cypher_query_async = execute_cypher_query_async