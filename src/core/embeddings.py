import logging
import os
from typing import List, Optional, Dict, Any

from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Neo4jVector

from src.core.model_provider import ModelProvider

logger = logging.getLogger(__name__)

class EmbeddingsManager:
    """Manager class for embeddings and vector operations"""
    
    @staticmethod
    def get_working_embeddings(provider: str = None, **kwargs) -> Optional[Embeddings]:
        """Get a working embeddings model

        Args:
            provider: Provider for embeddings ("ollama" or "openai")
            **kwargs: Additional parameters for embeddings initialization

        Returns:
            Embeddings model if successful, None otherwise
        """
        # Determine provider from environment if not specified
        if provider is None:
            provider = os.environ.get("MODEL_PROVIDER", "ollama")
            
        # Get embeddings from provider
        return ModelProvider.get_embeddings(provider=provider, **kwargs)
            
    def create_vector_index(self, 
                           embeddings: Embeddings,
                           neo4j_url: str, 
                           neo4j_user: str, 
                           neo4j_password: str, 
                           index_name: str = "vector", 
                           recreate: bool = False,
                           node_label: str = "Document",
                           text_node_properties: List[str] = ["text"]) -> Optional[Any]:
        """Create and return a vector index for document retrieval
        
        Args:
            embeddings: Embeddings model to use
            neo4j_url: Neo4j connection URL
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            index_name: Name of the vector index
            recreate: If True, drop existing index and recreate it
            node_label: Label for document nodes
            text_node_properties: Node properties to index
            
        Returns:
            Vector retriever if successful, None otherwise
        """
        from src.core.neo4j_manager import Neo4jConnectionManager # Move import here

        print(f"Creating vector index '{index_name}' in Neo4j...")
        logger.info(f"Creating vector index '{index_name}' in Neo4j")

        try:
            # Now create the vector index
            vector_index = Neo4jVector.from_existing_graph(
                embeddings,
                url=neo4j_url,
                username=neo4j_user,
                password=neo4j_password,
                index_name=index_name,
                search_type="hybrid",
                node_label=node_label,
                text_node_properties=text_node_properties,
                embedding_node_property="embedding"
            )
            print(f"✓ Vector index '{index_name}' created successfully")
            return vector_index.as_retriever()
        except ValueError as e:
            # Handle dimension mismatch error
            if "dimensions do not match" in str(e):
                logger.error(f"Vector dimension mismatch: {str(e)}")
                print("\n❌ Vector dimension mismatch detected. Trying to recreate index...")

                # Get driver instance and drop index
                driver = Neo4jConnectionManager.get_instance(neo4j_url, (neo4j_user, neo4j_password))
                with driver.session() as session:
                    try:
                        session.run(f"DROP VECTOR INDEX {index_name}")
                        print(f"✓ Dropped existing vector index '{index_name}'")
                    except Exception as inner_e:
                        logger.error(f"Failed to drop index: {str(inner_e)}")

                # Try again with recreate=False since we've manually dropped the index
                return self.create_vector_index(
                    embeddings, neo4j_url, neo4j_user, neo4j_password,
                    index_name, False, node_label, text_node_properties
                )
            else:
                logger.error(f"Error creating vector index: {str(e)}")
                print(f"❌ Failed to create vector index: {str(e)}")
                return None
        except Exception as e:
            logger.error(f"Error creating vector index: {str(e)}")
            print(f"❌ Failed to create vector index: {str(e)}")
            return None
