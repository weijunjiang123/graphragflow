import logging
from typing import Optional, Dict, Any, Tuple

from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph

logger = logging.getLogger(__name__)

class Neo4jConnectionManager:
    """Singleton class to manage Neo4j driver connections"""
    _instance: Optional[GraphDatabase.driver] = None
    
    @classmethod
    def get_instance(cls, uri: str, auth: Tuple[str, str], **kwargs) -> GraphDatabase.driver:
        """Get or create a Neo4j driver instance
        
        Args:
            uri: Neo4j connection URI
            auth: Tuple of (username, password)
            **kwargs: Additional driver configuration
            
        Returns:
            Neo4j driver instance
        """
        if cls._instance is None:
            # Set reasonable defaults if not provided
            config = {
                'max_connection_lifetime': 3600,
                'max_connection_pool_size': 50,
                'connection_acquisition_timeout': 60
            }
            config.update(kwargs)
            
            cls._instance = GraphDatabase.driver(
                uri=uri, 
                auth=auth,
                **config
            )
            logger.info("Created new Neo4j driver connection")
        return cls._instance
    
    @classmethod
    def close(cls) -> None:
        """Close the Neo4j driver connection"""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
            logger.info("Closed Neo4j driver connection")


class Neo4jManager:
    """Manager class for Neo4j operations"""
    
    def __init__(self, url: str, username: str, password: str):
        """Initialize Neo4j manager
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
        """
        self.url = url
        self.username = username
        self.password = password
        self.driver = Neo4jConnectionManager.get_instance(url, (username, password))
        self.graph = Neo4jGraph(url, username, password)
        
    def create_fulltext_index(self, index_name: str = "fulltext_entity_id") -> bool:
        """Create fulltext index if it doesn't exist
        
        Args:
            index_name: Name of the fulltext index
            
        Returns:
            True if index was created or already exists, False otherwise
        """
        query = f'''
        CREATE FULLTEXT INDEX `{index_name}` 
        IF NOT EXISTS
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info(f"Fulltext index '{index_name}' created successfully.")
                return True
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Index '{index_name}' already exists, skipping creation.")
                return True
            else:
                logger.error(f"Error creating fulltext index: {str(e)}")
                return False
                
    def add_graph_documents(self, graph_documents, **kwargs):
        """Add graph documents to Neo4j
        
        Args:
            graph_documents: List of graph documents to add
            **kwargs: Additional parameters for add_graph_documents
        """
        return self.graph.add_graph_documents(
            graph_documents,
            **kwargs
        )
        
    def drop_index(self, index_name: str) -> bool:
        """Drop an index if it exists
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            True if index was dropped, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Check if index exists
                result = session.run(
                    f"SHOW VECTOR INDEXES WHERE name = $name",
                    name=index_name
                ).single()
                
                if result:
                    logger.info(f"Found existing vector index '{index_name}' - dropping it...")
                    session.run(f"DROP VECTOR INDEX {index_name}")
                    logger.info(f"Dropped existing vector index '{index_name}'")
                    return True
                else:
                    logger.info(f"No index named '{index_name}' found to drop")
                    return False
        except Exception as e:
            logger.warning(f"Error when trying to drop vector index: {str(e)}")
            return False
