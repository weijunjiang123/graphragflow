import os
import json
import logging
import datetime
import time
from typing import Optional, List
import asyncio
from tqdm import tqdm
from pydantic import BaseModel, Field

from neo4j import GraphDatabase
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain_ollama import OllamaEmbeddings
from langchain_experimental.llms.ollama_functions import OllamaFunctions

from utils import get_working_embeddings, save_graph_documents

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Neo4j connection parameters
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "jwj20020124"

# Document processing parameters
CHUNK_SIZE = 256
CHUNK_OVERLAP = 24
LLM_MODEL = "qwen2.5"
DOCUMENT_PATH = "乡土中国.txt"

class ProgressTracker:
    """Track progress across different stages of processing"""
    def __init__(self, total_stages):
        self.total_stages = total_stages
        self.current_stage = 0
        self.start_time = time.time()
        
    def update(self, description):
        """Update progress to the next stage"""
        self.current_stage += 1
        elapsed = time.time() - self.start_time
        logger.info(f"[{self.current_stage}/{self.total_stages}] {description} - Elapsed: {elapsed:.2f}s")
        print(f"Progress: {self.current_stage}/{self.total_stages} - {description}")
        
    def reset(self):
        """Reset the tracker"""
        self.current_stage = 0
        self.start_time = time.time()

class Neo4jConnectionManager:
    """Singleton class to manage Neo4j driver connections"""
    _instance: Optional[GraphDatabase.driver] = None
    
    @classmethod
    def get_instance(cls, uri, auth, **kwargs):
        """Get or create a Neo4j driver instance"""
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
    def close(cls):
        """Close the Neo4j driver connection"""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
            logger.info("Closed Neo4j driver connection")


def create_fulltext_index(driver):
    """Create fulltext index if it doesn't exist"""
    query = '''
    CREATE FULLTEXT INDEX `fulltext_entity_id` 
    IF NOT EXISTS
    FOR (n:__Entity__) 
    ON EACH [n.id];
    '''
    
    try:
        with driver.session() as session:
            session.run(query)
            logger.info("Fulltext index created successfully.")
            return True
    except Exception as e:
        if "already exists" in str(e):
            logger.info("Index already exists, skipping creation.")
            return True
        else:
            logger.error(f"Error creating fulltext index: {str(e)}")
            return False


class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )


def load_and_process_documents(file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Load and split documents into chunks with progress indication"""
    logger.info(f"Loading documents from {file_path}")
    
    # Check if file exists
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Document not found: {file_path}")
    
    # Load document
    loader = TextLoader(file_path=file_path)
    docs = loader.load()
    logger.info(f"Loaded document with {len(docs)} pages")
    
    # Show progress during splitting
    print(f"Splitting document into chunks (size={chunk_size}, overlap={chunk_overlap})...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    documents = text_splitter.split_documents(documents=docs)
    
    logger.info(f"Split documents into {len(documents)} chunks")
    print(f"✓ Document processing complete: {len(documents)} chunks created")
    
    return documents


def batch_process_documents(documents: List, batch_size: int = 5, process_fn=None):
    """Process documents in batches with progress bar"""
    results = []
    
    with tqdm(total=len(documents), desc="Processing documents") as pbar:
        for i in range(0, len(documents), batch_size):
            batch = documents[i:min(i+batch_size, len(documents))]
            
            # Process the batch
            if process_fn:
                batch_results = process_fn(batch)
                results.extend(batch_results)
            else:
                results.extend(batch)
                
            # Update progress
            pbar.update(len(batch))
            
    return results


def create_graph_from_documents(documents, llm_model=LLM_MODEL):
    """Convert documents to graph documents using LLM transformer with progress tracking"""
    total_docs = len(documents)
    logger.info(f"Initializing LLM ({llm_model}) for graph transformation of {total_docs} documents")
    print(f"Initializing LLM model: {llm_model}")
    
    llm = OllamaFunctions(model=llm_model, temperature=0, format="json")
    llm_transformer = LLMGraphTransformer(llm=llm)
    
    print(f"Converting {total_docs} documents to graph format (this may take a while)...")
    
    # Process in batches with progress bar
    start_time = time.time()
    graph_documents = []
    
    batch_size = 5  # Adjust based on your system's capacity
    for i in range(0, len(documents), batch_size):
        batch = documents[i:min(i+batch_size, len(documents))]
        
        # Update progress
        progress = min(i+batch_size, len(documents))
        percentage = (progress / total_docs) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / progress) * (total_docs - progress) if progress > 0 else 0
        
        print(f"\rProcessing: {progress}/{total_docs} documents ({percentage:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s", end="")
        
        # Process batch
        batch_results = llm_transformer.convert_to_graph_documents(batch)
        graph_documents.extend(batch_results)
    
    print(f"\n✓ Conversion complete: {len(graph_documents)} graph documents created in {time.time()-start_time:.1f}s")
    logger.info(f"Created {len(graph_documents)} graph documents")
    
    return graph_documents, llm

def create_vector_index(neo4j_url, neo4j_user, neo4j_password, index_name="vector", recreate=False):
    """Create and return a vector index for document retrieval
    
    Args:
        neo4j_url: Neo4j connection URL
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        index_name: Name of the vector index
        recreate: If True, drop existing index and recreate it
    """
    print("Creating vector index in Neo4j...")
    
    # Get working embeddings
    embeddings = get_working_embeddings()
    
    # Create vector index with progress updates
    logger.info("Creating vector index in Neo4j")
    try:
        # First, check if index exists and needs to be dropped
        if recreate:
            try:
                # Connect to Neo4j and drop the index if it exists
                driver = Neo4jConnectionManager.get_instance(neo4j_url, (neo4j_user, neo4j_password))
                with driver.session() as session:
                    # Check if index exists
                    result = session.run(
                        f"SHOW VECTOR INDEXES WHERE name = $name",
                        name=index_name
                    ).single()
                    
                    if result:
                        print(f"Found existing vector index '{index_name}' - dropping it...")
                        session.run(f"DROP VECTOR INDEX {index_name}")
                        print(f"✓ Dropped existing vector index '{index_name}'")
                        
            except Exception as e:
                logger.warning(f"Error when trying to drop vector index: {str(e)}")
                print(f"Warning: Could not drop existing index: {str(e)}")
            
        # Now create the vector index
        # Use index_name parameter for the index name
        vector_index = Neo4jVector.from_existing_graph(
            embeddings,
            url=neo4j_url,  
            username=neo4j_user,
            password=neo4j_password,
            index_name=index_name,  # Use custom index name
            search_type="hybrid",
            node_label="Document",
            text_node_properties=["text"],
            embedding_node_property="embedding"
        )
        print(f"✓ Vector index '{index_name}' created successfully")
        return vector_index.as_retriever()
    except ValueError as e:
        # Handle dimension mismatch error
        if "dimensions do not match" in str(e):
            logger.error(f"Vector dimension mismatch: {str(e)}")
            print("\n❌ Vector dimension mismatch detected. Trying to recreate index...")
            
            # Recursively call this function but with recreate=True
            if not recreate:
                return create_vector_index(neo4j_url, neo4j_user, neo4j_password, index_name, recreate=True)
            else:
                logger.error("Failed to recreate vector index after dropping")
                print("❌ Failed to recreate vector index even after dropping the old one")
                raise
        else:
            logger.error(f"Error creating vector index: {str(e)}")
            print(f"❌ Failed to create vector index: {str(e)}")
            raise
    except Exception as e:
        logger.error(f"Error creating vector index: {str(e)}")
        print(f"❌ Failed to create vector index: {str(e)}")
        raise


def setup_entity_extraction(llm):
    """Set up the entity extraction chain"""
    print("Setting up entity extraction chain...")
    
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ])
    
    entity_chain = llm.with_structured_output(Entities)
    print("✓ Entity extraction chain ready")
    return entity_chain


def main():
    """Main execution function with progress tracking"""
    # Initialize progress tracker (adjust number of stages based on your workflow)
    progress = ProgressTracker(total_stages=7)
    
    try:
        # STAGE 1: Load and process documents
        progress.update("Loading and processing documents")
        documents = load_and_process_documents(DOCUMENT_PATH)
        
        # STAGE 2: Convert to graph documents
        progress.update("Converting documents to graph format")
        graph_documents, llm = create_graph_from_documents(documents)
        
        # STAGE 3: Save graph documents
        progress.update("Saving extracted graph documents")
        saved_file = save_graph_documents(graph_documents)
        logger.info(f"Saved graph documents to {saved_file}")
        print(f"✓ Saved graph documents to {saved_file}")
        
        # STAGE 4: Initialize Neo4j graph
        progress.update("Initializing Neo4j graph")
        graph = Neo4jGraph(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
        print("✓ Neo4j graph initialized")
        
        # STAGE 5: Add graph documents to Neo4j
        progress.update("Adding graph documents to Neo4j")
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        print(f"✓ Added {len(graph_documents)} graph documents to Neo4j")
        
        # STAGE 6: Create vector index and fulltext index
        progress.update("Creating vector and fulltext indices")
        try:
            vector_retriever = create_vector_index(
                NEO4J_URL, 
                NEO4J_USER, 
                NEO4J_PASSWORD, 
                index_name="document_vector",  # 使用新的索引名称
                recreate=False  # 首次尝试不删除已有索引
            )
            print("✓ Vector index created successfully")
        except Exception as e:
            logger.error(f"Vector index creation failed: {str(e)}")
            print(f"❌ Vector index creation failed: {str(e)}")
            print("Continuing without vector retrieval...")

        # 继续创建全文索引
        driver = Neo4jConnectionManager.get_instance(NEO4J_URL, (NEO4J_USER, NEO4J_PASSWORD))
        index_result = create_fulltext_index(driver)
        if index_result:
            print("✓ Fulltext index created/verified")
        
        # STAGE 7: Set up entity extraction
        progress.update("Setting up entity extraction")
        entity_chain = setup_entity_extraction(llm)
        
        # Complete
        print("\n✅ Setup complete! The system is ready for queries.")
        print(f"Total processing time: {time.time() - progress.start_time:.2f} seconds")
        
        # Example of how to use entity extraction (uncomment to use)
        # example_text = "Google's CEO Sundar Pichai announced a partnership with Microsoft's CEO Satya Nadella."
        # print("\nExample entity extraction:")
        # result = entity_chain.invoke(example_text)
        # print(f"Entities found: {result.names}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)
        print(f"\n❌ Error: {str(e)}")
    finally:
        # Clean up resources
        Neo4jConnectionManager.close()
        logger.info("Resources cleaned up")
        print("\nResources cleaned up")


if __name__ == "__main__":
    main()