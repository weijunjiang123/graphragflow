import os
import json
import logging
import datetime
from pathlib import Path
import time
from typing import Optional, List
import asyncio
from tqdm import tqdm
from pydantic import BaseModel, Field
# 确定项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent

from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from src.utils import save_graph_documents
from src.config import DATABASE, DOCUMENT, MODEL  # Import the new config classes
from src.core.embeddings import EmbeddingsManager


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Neo4j connection parameters - now using config classes
NEO4J_URL = DATABASE.URI
NEO4J_USER = DATABASE.USERNAME
NEO4J_PASSWORD = DATABASE.PASSWORD

# Document processing parameters - now using config classes
CHUNK_SIZE = DOCUMENT.CHUNK_SIZE
CHUNK_OVERLAP = DOCUMENT.CHUNK_OVERLAP
DOCUMENT_PATH = DOCUMENT.DOCUMENT_PATH

OLLAMA_LLM_MODEL = MODEL.OLLAMA_LLM_MODEL

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


def create_graph_from_documents(documents, llm_model=OLLAMA_LLM_MODEL):
    """Convert documents to graph documents using LLM transformer with progress tracking"""
    total_docs = len(documents)
    logger.info(f"Initializing LLM ({llm_model}) for graph transformation of {total_docs} documents")
    print(f"Initializing LLM model: {llm_model}")
    
    # Check if the model supports function calling properly
    function_calling_models = ["qwen2.5", "mistral", "llama3.1", "gemma2", "openai", "deepseek"]
    simple_models = ["llama3", "llama3.2", "llama2", "phi3"]
    
    # Import the custom graph transformer
    from src.core.graph_transformer import GraphTransformerWrapper
    
    # Determine if we need to use a fallback approach
    use_fallback = any(model in llm_model.lower() for model in simple_models)
    
    try:
        if use_fallback:
            print(f"⚠️ Warning: Model {llm_model} may have limited function calling support")
            print("Using fallback approach with simpler prompting...")
            
            # Use regular ChatOllama with structured output guidance
            from langchain_core.output_parsers import StrOutputParser
            from langchain_experimental.graph_transformers import SimpleGraphTransformer
            from langchain_community.chat_models import ChatOllama
            
            # Use a simpler model setup with prompt guidance for structured output
            llm = ChatOllama(model=llm_model, temperature=0)
            transformer = SimpleGraphTransformer(llm=llm)
            
            # Process the documents using the custom transformer wrapper
            print(f"Converting {total_docs} documents to graph format (this may take a while)...")
            graph_documents = []
            
            start_time = time.time()
            for doc in tqdm(documents, desc="Processing documents"):
                try:
                    result = transformer.convert_to_graph_documents([doc])
                    graph_documents.extend(result)
                except Exception as e:
                    logger.error(f"Error processing document: {str(e)}")
                    # Create a minimal graph document to avoid complete failure
                    text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    graph_documents.append({
                        "nodes": [{"id": f"doc_{len(graph_documents)}", "type": "Document", "properties": {"text": text[:1000]}}],
                        "edges": [],
                        "source": doc
                    })
            
            print(f"✓ Conversion complete: {len(graph_documents)} graph documents created in {time.time()-start_time:.1f}s")
        else:
            # Use the custom GraphTransformerWrapper for better JSON handling
            try:
                # Use OllamaFunctions for function-calling capabilities when available
                from langchain_ollama import OllamaFunctions
                llm = OllamaFunctions(model=llm_model, temperature=0, format="json")
            except Exception as e:
                logger.warning(f"Error initializing OllamaFunctions: {str(e)}")
                print(f"⚠️ Falling back to regular ChatOllama: {str(e)}")
                from langchain_community.chat_models import ChatOllama
                llm = ChatOllama(model=llm_model, temperature=0)
            
            # Initialize the custom graph transformer wrapper
            transformer_wrapper = GraphTransformerWrapper(llm=llm)
            
            # Convert documents to graph format with better JSON handling
            graph_documents, llm = transformer_wrapper.create_graph_from_documents(
                documents, 
                batch_size=5
            )
    except Exception as e:
        logger.error(f"Error during graph transformation: {str(e)}")
        print(f"❌ Error during graph transformation: {str(e)}")
        # Fallback to simpler approach
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model=llm_model, temperature=0)
        
        # Create minimal graph documents
        graph_documents = []
        for doc in documents:
            text = doc.page_content if hasattr(doc, "page_content") else str(doc)
            graph_documents.append({
                "nodes": [{"id": f"doc_{len(graph_documents)}", "type": "Document", "properties": {"text": text[:1000]}}],
                "edges": [],
                "source": doc
            })
        print(f"⚠️ Created {len(graph_documents)} minimal graph documents due to error")
    
    failed_docs = total_docs - len(graph_documents)
    if failed_docs > 0:
        print(f"⚠️ {failed_docs} documents could not be processed")
    
    logger.info(f"Created {len(graph_documents)} graph documents (failed: {failed_docs})")
    
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

    # Initialize EmbeddingsManager
    embeddings_manager = EmbeddingsManager()
    embeddings = embeddings_manager.get_working_embeddings()

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

        if embeddings:
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
        else:
            logger.error("Failed to initialize embeddings")
            print("❌ Failed to initialize embeddings")
            return None
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
            return create_vector_index(neo4j_url, neo4j_user, neo4j_password, index_name, recreate=True)
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
        # 此处显式传递CHUNK_SIZE和CHUNK_OVERLAP参数，确保使用配置文件的值
        documents = load_and_process_documents(DOCUMENT_PATH, CHUNK_SIZE, CHUNK_OVERLAP)
        
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
