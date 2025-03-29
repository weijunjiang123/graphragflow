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
from config import DATABASE, DOCUMENT  # Import the new config classes

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
LLM_MODEL = DOCUMENT.LLM_MODEL
DOCUMENT_PATH = DOCUMENT.DOCUMENT_PATH

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
    
    # Check if the model supports function calling properly
    function_calling_models = ["qwen2.5", "mistral", "llama3.1", "gemma2", "openai"]
    simple_models = ["llama3", "llama3.2", "llama2", "phi3"]
    
    # Determine if we need to use a fallback approach
    use_fallback = any(model in llm_model.lower() for model in simple_models)
    if use_fallback:
        print(f"⚠️ Warning: Model {llm_model} may have limited function calling support")
        print("Using fallback approach with simpler prompting...")
        
        # Use regular ChatOllama with structured output guidance
        from langchain_core.output_parsers import StrOutputParser
        from langchain_experimental.graph_transformers import SimpleGraphTransformer
        from langchain_community.chat_models import ChatOllama
        
        # Use a simpler model setup with prompt guidance for structured output
        llm = ChatOllama(model=llm_model, temperature=0)
        llm_transformer = SimpleGraphTransformer(llm=llm)
    else:
        # Use the standard OllamaFunctions approach for function-calling capable models
        try:
            llm = OllamaFunctions(model=llm_model, temperature=0, format="json")
            llm_transformer = LLMGraphTransformer(llm=llm)
        except Exception as e:
            logger.warning(f"Error initializing OllamaFunctions: {str(e)}")
            print(f"⚠️ Falling back to simpler model approach due to: {str(e)}")
            
            # Fallback to regular ChatOllama
            from langchain_experimental.graph_transformers import SimpleGraphTransformer
            from langchain_community.chat_models import ChatOllama
            
            llm = ChatOllama(model=llm_model, temperature=0)
            llm_transformer = SimpleGraphTransformer(llm=llm)
    
    print(f"Converting {total_docs} documents to graph format (this may take a while)...")
    
    # Process in batches with progress bar and error handling
    start_time = time.time()
    graph_documents = []
    failed_docs = 0
    
    batch_size = 5  # Adjust based on your system's capacity
    for i in range(0, len(documents), batch_size):
        batch = documents[i:min(i+batch_size, len(documents))]
        
        # Update progress
        progress = min(i+batch_size, len(documents))
        percentage = (progress / total_docs) * 100
        elapsed = time.time() - start_time
        eta = (elapsed / progress) * (total_docs - progress) if progress > 0 else 0
        
        print(f"\rProcessing: {progress}/{total_docs} documents ({percentage:.1f}%) - Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s", end="")
        
        # Process batch with error handling for individual documents
        try:
            batch_results = llm_transformer.convert_to_graph_documents(batch)
            graph_documents.extend(batch_results)
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            # Try processing one by one to salvage what we can
            for doc in batch:
                try:
                    result = llm_transformer.convert_to_graph_documents([doc])
                    graph_documents.extend(result)
                except Exception as inner_e:
                    logger.error(f"Failed to process document: {str(inner_e)[:100]}...")
                    failed_docs += 1
    
    total_processed = len(graph_documents)
    print(f"\n✓ Conversion complete: {total_processed} graph documents created in {time.time()-start_time:.1f}s")
    if failed_docs > 0:
        print(f"⚠️ {failed_docs} documents could not be processed")
    
    logger.info(f"Created {total_processed} graph documents (failed: {failed_docs})")
    
    return graph_documents, llm

class SimpleGraphTransformer:
    """A simpler graph transformer for models without function calling support"""
    def __init__(self, llm):
        self.llm = llm
        self.parser = StrOutputParser()
        self.entity_prompt = ChatPromptTemplate.from_template(
            """You are extracting information from documents into a graph format.
            
            For the following text, extract:
            1. Entities (people, organizations, concepts, etc.)
            2. Relationships between entities
            3. Properties of entities
            
            Format your output as JSON with:
            {
              "entities": [
                {"entity_id": "unique_id", "entity_type": "type", "entity_name": "name"}
              ],
              "relationships": [
                {"source": "entity_id", "target": "entity_id", "relationship_type": "type", "relationship_name": "name"}
              ],
              "properties": [
                {"entity_id": "entity_id", "property_name": "name", "property_value": "value"}
              ]
            }
            
            TEXT:
            {text}
            
            JSON OUTPUT:
            """
        )
        self.chain = self.entity_prompt | self.llm | self.parser
    
    def convert_to_graph_documents(self, documents):
        """Convert documents to graph format with basic error handling"""
        results = []
        for doc in documents:
            try:
                # Get text from document
                if hasattr(doc, 'page_content'):
                    text = doc.page_content
                else:
                    text = str(doc)
                
                # Get structured output
                response = self.chain.invoke({"text": text})
                
                # Parse response (try to handle different response formats)
                graph_doc = self._parse_response(response, doc)
                if graph_doc:
                    results.append(graph_doc)
            except Exception as e:
                logger.error(f"Error processing document: {str(e)}")
                # Create minimal graph document to avoid complete failure
                results.append({
                    "nodes": [{"id": f"doc_{len(results)}", "type": "Document", "properties": {"text": text[:1000]}}],
                    "edges": [],
                    "source": doc
                })
        return results
    
    def _parse_response(self, response, doc):
        """Parse the response into a graph document format"""
        # Try to extract JSON part from response
        import json
        import re
        
        # Find JSON block in response
        json_match = re.search(r'({[\s\S]*})', response)
        if json_match:
            try:
                extracted_json = json_match.group(1)
                data = json.loads(extracted_json)
                
                # Convert to expected graph document format
                nodes = []
                edges = []
                
                # Add document node
                doc_id = f"doc_{hash(str(doc))}"
                nodes.append({"id": doc_id, "type": "Document", "properties": {"text": str(doc)}})
                
                # Add entities as nodes
                if "entities" in data:
                    for entity in data["entities"]:
                        node_id = entity.get("entity_id", f"entity_{len(nodes)}")
                        nodes.append({
                            "id": node_id,
                            "type": entity.get("entity_type", "Entity"),
                            "properties": {"name": entity.get("entity_name", "Unknown")}
                        })
                        
                        # Add connection to document
                        edges.append({
                            "source": doc_id,
                            "target": node_id,
                            "type": "CONTAINS",
                            "properties": {}
                        })
                
                # Add relationships as edges
                if "relationships" in data:
                    for rel in data["relationships"]:
                        edges.append({
                            "source": rel.get("source", ""),
                            "target": rel.get("target", ""),
                            "type": rel.get("relationship_type", "RELATED_TO"),
                            "properties": {"name": rel.get("relationship_name", "")}
                        })
                
                return {"nodes": nodes, "edges": edges, "source": doc}
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {response[:100]}...")
        
        # Fallback: create minimal document node
        return {
            "nodes": [{"id": f"doc_{hash(str(doc))}", "type": "Document", "properties": {"text": str(doc)}}],
            "edges": [],
            "source": doc
        }

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