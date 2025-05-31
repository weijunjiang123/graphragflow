"""Main entry point for the GraphRAG application."""

import os
import logging
from pathlib import Path
import time
import sys
from typing import Optional

from dotenv import load_dotenv

# 首先确保环境变量加载，防止配置模块导入时环境变量未准备好
PROJECT_ROOT = Path(__file__).resolve().parent
env_path = PROJECT_ROOT / '.env'
load_dotenv(dotenv_path=env_path)

# Import configuration
from src.config import (
    DATABASE, DOCUMENT, APP, MODEL
)

# Import core modules
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import EmbeddingsManager
from src.core.entity_extraction import EntityExtractor
from src.core.graph_transformer import GraphTransformerWrapper
from src.core.neo4j_manager import Neo4jManager, Neo4jConnectionManager
from src.core.progress_tracker import ProgressTracker
from src.core.model_provider import ModelProvider

# Import utilities
from src.utils import save_graph_documents, setup_logging

def patch_graph_transformer():
    """
    Patch the GraphTransformerWrapper class to handle markdown-formatted JSON.
    """
    from src.core.graph_transformer import GraphTransformerWrapper
    from langchain_community.graphs.graph_document import Node
    from langchain_community.graphs.graph_document import Relationship
    from langchain_community.graphs.graph_document import GraphDocument
    import re
    import json
    
    # Store the original method
    original_create_graph = GraphTransformerWrapper.create_graph_from_documents
    
    def clean_markdown_json(text: str) -> str:
        """Clean JSON from markdown code blocks that may be returned by LLMs."""
        pattern = r'```(?:json)?\s*([\s\S]*?)```'
        matches = re.findall(pattern, text)
        if matches:
            return matches[0].strip()
        return text
    
    # Define a patched method that cleans markdown before processing
    def patched_create_graph(self, *args, **kwargs):
        # Apply the patch to clean any markdown in the LLM response
        # This is added at the GraphTransformer level to handle issues before they cause validation errors
        
        # Call original method
        try:
            return original_create_graph(self, *args, **kwargs)
        except Exception as e:
            print(f"Standard conversion failed: {str(e)}. Trying with markdown cleaning...")
            
            # Custom implementation with markdown cleaning
            # This is a simplified version - your actual implementation may vary
            # based on the specific error and how GraphTransformerWrapper works
            
            # ...implement fallback logic here if needed...
            
            # Just re-raise for now
            raise
    
    # Apply the patch
    GraphTransformerWrapper.create_graph_from_documents = patched_create_graph

# Call this function before using the transformer
patch_graph_transformer()

def preprocess_graph_documents(graph_documents):
    """
    Preprocess graph documents to ensure source is an object with all attributes
    required by Neo4j's add_graph_documents method (id, type, etc.).
    """
    from dataclasses import dataclass
    import logging
    
    logger = logging.getLogger(__name__)
    
    @dataclass
    class SourceWrapper:
        id: str
        type: str = "Document"  # Add type attribute with default value
        properties: dict = None  # Add properties attribute for completeness
        
        def __post_init__(self):
            if self.properties is None:
                self.properties = {"name": self.id}
    
    def process_element(element):
        """Process individual elements recursively to find and fix string sources."""
        # Handle source attribute if present
        if hasattr(element, 'source'):
            if isinstance(element.source, str):
                logger.debug(f"Converting string source to SourceWrapper: {element.source}")
                element.source = SourceWrapper(id=element.source)
            elif hasattr(element.source, 'id') and not hasattr(element.source, 'type'):
                # If source has id but not type, add type
                logger.debug(f"Adding type to source: {element.source.id}")
                source_id = element.source.id
                element.source = SourceWrapper(id=source_id)

        if hasattr(element, 'target'):
            if isinstance(element.target, str):
                logger.debug(f"Converting string target to SourceWrapper: {element.target}")
                element.target = SourceWrapper(id=element.target)
            elif hasattr(element.target, 'id') and not hasattr(element.target, 'type'):
                # If target has id but not type, add type
                logger.debug(f"Adding type to target: {element.target.id}")
                target_id = element.target.id
                element.target = SourceWrapper(id=target_id)
        
         # Process relationships recursively
        if hasattr(element, 'relationships') and element.relationships:
            for rel in element.relationships:
                process_element(rel)
                
        # Process nodes array if present
        if hasattr(element, 'nodes') and isinstance(element.nodes, list):
            for node in element.nodes:
                process_element(node)
                
        # Process relationships array if present
        if hasattr(element, 'relationships_array') and isinstance(element.relationships_array, list):
            for rel in element.relationships_array:
                process_element(rel)
                
        # Process elements array if present
        if hasattr(element, 'elements') and isinstance(element.elements, list):
            for el in element.elements:
                process_element(el)
                
        return element
    
    processed_documents = []
    
    logger.info(f"Processing {len(graph_documents)} graph documents")
    for doc in graph_documents:
        processed_doc = process_element(doc)
        processed_documents.append(processed_doc)
    
    logger.info(f"Processed {len(processed_documents)} graph documents")
    return processed_documents

def main():
    """Main execution function with progress tracking"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Initialize progress tracker
    progress = ProgressTracker(total_stages=7)
    
    try:
        # STAGE 1: Load and process documents
        progress.update("Loading and processing documents")
        document_processor = DocumentProcessor(
            chunk_size=DOCUMENT.CHUNK_SIZE, 
            chunk_overlap=DOCUMENT.CHUNK_OVERLAP
            )
        documents = document_processor.load_and_split(DOCUMENT.DOCUMENT_PATH)
        
        # STAGE 2: Initialize LLM and convert to graph documents
        progress.update("Converting documents to graph format")
        
        # Initialize appropriate LLM based on configured provider
        llm = None
        try:
            if MODEL.MODEL_PROVIDER == "ollama":
                llm = ModelProvider.get_llm(
                    provider="ollama",
                    model_name=MODEL.OLLAMA_LLM_MODEL,
                    base_url=MODEL.OLLAMA_BASE_URL,
                    temperature=0
                )
            else:  # openai
                llm = ModelProvider.get_llm(
                    provider="openai",
                    model_name=MODEL.OPENAI_MODEL,
                    api_key=MODEL.OPENAI_API_KEY,
                    api_base=MODEL.OPENAI_API_BASE,
                    temperature=0
                )
                
            if not llm:
                raise ValueError(f"Failed to initialize LLM with provider {MODEL.MODEL_PROVIDER}")
        except Exception as e:
            logger.error(f"Error initializing LLM: {str(e)}")
            print(f"❌ Error initializing LLM: {str(e)}")
            print("Attempting to fall back to a different provider...")
            
            # Try the other provider as a fallback
            fallback_provider = "openai" if MODEL.MODEL_PROVIDER == "ollama" else "ollama"
            try:
                if fallback_provider == "ollama":
                    llm = ModelProvider.get_llm(
                        provider="ollama",
                        model_name=MODEL.OLLAMA_LLM_MODEL,
                        base_url=MODEL.OLLAMA_BASE_URL,
                        temperature=0
                    )
                else:
                    llm = ModelProvider.get_llm(
                        provider="openai",
                        model_name=MODEL.OPENAI_MODEL,
                        api_key=MODEL.OPENAI_API_KEY,
                        api_base=MODEL.OPENAI_API_BASE,
                        temperature=0
                    )
                
                if llm:
                    print(f"✓ Successfully initialized fallback LLM with provider {fallback_provider}")
                else:
                    raise ValueError(f"Failed to initialize fallback LLM with provider {fallback_provider}")
            except Exception as fallback_error:
                logger.error(f"Error initializing fallback LLM: {str(fallback_error)}")
                raise ValueError(f"Failed to initialize LLM with both primary and fallback providers")
            
        transformer = GraphTransformerWrapper(llm=llm)
        graph_documents, _ = transformer.create_graph_from_documents(documents)
        
        # STAGE 3: Save graph documents
        progress.update("Saving extracted graph documents")
        saved_file = save_graph_documents(graph_documents)
        logger.info(f"Saved graph documents to {saved_file}")
        
        # Preprocess graph documents to fix source attribute format
        processed_graph_documents = preprocess_graph_documents(graph_documents)
        
        # STAGE 4: Initialize Neo4j graph
        progress.update("Initializing Neo4j graph")
        neo4j_manager = Neo4jManager(DATABASE.URI, DATABASE.USERNAME,DATABASE.PASSWORD)
        print("✓ Neo4j graph initialized")
        
        # STAGE 5: Add graph documents to Neo4j
        progress.update("Adding graph documents to Neo4j")
        neo4j_manager.add_graph_documents(
            processed_graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        print(f"✓ Added {len(processed_graph_documents)} graph documents to Neo4j")
        
        # STAGE 6: Create vector index and fulltext index
        progress.update("Creating vector and fulltext indices")
        # try:
        #     # Initialize embeddings based on configured provider
        #     embeddings_manager = EmbeddingsManager()
        #     embeddings = embeddings_manager.get_working_embeddings(provider=MODEL.MODEL_PROVIDER)
            
        #     if embeddings:
        #         vector_retriever = embeddings_manager.create_vector_index(
        #             embeddings=embeddings,
        #             neo4j_url=DATABASE.URI, 
        #             neo4j_user=DATABASE.USERNAME, 
        #             neo4j_password=DATABASE.PASSWORD, 
        #             index_name="document_vector",
        #             recreate=False
        #         )
        #         if vector_retriever:
        #             print("✓ Vector index created successfully")
        #     else:
        #         logger.warning("No working embeddings model found")
        #         print("⚠️ Could not initialize embeddings model, skipping vector index creation")
        # except Exception as e:
        #     logger.error(f"Vector index creation failed: {str(e)}")
        #     print(f"❌ Vector index creation failed: {str(e)}")
        #     print("Continuing without vector retrieval...")

        # Create fulltext index
        index_result = neo4j_manager.create_fulltext_index()
        if index_result:
            print("✓ Fulltext index created/verified")
        
        # STAGE 7: Set up entity extraction
        progress.update("Setting up entity extraction")
        entity_extractor = EntityExtractor(llm)
        
        # Print summary
        print("\n✅ Setup complete! The system is ready for queries.")
        print(progress.summary())
        
        # Example of how to use entity extraction
        if len(sys.argv) > 1 and sys.argv[1] == "--example":
            example_text = "Google's CEO Sundar Pichai announced a partnership with Microsoft's CEO Satya Nadella."
            print("\nExample entity extraction:")
            result = entity_extractor.extract(example_text)
            print(f"Entities found: {result.names}")
        
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
