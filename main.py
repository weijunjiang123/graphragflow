"""Main entry point for the GraphRAG application."""

import os
import logging
import time
import sys
from typing import Optional

from langchain_experimental.llms.ollama_functions import OllamaFunctions

# Import configuration
from src.config import (
    NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD, 
    CHUNK_SIZE, CHUNK_OVERLAP, LLM_MODEL, DOCUMENT_PATH,
    LOG_LEVEL, OUTPUT_DIR
)

# Import core modules
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import EmbeddingsManager
from src.core.entity_extraction import EntityExtractor
from src.core.graph_transformer import GraphTransformerWrapper
from src.core.neo4j_manager import Neo4jManager, Neo4jConnectionManager
from src.core.progress_tracker import ProgressTracker

# Import utilities
from src.utils import save_graph_documents, setup_logging

def main():
    """Main execution function with progress tracking"""
    # Set up logging
    setup_logging(level=getattr(logging, LOG_LEVEL))
    logger = logging.getLogger(__name__)
    
    # Initialize progress tracker
    progress = ProgressTracker(total_stages=7)
    
    try:
        # STAGE 1: Load and process documents
        progress.update("Loading and processing documents")
        document_processor = DocumentProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        documents = document_processor.load_and_split(DOCUMENT_PATH)
        
        # STAGE 2: Initialize LLM and convert to graph documents
        progress.update("Converting documents to graph format")
        llm = OllamaFunctions(model=LLM_MODEL, temperature=0, format="json")
        transformer = GraphTransformerWrapper(llm=llm)
        graph_documents, _ = transformer.create_graph_from_documents(documents)
        
        # STAGE 3: Save graph documents
        progress.update("Saving extracted graph documents")
        saved_file = save_graph_documents(graph_documents, output_dir=OUTPUT_DIR)
        logger.info(f"Saved graph documents to {saved_file}")
        
        # STAGE 4: Initialize Neo4j graph
        progress.update("Initializing Neo4j graph")
        neo4j_manager = Neo4jManager(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
        print("✓ Neo4j graph initialized")
        
        # STAGE 5: Add graph documents to Neo4j
        progress.update("Adding graph documents to Neo4j")
        neo4j_manager.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        print(f"✓ Added {len(graph_documents)} graph documents to Neo4j")
        
        # STAGE 6: Create vector index and fulltext index
        progress.update("Creating vector and fulltext indices")
        try:
            embeddings_manager = EmbeddingsManager()
            embeddings = embeddings_manager.get_working_embeddings()
            
            if embeddings:
                vector_retriever = embeddings_manager.create_vector_index(
                    embeddings=embeddings,
                    neo4j_url=NEO4J_URL, 
                    neo4j_user=NEO4J_USER, 
                    neo4j_password=NEO4J_PASSWORD, 
                    index_name="document_vector",
                    recreate=False
                )
                if vector_retriever:
                    print("✓ Vector index created successfully")
            else:
                logger.warning("No working embeddings model found")
                print("⚠️ Could not initialize embeddings model, skipping vector index creation")
        except Exception as e:
            logger.error(f"Vector index creation failed: {str(e)}")
            print(f"❌ Vector index creation failed: {str(e)}")
            print("Continuing without vector retrieval...")

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