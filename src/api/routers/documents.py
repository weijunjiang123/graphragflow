import logging
from typing import List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException

from langchain_core.documents import Document as LangchainDocument

from src.api.models import DocumentProcessRequest, DocumentProcessResponse
from src.api.main import (
    get_document_processor,
    get_graph_transformer,
    get_neo4j_manager,
    # get_embeddings_manager # Not directly used for adding, but for setup
)

# Core components
from src.core.document_processor import DocumentProcessor
from src.core.graph_transformer import GraphTransformerWrapper
from src.core.neo4j_manager import Neo4jManager
# from src.core.embeddings import EmbeddingsManager # For type hinting if needed

logger = logging.getLogger(__name__)
router = APIRouter()

# Copied from the original main.py, as it's a utility for document pre-processing
# before adding to Neo4j. This could be moved to a utils file later.
def preprocess_graph_documents_for_api(graph_documents: List[Any]) -> List[Any]:
    """
    Preprocess graph documents to ensure source/target nodes are objects
    with 'id' and 'type' attributes.
    """
    from dataclasses import dataclass, field
    
    @dataclass
    class SourceWrapper:
        id: str
        type: str = "Document"  # Default type
        properties: Dict[str, Any] = field(default_factory=dict)
        
        def __post_init__(self):
            if not self.properties:
                self.properties = {"name": self.id} # Ensure name property if none provided

    def process_element(element: Any):
        if hasattr(element, 'source'):
            if isinstance(element.source, str):
                logger.debug(f"API: Converting string source to SourceWrapper: {element.source}")
                element.source = SourceWrapper(id=element.source)
            elif hasattr(element.source, 'id') and not hasattr(element.source, 'type'):
                logger.debug(f"API: Adding type to source: {element.source.id}")
                element.source = SourceWrapper(id=element.source.id, 
                                               properties=getattr(element.source, 'properties', {}))


        if hasattr(element, 'target'):
            if isinstance(element.target, str):
                logger.debug(f"API: Converting string target to SourceWrapper: {element.target}")
                element.target = SourceWrapper(id=element.target)
            elif hasattr(element.target, 'id') and not hasattr(element.target, 'type'):
                logger.debug(f"API: Adding type to target: {element.target.id}")
                element.target = SourceWrapper(id=element.target.id,
                                               properties=getattr(element.target, 'properties', {}))
        
        if hasattr(element, 'relationships') and element.relationships:
            for rel in element.relationships:
                process_element(rel)
        if hasattr(element, 'nodes') and isinstance(element.nodes, list):
            for node in element.nodes:
                process_element(node)
        return element

    processed_docs = []
    for doc in graph_documents:
        processed_docs.append(process_element(doc))
    return processed_docs


@router.post("/documents/process", response_model=DocumentProcessResponse)
async def process_documents_endpoint(
    request: DocumentProcessRequest,
    doc_processor: DocumentProcessor = Depends(get_document_processor),
    graph_transformer: GraphTransformerWrapper = Depends(get_graph_transformer),
    neo4j_manager: Neo4jManager = Depends(get_neo4j_manager)
    # embeddings_manager: EmbeddingsManager = Depends(get_embeddings_manager) # For vector index re-creation if needed
):
    """
    Processes a list of raw texts, converts them into graph structures,
    and stores them in the Neo4j database.
    Embeddings are assumed to be generated and indexed by Neo4j based on the vector index setup.
    """
    processed_doc_ids = []
    all_graph_documents_to_add = []

    if not request.texts:
        raise HTTPException(status_code=400, detail="No texts provided for processing.")

    logger.info(f"Received request to process {len(request.texts)} texts.")

    try:
        for i, text_content in enumerate(request.texts):
            doc_id = f"api_doc_{i}" # Simple ID generation
            
            # 1. Convert raw text to Langchain Document object
            # Each text string is treated as a separate document.
            # Metadata can be added here if needed.
            langchain_doc = LangchainDocument(page_content=text_content, metadata={"source_id": doc_id, "api_source": True})
            
            # 2. Split document into chunks (if necessary)
            # For this API, let's assume each text is small enough or represents a pre-chunked document.
            # If chunking is desired, doc_processor.text_splitter.split_documents([langchain_doc]) can be used.
            # For now, we pass the document as a single item list.
            chunks = [langchain_doc] 
            # If actual chunking is needed:
            # chunks = doc_processor.text_splitter.split_documents([langchain_doc])
            # logger.info(f"Document {doc_id} split into {len(chunks)} chunks.")

            # 3. Convert document chunks to graph format
            # The transformer expects a list of Langchain Documents.
            if not chunks:
                logger.warning(f"No chunks generated for text {i}, skipping.")
                continue
            
            # create_graph_from_documents returns a tuple: (graph_documents, llm_model_instance)
            graph_documents_list, _ = graph_transformer.create_graph_from_documents(chunks)
            logger.info(f"Converted {len(chunks)} chunks from text {i} into {len(graph_documents_list)} graph documents.")

            if not graph_documents_list:
                logger.warning(f"No graph documents generated for text {i}, skipping.")
                continue
            
            # Assign source document ID to each graph document for tracking
            for gd in graph_documents_list:
                # The GraphDocument structure from graph_transformer.py needs to handle this.
                # It has a `source` attribute. We ensure it's set or updated.
                # The GraphTransformerWrapper already sets the source to the input Document.
                # We just need to make sure our `langchain_doc` has the right ID in its metadata.
                pass # Source is handled by the transformer if input doc has metadata

            all_graph_documents_to_add.extend(graph_documents_list)
            processed_doc_ids.append(doc_id)

        if not all_graph_documents_to_add:
            raise HTTPException(status_code=400, detail="No graph documents could be generated from the provided texts.")

        # 4. Preprocess graph documents (ensure source/target are objects)
        # This step is crucial for compatibility with Neo4jGraph.add_graph_documents
        preprocessed_graph_docs = preprocess_graph_documents_for_api(all_graph_documents_to_add)
        logger.info(f"Preprocessed {len(preprocessed_graph_docs)} graph documents for Neo4j import.")
        
        # 5. Add graph documents to Neo4j
        # `baseEntityLabel=True` ensures nodes get a common label (e.g., "__Entity__")
        # `include_source=True` links graph elements to the source Document node.
        # The vector index on "Document" nodes with "embedding" property should pick up new docs.
        neo4j_manager.add_graph_documents(
            preprocessed_graph_docs,
            include_source=True, # Creates :Document nodes and links entities to them
            baseEntityLabel=True # Adds a common label like __Entity__ to all nodes
        )
        logger.info(f"Successfully added {len(preprocessed_graph_docs)} graph documents to Neo4j for {len(processed_doc_ids)} source texts.")

        # 6. Embeddings and Indexing:
        # As per previous reasoning, this is handled by Neo4j automatically if a vector index
        # is configured for the "Document" label and "embedding" property.
        # The `Neo4jVector.from_existing_graph` in `api/main.py` sets up this listener.
        # If manual re-indexing or adding specific embeddings were needed,
        # methods on `embeddings_manager` or `vector_retriever` would be called here.
        # For example, if `vector_retriever` had an `add_documents` method.

        return DocumentProcessResponse(
            message=f"Successfully processed and stored {len(processed_doc_ids)} texts as graph documents.",
            document_ids=processed_doc_ids
        )

    except HTTPException as he:
        logger.error(f"HTTP Exception in document processing: {he.detail}", exc_info=True)
        raise he
    except Exception as e:
        logger.error(f"Error processing documents: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Example of how to include this router in main.py (already done in previous step):
# from .routers import documents
# app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
