import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException
from langchain_core.documents import Document as LangchainDocument

from src.api.models import QueryRequest, QueryResponse, QueryResultItem
from src.api.main import (
    get_neo4j_manager,
    get_vector_retriever,
    # get_embeddings_manager # Not directly used for querying unless new embeddings are generated
)

from src.core.neo4j_manager import Neo4jManager
# from src.core.embeddings import EmbeddingsManager # For type hinting

logger = logging.getLogger(__name__)
router = APIRouter()

def format_retrieved_documents(docs: List[LangchainDocument], search_type: str) -> List[QueryResultItem]:
    """Formats Langchain documents (typically from vector search) into QueryResultItem."""
    results = []
    for doc in docs:
        # Assuming 'source_id' is in metadata from document processing step
        source_doc_id = doc.metadata.get("source_id", doc.metadata.get("source", "unknown_source"))
        # If the document itself is a chunk, its content is the snippet
        text_snippet = doc.page_content
        score = doc.metadata.get("score", None) # Some retrievers add score to metadata

        results.append(QueryResultItem(
            source_document_id=str(source_doc_id), # Ensure it's a string
            text_snippet=text_snippet,
            score=score,
            node_id=doc.metadata.get("node_id", str(source_doc_id)), # If it's a Document node
            node_type="Document" # Vector search usually returns Document nodes
        ))
    return results

def format_neo4j_records(records: List[Dict[str, Any]], search_type: str) -> List[QueryResultItem]:
    """Formats Neo4j records (from Cypher queries) into QueryResultItem."""
    results = []
    for record in records:
        node = record.get("node")
        score = record.get("score") # Fulltext search provides score

        if node:
            node_id = str(node.get("id", node.element_id)) # Use 'id' property if available, else element_id
            node_type = list(node.labels)[0] if node.labels else "Unknown"
            
            # Try to get a meaningful text snippet
            # Order of preference: 'text' (from Document nodes), 'name', then 'id'
            text_snippet = node.get("text", node.get("name", node_id))
            
            # Attempt to find a source document link
            source_doc_id = node.get("source_document_id") # If directly linked
            if not source_doc_id and "source_id" in node: # If it's a Document node itself
                 source_doc_id = node.get("source_id")

            results.append(QueryResultItem(
                source_document_id=str(source_doc_id) if source_doc_id else node_id,
                text_snippet=str(text_snippet)[:500], # Truncate long snippets
                score=float(score) if score is not None else None,
                graph_context=dict(node.items()), # All node properties
                node_id=node_id,
                node_type=node_type
            ))
        else:
            # Handle records that might not have a 'node' (e.g. pure relationship queries)
            # For now, we skip if no identifiable node
            logger.warning(f"Skipping record in {search_type} due to missing 'node' data: {record}")
            continue
            
    return results

@router.post("/query", response_model=QueryResponse)
async def query_graph(
    request: QueryRequest,
    neo4j_manager: Neo4jManager = Depends(get_neo4j_manager),
    vector_retriever: Optional[Any] = Depends(get_vector_retriever) # Can be None
):
    """
    Performs a query against the graph database using vector, fulltext, or hybrid search.
    """
    logger.info(f"Received query: '{request.query_text}' with search_type: '{request.search_type}'")
    results: List[QueryResultItem] = []

    try:
        if request.search_type == "vector":
            if not vector_retriever:
                raise HTTPException(status_code=503, detail="Vector retriever is not available. Check API initialization.")
            # The retriever itself is callable or has a method like get_relevant_documents / invoke
            retrieved_docs = await vector_retriever.ainvoke(request.query_text, k=request.top_k)
            # retrieved_docs = vector_retriever.get_relevant_documents(request.query_text, k=request.top_k) # for sync
            results = format_retrieved_documents(retrieved_docs, "vector")

        elif request.search_type == "fulltext":
            # Uses the fulltext index "fulltext_entity_id" on __Entity__.id
            # Or if a specific "document_text_ft" index on Document.text exists.
            # Let's assume "fulltext_entity_id" primarily targets entity name/id rather than full Document text.
            # A more common fulltext search would be on document text content.
            # If the vector index is on "Document" nodes, let's try a fulltext on Document text if available.
            # This query assumes a fulltext index named 'document_text_ft' on Document(text)
            # If not, it falls back to searching entity IDs/names via 'fulltext_entity_id'
            
            # Check if a dedicated document fulltext index exists (e.g., 'document_text_ft')
            # This is an example, actual index name might vary based on setup.
            # For now, we'll use the entity fulltext index created in Neo4jManager.
            # It's on n.id for __Entity__ nodes.
            # A better fulltext query for general content:
            # CALL db.index.fulltext.queryNodes("document_text_ft", $query) YIELD node, score
            # WHERE node:Document
            # RETURN node, score LIMIT $limit
            # For this example, we'll use the existing 'fulltext_entity_id' which searches node IDs.
            
            # Attempting a broader search across multiple properties if a general FT index is not specific.
            # This query is a generic keyword search, not strictly a "fulltext index" query unless such an index exists.
            # A proper fulltext query for entities:
            cypher_query = """
            CALL db.index.fulltext.queryNodes("fulltext_entity_id", $query_text) YIELD node, score
            RETURN node, score
            LIMIT $limit
            """
            # If you want to search all text properties (slower, not using specific FT index effectively):
            # cypher_query = """
            # MATCH (node)
            # WHERE (ANY(prop IN keys(node) WHERE node[prop] CONTAINS $query_text))
            #  AND NOT node:Embedding  // Exclude embedding nodes if they exist
            # WITH node, 0.5 AS score // Mock score for general property match
            # RETURN node, score
            # LIMIT $limit
            # """
            params = {"query_text": request.query_text, "limit": request.top_k}
            
            with neo4j_manager.driver.session() as session:
                query_results = session.run(cypher_query, params)
                records = [record.data() for record in query_results]
            results = format_neo4j_records(records, "fulltext")

        elif request.search_type == "graph":
            # Generic graph search: Find nodes that have a property containing the query text.
            # This is a simple keyword search. More sophisticated graph traversals would need specific query patterns.
            cypher_query = """
            MATCH (node)
            WHERE (ANY(key IN keys(node) WHERE apoc.meta.type(node[key]) = 'STRING' AND toLower(toString(node[key])) CONTAINS toLower($query_text)))
              AND NOT labels(node) IN [['Embedding']] // Exclude specific node types if necessary
            WITH node, PROPERTIES(node) as props
            RETURN node {.*, source_document_id: CASE WHEN node:Document THEN node.id ELSE null END, elementId: ID(node)} AS node, 0.7 as score // Mock score
            LIMIT $limit
            """
            # The `elementId` is added to ensure unique ID if `id` property is missing.
            # `source_document_id` is for consistency if the node itself is a Document.
            params = {"query_text": request.query_text, "limit": request.top_k}
            with neo4j_manager.driver.session() as session:
                query_results = session.run(cypher_query, params)
                records = [record.data() for record in query_results]
            results = format_neo4j_records(records, "graph")


        elif request.search_type == "hybrid":
            # Simple Hybrid: Combine Vector and Fulltext/Graph, then de-duplicate
            # 1. Vector Search
            vector_results: List[QueryResultItem] = []
            if vector_retriever:
                retrieved_docs = await vector_retriever.ainvoke(request.query_text, k=request.top_k)
                vector_results = format_retrieved_documents(retrieved_docs, "vector_hybrid")
            else:
                logger.warning("Vector retriever not available for hybrid search.")

            # 2. Graph Search (using the 'graph' logic from above as an example)
            graph_results: List[QueryResultItem] = []
            cypher_query = """
            MATCH (node)
            WHERE (ANY(key IN keys(node) WHERE apoc.meta.type(node[key]) = 'STRING' AND toLower(toString(node[key])) CONTAINS toLower($query_text)))
              AND NOT labels(node) IN [['Embedding']]
            RETURN node {.*, source_document_id: CASE WHEN node:Document THEN node.id ELSE null END, elementId: ID(node)} AS node, 0.6 as score
            LIMIT $limit
            """
            params = {"query_text": request.query_text, "limit": request.top_k}
            with neo4j_manager.driver.session() as session:
                query_results = session.run(cypher_query, params)
                records = [record.data() for record in query_results]
            graph_results = format_neo4j_records(records, "graph_hybrid")
            
            # Combine and de-duplicate (simple approach by node_id or snippet)
            combined_results: Dict[str, QueryResultItem] = {}
            for item in vector_results + graph_results:
                # Use node_id for Document nodes from vector search, or general node_id from graph search
                key = item.node_id if item.node_id else item.text_snippet 
                if key not in combined_results:
                    combined_results[key] = item
                else:
                    # Optional: update score if a new one is better, or merge info
                    if item.score and (not combined_results[key].score or item.score > combined_results[key].score):
                         combined_results[key].score = item.score
            
            results = sorted(list(combined_results.values()), key=lambda x: x.score or 0.0, reverse=True)[:request.top_k]

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported search_type: {request.search_type}. Supported types are 'vector', 'graph', 'fulltext', 'hybrid'.")

        # Ensure all results have some default score if None (e.g. for non-vector searches)
        for item in results:
            if item.score is None:
                item.score = 0.0 # Default score for items without explicit scoring

        return QueryResponse(
            query_text=request.query_text,
            search_type=request.search_type,
            results=results
        )

    except HTTPException as he:
        logger.error(f"HTTP Exception in query endpoint: {he.detail}", exc_info=True)
        raise he
    except Exception as e:
        logger.error(f"Error during query: {e}", exc_info=True)
        # Check for APOC availability if query failed
        if "apoc.meta.type" in str(e).lower() or "unknown function" in str(e).lower() and "apoc" in str(e).lower():
             error_detail = "Query failed due to missing APOC procedures. Please ensure APOC plugin is installed in Neo4j."
             logger.error(error_detail)
             raise HTTPException(status_code=501, detail=error_detail)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query: {str(e)}")
