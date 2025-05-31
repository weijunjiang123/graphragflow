from fastapi import APIRouter, Depends, HTTPException, Path, Query
from typing import Dict, Any, List, Optional
import logging
import time
from pydantic import BaseModel, Field

from src.core.text2cypher import GraphRetriever
from src.services.text2cypher_service import get_text2cypher_service

# Set up logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(tags=["graph"])

# ======== Request and Response Models ========

class EntitySearchRequest(BaseModel):
    """Entity search request model"""
    name: str = Field(..., description="Entity name or pattern to search for")
    fuzzy: bool = Field(True, description="Enable fuzzy matching")
    limit: int = Field(10, description="Maximum number of results", ge=1, le=50)


class EntityRelationRequest(BaseModel):
    """Entity relation path request model"""
    source_id: str = Field(..., description="Source entity ID")
    target_id: str = Field(..., description="Target entity ID")
    max_depth: int = Field(3, description="Maximum path depth", ge=1, le=5)
    relation_types: Optional[List[str]] = Field(None, description="Relation types to consider")


class CypherQueryRequest(BaseModel):
    """Cypher query execution request model"""
    query: str = Field(..., description="Cypher query to execute")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Query parameters")


# ======== Endpoints ========

@router.get("/entities/{entity_id}", response_model=Dict[str, Any])
async def get_entity(
    entity_id: str = Path(..., description="Entity ID"),
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Get details of an entity by ID
    
    Returns complete information about a specific entity.
    """
    try:
        # This is simplified - would need to implement a get_entity_by_id method in the service
        entities = await retrieval_service.find_entities_by_name_async(
            entity_name=entity_id,
            fuzzy_match=False,
            limit=1
        )
        
        if not entities:
            raise HTTPException(status_code=404, detail=f"Entity with ID {entity_id} not found")
            
        return entities[0]
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        logger.error(f"Failed to get entity: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve entity: {str(e)}")


@router.post("/entities/search", response_model=List[Dict[str, Any]])
async def search_entities(
    request: EntitySearchRequest,
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Search for entities by name
    
    Finds entities matching the provided name or pattern.
    """
    try:
        entities = await retrieval_service.find_entities_by_name_async(
            entity_name=request.name,
            fuzzy_match=request.fuzzy,
            limit=request.limit
        )
        return entities
        
    except Exception as e:
        logger.error(f"Entity search failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Entity search failed: {str(e)}")


@router.get("/entities/{entity_id}/neighbors", response_model=Dict[str, Any])
async def get_entity_neighbors(
    entity_id: str = Path(..., description="Entity ID"),
    depth: int = Query(1, ge=1, le=3, description="Depth of neighborhood search"),
    limit: int = Query(10, ge=1, le=50, description="Maximum neighbors to return"),
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Get neighboring entities
    
    Returns entities connected to the specified entity within the given depth.
    """
    try:
        neighbors = await retrieval_service.get_entity_neighbors_async(
            entity_id=entity_id,
            hop=depth,
            limit=limit
        )
        return neighbors
        
    except Exception as e:
        logger.error(f"Failed to get entity neighbors: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve neighbors: {str(e)}")


@router.post("/entities/path", response_model=Dict[str, Any])
async def find_entity_path(
    request: EntityRelationRequest,
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Find path between entities
    
    Discovers the shortest path connecting two entities.
    """
    try:
        path = await retrieval_service.find_shortest_path_async(
            source_entity_id=request.source_id,
            target_entity_id=request.target_id,
            max_depth=request.max_depth,
            relation_types=request.relation_types
        )
        return path
        
    except Exception as e:
        logger.error(f"Path finding failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Path finding failed: {str(e)}")


@router.post("/query/cypher", response_model=Dict[str, Any])
async def execute_cypher_query(
    request: CypherQueryRequest,
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Execute a Cypher query
    
    Runs a custom Cypher query against the knowledge graph.
    """
    try:
        start_time = time.time()
        
        results = await retrieval_service.execute_cypher_query_async(
            cypher_query=request.query,
            params=request.parameters
        )
        
        elapsed_time = time.time() - start_time
        
        return {
            "results": results,
            "count": len(results),
            "query": request.query,
            "elapsed_time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"Cypher query execution failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")


@router.post("/query/natural", response_model=Dict[str, Any])
async def natural_language_to_cypher(
    query: str = Query(..., description="Natural language query to convert to Cypher"),
    retrieval_service: GraphRetriever = Depends(get_text2cypher_service)
):
    """
    Convert natural language to Cypher
    
    Translates a natural language query into a Cypher query and executes it.
    """
    try:
        start_time = time.time()
        
        # Use the text2cypher service instead of the retrieval service for NL to Cypher conversion
        text2cypher_service = get_text2cypher_service()
        
        # Query using text2cypher service
        result = text2cypher_service.query(query)
        
        # Extract data from the result
        cypher_query = result.get("generated_cypher", "")
        results = result.get("raw_results", [])
        
        elapsed_time = time.time() - start_time
        
        return {
            "natural_query": query,
            "cypher_query": cypher_query,
            "results": results,
            "count": len(results),
            "elapsed_time": elapsed_time,
            "explanation": result.get("result", "")  # Include LLM explanation from text2cypher
        }
        
    except Exception as e:
        logger.error(f"Natural language query failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Natural language query failed: {str(e)}")