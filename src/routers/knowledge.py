"""
Knowledge Graph module router - Provides RESTful endpoints for knowledge graph management
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form, Path, Query
from fastapi.responses import JSONResponse
from typing import Dict, Any, List, Optional
import logging
import time
import os
import uuid
import json
from datetime import datetime
from pathlib import Path as FilePath
from pydantic import BaseModel, Field

# Import main processing functions
from src.main import load_and_process_documents, create_graph_from_documents
from src.utils import save_graph_documents
from src.config import DOCUMENT, DATABASE, APP, MODEL
from src.services.retrieval_service import RetrievalService, get_retrieval_service

# Configure logging
logger = logging.getLogger(__name__)

# Create router with appropriate tags and prefix
router = APIRouter(
    tags=["knowledge"],
    responses={404: {"description": "Not found"}}
)

# Task storage for async operations
_graph_tasks = {}

# Define project root
PROJECT_ROOT = FilePath(__file__).resolve().parent.parent.parent
UPLOAD_DIR = PROJECT_ROOT / "data"

# Ensure upload directory exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================== Models ==================

class GraphProcessingSettings(BaseModel):
    """Settings for knowledge graph processing"""
    chunk_size: int = Field(DOCUMENT.CHUNK_SIZE, description="Document chunk size")
    chunk_overlap: int = Field(DOCUMENT.CHUNK_OVERLAP, description="Document chunk overlap")
    model: str = Field(MODEL.OLLAMA_LLM_MODEL, description="LLM model to use for processing")
    create_vector_index: bool = Field(True, description="Whether to create vector index")
    vector_index_name: str = Field(APP.VECTOR_INDEX_NAME, description="Name of the vector index")


class GraphTaskResponse(BaseModel):
    """Asynchronous graph processing task response model"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    filename: str = Field(..., description="Uploaded filename")


class DocumentStats(BaseModel):
    """Document statistics model"""
    document_count: int = Field(..., description="Number of documents in the knowledge graph")
    entity_count: int = Field(..., description="Number of entities in the knowledge graph")
    relationship_count: int = Field(..., description="Number of relationships in the knowledge graph")
    last_updated: str = Field(..., description="Last update timestamp")


# ================== Endpoints ==================

@router.post("/documents", response_model=GraphTaskResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Optional[str] = Form(None)
):
    """
    Upload a document to build the knowledge graph
    
    Takes a document file and processes it to extract entities and relationships.
    The processing occurs asynchronously in the background.
    """
    try:
        # Generate unique filename to prevent conflicts
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = file.filename
        safe_filename = f"{timestamp}_{original_filename.replace(' ', '_')}"
        file_path = UPLOAD_DIR / safe_filename
        
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
            
        logger.info(f"File '{original_filename}' uploaded to {file_path}")
        
        # Parse settings if provided
        processing_settings = GraphProcessingSettings()
        if settings:
            try:
                settings_dict = json.loads(settings)
                processing_settings = GraphProcessingSettings(**settings_dict)
            except Exception as e:
                logger.warning(f"Could not parse settings: {str(e)}, using defaults")
        
        # Create task
        task_id = str(uuid.uuid4())
        _graph_tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "file_path": str(file_path),
            "original_filename": original_filename,
            "settings": processing_settings.dict(),
            "result": None
        }
        
        # Start processing in background
        background_tasks.add_task(
            _process_document_task,
            task_id=task_id,
            file_path=file_path,
            settings=processing_settings
        )
        
        return GraphTaskResponse(
            task_id=task_id,
            status="processing",
            filename=original_filename
        )
        
    except Exception as e:
        logger.error(f"Document upload failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Document upload failed: {str(e)}")


@router.get("/documents/tasks/{task_id}", response_model=Dict[str, Any])
async def get_document_processing_status(task_id: str):
    """
    Get the status of a document processing task
    
    Returns the current status or result of a previously submitted document processing task.
    """
    if task_id not in _graph_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    task_info = _graph_tasks[task_id]
    
    if task_info["status"] == "completed":
        return {
            "task_id": task_id,
            "status": "completed",
            "filename": task_info["original_filename"],
            "result": task_info["result"],
            "processing_time": task_info["completed_at"] - task_info["created_at"]
        }
    elif task_info["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "filename": task_info["original_filename"],
            "error": task_info.get("error", "Unknown error")
        }
    else:
        elapsed = time.time() - task_info["created_at"]
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "filename": task_info["original_filename"],
            "elapsed_time": elapsed
        }


@router.get("/documents/stats", response_model=DocumentStats)
async def get_knowledge_graph_stats(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    Get statistics about the knowledge graph
    
    Returns counts of documents, entities, and relationships in the knowledge graph.
    """
    try:
        # Execute Cypher queries to get statistics
        doc_count_query = "MATCH (d:Document) RETURN count(d) as count"
        entity_count_query = "MATCH (e) WHERE NOT e:Document RETURN count(e) as count"
        rel_count_query = "MATCH ()-[r]->() RETURN count(r) as count"
        
        # Get document count
        doc_results = await retrieval_service.execute_cypher_query_async(doc_count_query)
        doc_count = doc_results[0]["count"] if doc_results else 0
        
        # Get entity count
        entity_results = await retrieval_service.execute_cypher_query_async(entity_count_query)
        entity_count = entity_results[0]["count"] if entity_results else 0
        
        # Get relationship count
        rel_results = await retrieval_service.execute_cypher_query_async(rel_count_query)
        rel_count = rel_results[0]["count"] if rel_results else 0
        
        # Get last update time
        last_updated_query = """
        MATCH (n)
        RETURN n.created_at as timestamp
        ORDER BY n.created_at DESC
        LIMIT 1
        """
        time_results = await retrieval_service.execute_cypher_query_async(last_updated_query)
        
        # Format timestamp or use current time if not available
        if time_results and time_results[0].get("timestamp"):
            last_updated = time_results[0]["timestamp"]
        else:
            last_updated = datetime.now().isoformat()
        
        return DocumentStats(
            document_count=doc_count,
            entity_count=entity_count,
            relationship_count=rel_count,
            last_updated=last_updated
        )
        
    except Exception as e:
        logger.error(f"Failed to get knowledge graph stats: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")


@router.delete("/documents", response_model=Dict[str, Any])
async def reset_knowledge_graph(
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    Reset the knowledge graph
    
    Deletes all documents, entities, and relationships from the knowledge graph.
    Use with caution as this operation cannot be undone.
    """
    try:
        # Execute Cypher query to delete all nodes and relationships
        reset_query = "MATCH (n) DETACH DELETE n"
        await retrieval_service.execute_cypher_query_async(reset_query)
        
        logger.warning("Knowledge graph has been reset")
        
        return {
            "status": "success",
            "message": "Knowledge graph has been reset successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset knowledge graph: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to reset knowledge graph: {str(e)}")


# ================== Helper Functions ==================

async def _process_document_task(task_id: str, file_path, settings: GraphProcessingSettings):
    """Process a document to build the knowledge graph"""
    try:
        _graph_tasks[task_id]["status"] = "processing"
        logger.info(f"Starting processing of document: {file_path}")
        
        # Step 1: Load and process documents
        start_time = time.time()
        documents = load_and_process_documents(
            file_path=file_path,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap
        )
        logger.info(f"Document loaded and split into {len(documents)} chunks")
        
        # Step 2: Create graph documents
        _graph_tasks[task_id]["status"] = "transforming"
        graph_documents, llm = create_graph_from_documents(
            documents=documents,
            llm_model=settings.model
        )
        
        logger.info(f"Created {len(graph_documents)} graph documents")
        
        # Step 3: Save graph documents
        _graph_tasks[task_id]["status"] = "saving"
        saved_file = save_graph_documents(graph_documents)
        logger.info(f"Saved graph documents to {saved_file}")
        
        # Step 4: Import to Neo4j
        _graph_tasks[task_id]["status"] = "importing"
        from langchain_community.graphs import Neo4jGraph
        graph = Neo4jGraph(
            url=DATABASE.URI,
            username=DATABASE.USERNAME,
            password=DATABASE.PASSWORD
        )
        
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
        
        # Step 5: Create vector index if requested
        if settings.create_vector_index:
            _graph_tasks[task_id]["status"] = "creating_index"
            from src.main import create_vector_index
            vector_retriever = create_vector_index(
                neo4j_url=DATABASE.URI,
                neo4j_user=DATABASE.USERNAME,
                neo4j_password=DATABASE.PASSWORD,
                index_name=settings.vector_index_name,
                recreate=False
            )
            logger.info(f"Created vector index: {settings.vector_index_name}")
        
        # Update task info
        _graph_tasks[task_id]["status"] = "completed"
        _graph_tasks[task_id]["completed_at"] = time.time()
        _graph_tasks[task_id]["result"] = {
            "document_count": len(documents),
            "graph_nodes_count": sum(len(gd.get("nodes", [])) for gd in graph_documents),
            "graph_edges_count": sum(len(gd.get("edges", [])) for gd in graph_documents),
            "processing_time": time.time() - start_time,
            "output_file": saved_file
        }
        
        logger.info(f"Document processing task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Document processing task {task_id} failed: {str(e)}", exc_info=True)
        _graph_tasks[task_id]["status"] = "failed"
        _graph_tasks[task_id]["error"] = str(e)