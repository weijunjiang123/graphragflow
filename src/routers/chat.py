"""
Chat module router - Provides RESTful endpoints for the Q&A assistant
"""
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import Dict, Any, List, Optional
import logging
import time
import uuid
from pydantic import BaseModel, Field, validator

from src.services.graphrag_service import GraphRAGService, get_graphrag_service

# Configure logging
logger = logging.getLogger(__name__)

# Create router with appropriate tags and prefix
router = APIRouter(
    tags=["chat"],
    responses={404: {"description": "Not found"}}
)

# Task storage for async operations
_chat_tasks = {}

# ================== Models ==================

class Message(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model with conversation history"""
    messages: List[Message] = Field(..., description="Conversation history")
    settings: Optional[Dict[str, Any]] = Field(None, description="Optional chat settings")
    
    @validator('messages')
    def validate_messages(cls, v):
        """Validate that the last message is from the user"""
        if not v or len(v) == 0:
            raise ValueError("Messages cannot be empty")
        if v[-1].role != "user":
            raise ValueError("Last message must be from the user")
        return v


class ChatResponse(BaseModel):
    """Chat response model"""
    message: Message = Field(..., description="Assistant's response message")
    sources: Optional[List[Dict[str, Any]]] = Field(None, description="Knowledge sources used")
    stats: Optional[Dict[str, Any]] = Field(None, description="Performance statistics")


class AsyncTaskResponse(BaseModel):
    """Asynchronous task response model"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")


# ================== Endpoints ==================

@router.post("/conversations", response_model=ChatResponse)
async def create_chat_response(
    request: ChatRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """
    Create a chat response using GraphRAG
    
    Takes conversation history and generates a response using graph-enhanced retrieval.
    """
    try:
        start_time = time.time()
        user_query = request.messages[-1].content
        
        # Extract system prompt if present
        system_prompt = None
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
                break
                
        # Process settings if provided
        retrieval_strategy = "auto"
        graph_weight = 0.5
        vector_weight = 0.5
        include_sources = False
        
        if request.settings:
            retrieval_strategy = request.settings.get("retrieval_strategy", retrieval_strategy)
            graph_weight = request.settings.get("graph_weight", graph_weight)
            vector_weight = request.settings.get("vector_weight", vector_weight)
            include_sources = request.settings.get("include_sources", include_sources)
        
        # Process the query through GraphRAG
        result = await graphrag_service.process_query_async(
            query=user_query,
            system_prompt=system_prompt,
            retrieval_strategy=retrieval_strategy,
            graph_weight=graph_weight,
            vector_weight=vector_weight
        )
        
        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"Chat response generated in {elapsed_time:.2f}s")
        
        # Create response object
        response = ChatResponse(
            message=Message(
                role="assistant",
                content=result.get("response", "I'm sorry, I couldn't generate a response.")
            ),
            stats={
                "total_time": elapsed_time,
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "strategy": result.get("retrieval_strategy", retrieval_strategy)
            }
        )
        
        # Add sources if requested and available
        if include_sources and "contexts" in result:
            # Make sure to include all available contexts as sources
            response.sources = result["contexts"]
        elif include_sources and result.get("results", {}).get("merged_results"):
            # Backup location for contexts if the direct path isn't available
            response.sources = result["results"]["merged_results"]
            
        return response
        
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")


@router.post("/conversations/async", response_model=AsyncTaskResponse)
async def create_chat_response_async(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """
    Create an asynchronous chat response
    
    Starts a chat response task in the background and returns a task ID for polling.
    """
    try:
        task_id = str(uuid.uuid4())
        
        # Store task info
        _chat_tasks[task_id] = {
            "status": "processing",
            "created_at": time.time(),
            "result": None
        }
        
        # Process in background
        background_tasks.add_task(
            _process_chat_task,
            task_id=task_id,
            request=request,
            graphrag_service=graphrag_service
        )
        
        return AsyncTaskResponse(
            task_id=task_id,
            status="processing"
        )
        
    except Exception as e:
        logger.error(f"Failed to create async chat task: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create task: {str(e)}")


@router.get("/conversations/async/{task_id}", response_model=Dict[str, Any])
async def get_async_chat_result(task_id: str):
    """
    Get the result of an asynchronous chat task
    
    Returns the result if the task is complete, or the current status if it's still processing.
    """
    if task_id not in _chat_tasks:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        
    task_info = _chat_tasks[task_id]
    
    if task_info["status"] == "completed":
        return task_info["result"]
    elif task_info["status"] == "failed":
        return {
            "task_id": task_id,
            "status": "failed",
            "error": task_info.get("error", "Unknown error")
        }
    else:
        return {
            "task_id": task_id,
            "status": task_info["status"],
            "wait_time": time.time() - task_info["created_at"]
        }


# ================== Helper Functions ==================

async def _process_chat_task(task_id: str, request: ChatRequest, graphrag_service: GraphRAGService):
    """Process a chat task asynchronously"""
    try:
        user_query = request.messages[-1].content
        
        # Extract system prompt if present
        system_prompt = None
        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
                break
                
        # Process settings
        retrieval_strategy = "auto"
        graph_weight = 0.5
        vector_weight = 0.5
        include_sources = False
        
        if request.settings:
            retrieval_strategy = request.settings.get("retrieval_strategy", retrieval_strategy)
            graph_weight = request.settings.get("graph_weight", graph_weight)
            vector_weight = request.settings.get("vector_weight", vector_weight)
            include_sources = request.settings.get("include_sources", include_sources)
        
        # Process query
        start_time = time.time()
        result = await graphrag_service.process_query_async(
            query=user_query,
            system_prompt=system_prompt,
            retrieval_strategy=retrieval_strategy,
            graph_weight=graph_weight,
            vector_weight=vector_weight
        )
        
        elapsed_time = time.time() - start_time
        
        # Create response
        response = {
            "task_id": task_id,
            "status": "completed",
            "message": {
                "role": "assistant",
                "content": result.get("response", "")
            },
            "stats": {
                "total_time": elapsed_time,
                "retrieval_time": result.get("retrieval_time", 0),
                "generation_time": result.get("generation_time", 0),
                "strategy": result.get("retrieval_strategy", retrieval_strategy)
            }
        }
        
        # Handle sources/contexts more comprehensively
        # Check multiple possible locations for the context data
        if include_sources:
            if "contexts" in result:
                response["sources"] = result["contexts"]
            elif result.get("results", {}).get("merged_results"):
                response["sources"] = result["results"]["merged_results"]
            elif result.get("results", {}).get("organized_context", {}).get("graph_knowledge", {}).get("retrieved_entities"):
                response["sources"] = result["results"]["organized_context"]["graph_knowledge"]["retrieved_entities"]
        
        # Always flag if contexts are available (even if not included in response)
        if not include_sources:
            contexts_available = False
            if "contexts" in result and result["contexts"]:
                contexts_available = True
            elif result.get("results", {}).get("merged_results"):
                contexts_available = True
            elif result.get("results", {}).get("organized_context", {}).get("graph_knowledge", {}).get("retrieved_entities"):
                contexts_available = True
                
            if contexts_available:
                response["_contexts_available"] = True
            
        # Update task info
        _chat_tasks[task_id]["status"] = "completed"
        _chat_tasks[task_id]["result"] = response
        _chat_tasks[task_id]["completed_at"] = time.time()
        
        logger.info(f"Async chat task {task_id} completed in {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Async chat task {task_id} failed: {str(e)}", exc_info=True)
        _chat_tasks[task_id]["status"] = "failed"
        _chat_tasks[task_id]["error"] = str(e)