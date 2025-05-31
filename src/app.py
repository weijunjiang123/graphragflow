"""
GraphRAG FastAPI Application Entry Point
"""
import logging
import uvicorn
from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict, Any, List

from src.config import APP, DATABASE, MODEL
from src.services.graphrag_service import GraphRAGService, get_graphrag_service
from src.services.text2cypher_service import get_text2cypher_service, get_retrieval_service
from src.core.text2cypher import GraphRetriever
# Import the router modules
from src.routers import chat, knowledge, graph_query

# Define RetrievalService as an alias for GraphRetriever for compatibility
RetrievalService = GraphRetriever

# Set up logging
logging.basicConfig(
    level=APP.LOG_LEVEL_ENUM,
    format=APP.LOG_FORMAT,
    datefmt=APP.LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(APP.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="GraphRAG Q&A API",
    description="Intelligent Question Answering with Graph-enhanced Retrieval Augmented Generation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register RESTful API routers
app.include_router(chat.router, prefix="/api")
app.include_router(knowledge.router, prefix="/api")
app.include_router(graph_query.router, prefix="/api")

# Performance monitoring middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred", "type": str(type(exc).__name__)}
    )

# Health check endpoint
@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": "1.0.0"
    }

# System information endpoint
@app.get("/system", tags=["system"])
async def system_info(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """System information endpoint"""
    # Get text2cypher service status
    text2cypher_service = None
    try:
        text2cypher_service = get_text2cypher_service()
    except Exception:
        pass
        
    return {
        "app": {
            "name": "GraphRAG Q&A API",
            "version": "1.0.0",
            "debug_mode": APP.DEBUG_MODE,
        },
        "database": {
            "uri": DATABASE.URI,
            "database": DATABASE.DATABASE_NAME
        },
        "model": {
            "provider": MODEL.MODEL_PROVIDER,
            "model": MODEL.OPENAI_MODEL if MODEL.MODEL_PROVIDER == "openai" else MODEL.OLLAMA_LLM_MODEL,
            "vector_index": APP.VECTOR_INDEX_NAME
        },
        "services": {
            "retrieval": {
                "initialized": retrieval_service is not None,
                "vector_retriever": hasattr(retrieval_service, 'vector_retriever') and 
                                    retrieval_service.vector_retriever is not None,
                "graph_retriever": hasattr(retrieval_service, 'retriever') and 
                                   retrieval_service.retriever is not None
            },
            "graphrag": {
                "initialized": graphrag_service is not None,
                "llm": hasattr(graphrag_service, 'llm') and graphrag_service.llm is not None,
                "context_generator": hasattr(graphrag_service, 'context_generator') and 
                                    graphrag_service.context_generator is not None
            },
            "text2cypher": {
                "initialized": text2cypher_service is not None,
                "graph": hasattr(text2cypher_service, 'graph') and text2cypher_service.graph is not None,
                "chain": hasattr(text2cypher_service, 'chain') and text2cypher_service.chain is not None
            }
        }
    }

# Application startup event
@app.on_event("startup")
async def startup_event():
    """Application startup event handler"""
    logger.info("GraphRAG Q&A API service starting...")
    # Pre-warm services
    try:
        retrieval_service = get_retrieval_service()
        logger.info("Retrieval service initialized successfully")
    except Exception as e:
        logger.error(f"Retrieval service initialization failed: {e}")
        
    # Pre-warm GraphRAG service
    try:
        graphrag_service = get_graphrag_service()
        logger.info("GraphRAG service initialized successfully")
    except Exception as e:
        logger.error(f"GraphRAG service initialization failed: {e}")
    
    # Pre-warm Text2Cypher service
    try:
        text2cypher_service = get_text2cypher_service()
        logger.info("Text2Cypher service initialized successfully")
    except Exception as e:
        logger.error(f"Text2Cypher service initialization failed: {e}")
        
    logger.info("GraphRAG Q&A API service started")

# Application shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event handler"""
    logger.info("GraphRAG Q&A API service shutting down...")
    try:
        retrieval_service = get_retrieval_service()
        if retrieval_service:
            retrieval_service.close()
            
        graphrag_service = get_graphrag_service()
        if graphrag_service:
            graphrag_service.close()
            
        # No explicit close needed for text2cypher_service as it uses Neo4jGraph
        # which manages its own connection lifecycle
    except Exception as e:
        logger.error(f"Error during service shutdown: {e}")
    logger.info("GraphRAG Q&A API service stopped")

# Direct execution entry point
if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=APP.DEBUG_MODE
    )