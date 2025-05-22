import logging
from fastapi import FastAPI, Depends, HTTPException
from contextlib import asynccontextmanager

# Import configurations and core components
from src.config import DATABASE, DOCUMENT, MODEL, APP
from src.core.document_processor import DocumentProcessor
from src.core.embeddings import EmbeddingsManager
from src.core.entity_extraction import EntityExtractor
from src.core.graph_transformer import GraphTransformerWrapper
from src.core.neo4j_manager import Neo4jManager, Neo4jConnectionManager
from src.core.model_provider import ModelProvider
from src.utils import setup_logging

# Import routers (will be created in subsequent steps)
from .routers import documents, query, entities

# Setup logging
setup_logging(log_level=APP.LOG_LEVEL_ENUM, log_file=APP.LOG_FILE)
logger = logging.getLogger(__name__)

# --- Global Service Instances ---
# These will be initialized in the lifespan event
app_services = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize all services
    logger.info("Application startup: Initializing services...")
    print("INFO:     Application startup: Initializing services...") # For uvicorn visibility

    try:
        # Initialize LLM
        llm = None
        if MODEL.PROVIDER == "ollama":
            llm = ModelProvider.get_llm(
                provider="ollama",
                model_name=MODEL.OLLAMA_LLM_MODEL, # Corrected from DOCUMENT.OLLAMA_LLM_MODEL
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
            raise RuntimeError(f"Failed to initialize LLM with provider {MODEL.PROVIDER}")
        app_services["llm"] = llm
        logger.info(f"LLM initialized with provider: {MODEL.PROVIDER}")
        print(f"INFO:     LLM initialized with provider: {MODEL.PROVIDER}")

        # Initialize Neo4jManager
        # The Neo4jConnectionManager handles the driver instance globally
        neo4j_manager = Neo4jManager(DATABASE.URI, DATABASE.USERNAME, DATABASE.PASSWORD)
        app_services["neo4j_manager"] = neo4j_manager
        logger.info("Neo4jManager initialized.")
        print("INFO:     Neo4jManager initialized.")

        # Initialize DocumentProcessor
        document_processor = DocumentProcessor(
            chunk_size=DOCUMENT.CHUNK_SIZE,
            chunk_overlap=DOCUMENT.CHUNK_OVERLAP
        )
        app_services["document_processor"] = document_processor
        logger.info("DocumentProcessor initialized.")
        print("INFO:     DocumentProcessor initialized.")

        # Initialize GraphTransformerWrapper
        graph_transformer = GraphTransformerWrapper(llm=llm)
        app_services["graph_transformer"] = graph_transformer
        logger.info("GraphTransformerWrapper initialized.")
        print("INFO:     GraphTransformerWrapper initialized.")
        
        # Initialize EmbeddingsManager and Vector Retriever
        embeddings_manager = EmbeddingsManager()
        app_services["embeddings_manager"] = embeddings_manager
        
        embeddings = embeddings_manager.get_working_embeddings(provider=MODEL.PROVIDER)
        if embeddings:
            app_services["embeddings_model"] = embeddings
            vector_retriever = embeddings_manager.create_vector_index(
                embeddings=embeddings,
                neo4j_url=DATABASE.URI,
                neo4j_user=DATABASE.USERNAME,
                neo4j_password=DATABASE.PASSWORD,
                index_name="document_vector_api", # Potentially different from main.py's index
                recreate=False # Avoid recreating frequently in API context, manage manually
            )
            if vector_retriever:
                app_services["vector_retriever"] = vector_retriever
                logger.info("EmbeddingsManager and Vector Retriever initialized.")
                print("INFO:     EmbeddingsManager and Vector Retriever initialized.")
            else:
                logger.warning("Failed to initialize vector retriever. Vector search might not work.")
                print("WARNING:  Failed to initialize vector retriever. Vector search might not work.")
                app_services["vector_retriever"] = None
        else:
            logger.warning("Failed to initialize embeddings model. Vector search will not be available.")
            print("WARNING:  Failed to initialize embeddings model. Vector search will not be available.")
            app_services["embeddings_model"] = None
            app_services["vector_retriever"] = None


        # Initialize EntityExtractor
        entity_extractor = EntityExtractor(llm=llm)
        app_services["entity_extractor"] = entity_extractor
        logger.info("EntityExtractor initialized.")
        print("INFO:     EntityExtractor initialized.")
        
        logger.info("All services initialized successfully.")
        print("INFO:     All services initialized successfully.")

    except Exception as e:
        logger.error(f"Error during service initialization: {e}", exc_info=True)
        print(f"ERROR:    Error during service initialization: {e}")
        # Depending on the severity, you might want to prevent the app from starting
        # by raising the exception or sys.exit()
        raise RuntimeError(f"Core service initialization failed: {e}")

    yield  # API is now running

    # Shutdown: Clean up resources
    logger.info("Application shutdown: Cleaning up resources...")
    print("INFO:     Application shutdown: Cleaning up resources...")
    Neo4jConnectionManager.close()
    logger.info("Neo4j connection closed.")
    print("INFO:     Neo4j connection closed.")
    app_services.clear()
    logger.info("Application services cleared.")
    print("INFO:     Application services cleared.")


# Create FastAPI app instance with lifespan management
app = FastAPI(
    title="GraphRAG API",
    description="API for processing documents, querying a graph database, and extracting entities.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Dependency Functions for Routers ---
def get_neo4j_manager() -> Neo4jManager:
    nm = app_services.get("neo4j_manager")
    if not nm:
        raise HTTPException(status_code=503, detail="Neo4jManager not available.")
    return nm

def get_document_processor() -> DocumentProcessor:
    dp = app_services.get("document_processor")
    if not dp:
        raise HTTPException(status_code=503, detail="DocumentProcessor not available.")
    return dp

def get_graph_transformer() -> GraphTransformerWrapper:
    gt = app_services.get("graph_transformer")
    if not gt:
        raise HTTPException(status_code=503, detail="GraphTransformer not available.")
    return gt

def get_embeddings_manager() -> EmbeddingsManager:
    em = app_services.get("embeddings_manager")
    if not em:
        raise HTTPException(status_code=503, detail="EmbeddingsManager not available.")
    return em
    
def get_vector_retriever(): # No type hint, can be None
    vr = app_services.get("vector_retriever")
    # Allow None if initialization failed, router should handle it
    return vr

def get_entity_extractor() -> EntityExtractor:
    ee = app_services.get("entity_extractor")
    if not ee:
        raise HTTPException(status_code=503, detail="EntityExtractor not available.")
    return ee

# Include routers
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(query.router, prefix="/api/v1", tags=["Query"])
app.include_router(entities.router, prefix="/api/v1", tags=["Entities"])

# Basic root endpoint
@app.get("/", tags=["Root"])
async def read_root():
    return {"message": "Welcome to the GraphRAG API. Navigate to /docs for API documentation."}

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    # Check Neo4j connection
    neo4j_status = "disconnected"
    try:
        nm = get_neo4j_manager()
        with nm.driver.session() as session:
            session.run("RETURN 1")
        neo4j_status = "connected"
    except Exception as e:
        logger.error(f"Health check: Neo4j connection failed: {e}")
        neo4j_status = f"error: {str(e)}"

    # Check LLM (indirectly via EntityExtractor or GraphTransformer)
    llm_status = "not_initialized"
    if app_services.get("llm"):
        llm_status = "initialized"
        try:
            # A light check, e.g. try to invoke entity extractor with a simple string
            ee = get_entity_extractor()
            ee.extract("hello") # A simple test
            llm_status = "responsive"
        except Exception as e:
            logger.error(f"Health check: LLM responsiveness test failed: {e}")
            llm_status = f"error: {str(e)}"


    return {
        "status": "ok",
        "services": {
            "neo4j": neo4j_status,
            "llm": llm_status,
            "document_processor": "initialized" if app_services.get("document_processor") else "not_initialized",
            "graph_transformer": "initialized" if app_services.get("graph_transformer") else "not_initialized",
            "embeddings_manager": "initialized" if app_services.get("embeddings_manager") else "not_initialized",
            "vector_retriever": "initialized" if app_services.get("vector_retriever") else "not_initialized_or_failed",
            "entity_extractor": "initialized" if app_services.get("entity_extractor") else "not_initialized",
        }
    }

if __name__ == "__main__":
    # This is for local development testing if you run this file directly.
    # Production deployment should use a proper ASGI server like Uvicorn.
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
