"""
GraphRAG FastAPI 应用入口
"""
import logging
import uvicorn
from fastapi import FastAPI, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from typing import Dict, Any, List

from src.config import APP, DATABASE, MODEL
from src.services.retrieval_service import get_retrieval_service, RetrievalService
from src.services.graphrag_service import GraphRAGService
from src.routers import retrieval, graphrag

# 设置日志
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

# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG API",
    description="图检索增强生成系统API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 在生产环境中应该限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(retrieval.router)
app.include_router(graphrag.router)

# 性能监控中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """添加处理时间头部"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# 健康检查端点
@app.get("/health", tags=["health"])
async def health_check():
    """健康检查端点"""
    return {
        "status": "ok",
        "version": "1.0.0"
    }

# 获取系统信息
@app.get("/info", tags=["system"])
async def system_info(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    graphrag_service: GraphRAGService = Depends(graphrag.get_graphrag_service)
):
    """获取系统信息"""
    return {
        "app": {
            "name": "GraphRAG API",
            "version": "1.0.0",
            "debug_mode": APP.DEBUG_MODE,
        },
        "database": {
            "uri": DATABASE.URI,
            "username": DATABASE.USERNAME,
            "database": DATABASE.DATABASE_NAME
        },
        "model": {
            "provider": MODEL.MODEL_PROVIDER,
            "model": MODEL.OPENAI_MODEL if MODEL.MODEL_PROVIDER == "openai" else MODEL.OLLAMA_LLM_MODEL,
            "vector_index": APP.VECTOR_INDEX_NAME
        },
        "retrieval_service": {
            "initialized": retrieval_service is not None,
            "vector_retriever_available": hasattr(retrieval_service, 'vector_retriever') and 
                                         retrieval_service.vector_retriever is not None
        },
        "graphrag_service": {
            "initialized": graphrag_service is not None,
            "llm_available": hasattr(graphrag_service, 'llm') and graphrag_service.llm is not None,
            "vector_retriever_available": hasattr(graphrag_service, 'vector_retriever') and 
                                         graphrag_service.vector_retriever is not None
        }
    }

# 应用启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时的事件处理器"""
    logger.info("GraphRAG API 服务正在启动...")
    # 预热检索服务
    try:
        retrieval_service = get_retrieval_service()
        logger.info("检索服务初始化成功")
    except Exception as e:
        logger.error(f"检索服务初始化失败: {e}")
        
    # 预热GraphRAG服务
    try:
        graphrag_service = graphrag.get_graphrag_service()
        logger.info("GraphRAG服务初始化成功")
    except Exception as e:
        logger.error(f"GraphRAG服务初始化失败: {e}")
        
    logger.info("GraphRAG API 服务启动完成")

# 应用关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时的事件处理器"""
    logger.info("GraphRAG API 服务正在关闭...")
    try:
        retrieval_service = get_retrieval_service()
        if retrieval_service:
            retrieval_service.close()
            
        graphrag_service = graphrag.get_graphrag_service()
        if graphrag_service:
            graphrag_service.close()
    except Exception as e:
        logger.error(f"关闭服务时出错: {e}")
    logger.info("GraphRAG API 服务已关闭")

# 直接运行时的入口
if __name__ == "__main__":
    uvicorn.run(
        "src.app:app",
        host="0.0.0.0",
        port=8000,
        reload=APP.DEBUG_MODE
    )