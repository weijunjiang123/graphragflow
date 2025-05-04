from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import time
from pydantic import BaseModel, Field

from src.services.graphrag_service import GraphRAGService

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/graphrag",
    tags=["graphrag"],
    responses={404: {"description": "Not found"}},
)

# 请求和响应模型
class GraphRAGRequest(BaseModel):
    """GraphRAG请求模型"""
    query: str = Field(..., description="用户查询文本")
    system_prompt: Optional[str] = Field(None, description="系统提示词")
    retrieval_strategy: str = Field("auto", description="检索策略: auto, graph_only, balanced, vector_only")
    graph_weight: float = Field(0.5, description="图检索权重", ge=0.0, le=1.0)
    vector_weight: float = Field(0.5, description="向量检索权重", ge=0.0, le=1.0)
    include_context: bool = Field(False, description="是否在响应中包含上下文")


# 服务实例
_graphrag_service = None

def get_graphrag_service():
    """获取GraphRAG服务实例"""
    global _graphrag_service
    if _graphrag_service is None:
        _graphrag_service = GraphRAGService()
    return _graphrag_service


@router.post("/query", response_model=Dict[str, Any])
async def process_query(
    request: GraphRAGRequest,
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """
    处理用户查询，使用图检索增强大语言模型生成
    
    根据用户查询，从知识图谱和向量存储中检索相关信息，
    然后将检索到的信息作为上下文提供给大语言模型，生成回答。
    """
    try:
        start_time = time.time()
        logger.info(f"接收GraphRAG查询请求: {request.query}")
        
        # 处理查询
        result = await graphrag_service.process_query_async(
            query=request.query,
            system_prompt=request.system_prompt,
            retrieval_strategy=request.retrieval_strategy,
            graph_weight=request.graph_weight,
            vector_weight=request.vector_weight
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"GraphRAG查询处理完成，耗时: {elapsed_time:.2f}秒")
        
        # 添加性能指标
        result["performance"] = {
            "total_time": elapsed_time,
            "retrieval_time": result.get("retrieval_time", 0),
            "generation_time": result.get("generation_time", 0)
        }
        
        return result
    
    except Exception as e:
        logger.error(f"GraphRAG查询处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"GraphRAG处理失败: {str(e)}")


@router.get("/health", response_model=Dict[str, Any])
async def health_check(
    graphrag_service: GraphRAGService = Depends(get_graphrag_service)
):
    """
    检查GraphRAG服务状态
    
    验证各个组件（图检索、向量检索、LLM等）是否正常工作。
    """
    status = {
        "status": "healthy",
        "components": {
            "llm": graphrag_service.llm is not None,
            "graph_retriever": graphrag_service.retriever is not None,
            "vector_retriever": graphrag_service.vector_retriever is not None,
            "context_generator": graphrag_service.context_generator is not None
        }
    }
    
    if not all(status["components"].values()):
        status["status"] = "degraded"
        
    return status