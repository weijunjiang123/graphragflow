"""
GraphRAG API - 搜索路由
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from src.api.models import SearchRequest, SearchResponse
from src.service.text2cypher_service import Text2CypherService
from src.service.dependencies import get_graph_retriever
from src.api.cache import async_cached, text2cypher_cache
from src.api.versioning import APIVersioning

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/search", response_model=SearchResponse, summary="执行统一搜索")
@async_cached(text2cypher_cache, prefix="search")
async def search(
    request: SearchRequest,
    req: Request,
    res: Response,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    统一搜索接口，支持多种搜索类型
    
    - **query**: 搜索查询
    - **limit**: 结果数量限制 (默认: 5)
    - **search_type**: 搜索类型 (text2cypher, hybrid, vector, fulltext)
    
    根据search_type参数，选择相应的搜索方式：
    - **text2cypher**: 使用text2Cypher搜索，将自然语言转换为Cypher查询
    - **hybrid**: 使用混合搜索，结合向量和图结构
    - **vector**: 使用向量搜索，基于语义相似度
    - **fulltext**: 使用全文索引搜索，基于关键词匹配
    
    返回:
    - **success**: 是否成功
    - **original_query**: 原始查询
    - **search_type**: 使用的搜索类型
    - **results**: 搜索结果
    - **result_count**: 结果数量
    - **cypher_query**: 生成的Cypher查询 (仅当search_type为text2cypher时)
    """
    try:
        # 获取API版本并添加到响应
        version = APIVersioning.version_header(req)
        APIVersioning.version_response(res, version)
        
        service = Text2CypherService(graph_retriever)
        timestamp = req.state.start_time if hasattr(req.state, "start_time") else None
        
        if request.search_type == "text2cypher":
            result = service.search(request.query, request.limit)
            result["search_type"] = "text2cypher"
            result["success"] = True
            result["timestamp"] = timestamp
            return result
            
        elif request.search_type == "hybrid":
            results = graph_retriever.hybrid_search(request.query, request.limit)
            return {
                "success": True,
                "original_query": request.query,
                "search_type": "hybrid",
                "results": results,
                "result_count": len(results),
                "timestamp": timestamp
            }
            
        elif request.search_type == "vector":
            results = graph_retriever.vector_search(request.query, request.limit)
            return {
                "success": True,
                "original_query": request.query,
                "search_type": "vector",
                "results": [doc.dict() for doc in results] if hasattr(results[0], "dict") else results,
                "result_count": len(results),
                "timestamp": timestamp
            }
            
        elif request.search_type == "fulltext":
            results = graph_retriever.fulltext_search(request.query, request.limit)
            return {
                "success": True,
                "original_query": request.query,
                "search_type": "fulltext",
                "results": results,
                "result_count": len(results),
                "timestamp": timestamp
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"不支持的搜索类型: {request.search_type}")
            
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")
