"""
GraphRAG API - Text2Cypher路由
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from src.api.models import Text2CypherRequest, Text2CypherResponse
from src.service.text2cypher_service import Text2CypherService
from src.service.dependencies import get_graph_retriever
from src.api.cache import async_cached, text2cypher_cache
from src.api.versioning import APIVersioning

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/text2cypher", response_model=Text2CypherResponse, summary="执行Text2Cypher查询")
@async_cached(text2cypher_cache, prefix="text2cypher")
async def text2cypher(
    request: Text2CypherRequest,
    req: Request,
    res: Response,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    使用text2Cypher进行搜索
    
    将自然语言查询转换为Cypher查询语句，并执行查询，返回结果。
    
    - **query**: 自然语言查询
    - **limit**: 结果数量限制 (默认: 5)
    
    返回:
    - **original_query**: 原始查询
    - **cypher_query**: 生成的Cypher查询
    - **results**: 查询结果
    - **result_count**: 结果数量
    - **explanation**: 结果解释 (如果有)
    - **fallback_method**: 回退方法 (如果使用)
    """
    try:
        # 获取API版本并添加到响应
        version = APIVersioning.version_header(req)
        APIVersioning.version_response(res, version)
        
        service = Text2CypherService(graph_retriever)
        result = service.search(request.query, request.limit)
        
        # 统一响应格式
        result["success"] = True
        result["timestamp"] = req.state.start_time if hasattr(req.state, "start_time") else None
        
        return result
    except Exception as e:
        logger.error(f"text2Cypher搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@router.get("/test", summary="测试Text2Cypher功能")
async def test_text2cypher(
    req: Request,
    res: Response,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    测试API服务状态和text2Cypher功能
    
    返回:
    - **api_status**: API状态
    - **text2cypher_supported**: 是否支持text2Cypher
    - **test_result**: 测试结果
    """
    try:
        # 获取API版本并添加到响应
        version = APIVersioning.version_header(req)
        APIVersioning.version_response(res, version)
        
        test_result = graph_retriever.test_text2cypher_capability()
        return {
            "success": True,
            "api_status": "online",
            "text2cypher_supported": test_result["supports_text2cypher"],
            "test_result": test_result,
            "timestamp": req.state.start_time if hasattr(req.state, "start_time") else None
        }
    except Exception as e:
        logger.error(f"测试API失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")
