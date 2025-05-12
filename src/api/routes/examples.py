"""
GraphRAG API - 示例查询路由
"""
import logging
from fastapi import APIRouter, HTTPException, Depends, Request, Response
from typing import List
from src.api.models import ExampleQuery
from src.service.dependencies import get_graph_retriever
from src.api.cache import async_cached, example_cache
from src.api.versioning import APIVersioning

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/examples", response_model=List[ExampleQuery], summary="获取示例查询")
@async_cached(example_cache, prefix="examples")
async def get_examples(
    req: Request,
    res: Response,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    获取text2Cypher示例查询
    
    返回一系列示例查询，包括描述、查询文本和解释
    
    返回:
    - 示例查询列表，每个示例包含description、query和explanation字段
    """
    try:
        # 获取API版本并添加到响应
        version = APIVersioning.version_header(req)
        APIVersioning.version_response(res, version)
        
        examples = graph_retriever.get_text2cypher_examples()
        
        # 添加元数据
        response_data = {
            "success": True,
            "data": examples,
            "count": len(examples),
            "timestamp": req.state.start_time if hasattr(req.state, "start_time") else None
        }
        
        # 因为返回类型要求是List[ExampleQuery]，所以只返回数据部分
        return examples
    except Exception as e:
        logger.error(f"获取示例查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取示例查询失败: {str(e)}")
