"""
Text2Cypher API路由 - 提供自然语言到Cypher查询的API接口
"""
import logging
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.services import get_text2cypher_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/graph-query",
    tags=["graph-query"],
    responses={404: {"description": "Not found"}},
)

class Text2CypherRequest(BaseModel):
    """自然语言到Cypher的查询请求"""
    query: str = Field(..., description="自然语言查询", example="列出所有Person节点及其名称")
    return_direct: bool = Field(False, description="是否直接返回数据库结果，而不经过LLM解释")
    top_k: Optional[int] = Field(10, description="返回结果的最大数量", ge=1, le=100)


@router.post("/text2cypher", response_model=Dict[str, Any], summary="将自然语言转换为Cypher查询并执行")
async def text_to_cypher(request: Text2CypherRequest):
    """将自然语言查询转换为Cypher查询并执行
    
    - **query**: 自然语言查询，例如"查找所有Person节点的名字"
    - **return_direct**: 是否直接返回数据库结果，而不经过LLM解释
    - **top_k**: 返回结果的最大数量
    
    返回:
    - **query**: 原始自然语言查询
    - **generated_cypher**: 生成的Cypher查询
    - **raw_results**: 数据库查询结果
    - **result**: LLM解释的结果
    - **entities**: 提取的实体列表
    - **metadata**: 查询元数据
    """
    try:
        service = get_text2cypher_service(
            top_k=request.top_k,
            return_direct=request.return_direct
        )
        
        result = service.query(request.query)
        return result
    except Exception as e:
        logger.error(f"Text2Cypher查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"查询处理失败: {str(e)}")


@router.post("/execute-cypher", response_model=Dict[str, Any], summary="直接执行Cypher查询")
async def execute_cypher(cypher_query: str):
    """直接执行Cypher查询
    
    - **cypher_query**: Cypher查询语句
    
    返回:
    - **results**: 查询结果列表
    """
    try:
        service = get_text2cypher_service()
        results = service.execute_cypher(cypher_query)
        
        return {
            "cypher_query": cypher_query,
            "results": results,
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"执行Cypher查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"执行Cypher查询失败: {str(e)}")


@router.get("/schema", response_model=Dict[str, Any], summary="获取图数据库模式")
async def get_graph_schema():
    """获取当前Neo4j图数据库的模式信息
    
    返回:
    - **schema**: 格式化的图数据库模式字符串
    """
    try:
        service = get_text2cypher_service()
        schema = service.get_schema()
        
        return {
            "schema": schema
        }
    except Exception as e:
        logger.error(f"获取图模式失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取图模式失败: {str(e)}")