from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import logging
import time
from pydantic import BaseModel, Field

from src.core.graph_retrieval import GraphRetriever
from src.services.retrieval_service import RetrievalService, get_retrieval_service

# 设置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(
    prefix="/api/retrieval",
    tags=["retrieval"],
    responses={404: {"description": "Not found"}},
)


# 请求和响应模型
class QueryRequest(BaseModel):
    """检索请求模型"""
    query: str = Field(..., description="用户查询文本")
    limit: int = Field(5, description="结果数量限制")
    strategy: Optional[str] = Field("auto", description="检索策略: auto, graph_only, balanced, vector_only")
    graph_weight: float = Field(0.5, description="图检索权重", ge=0.0, le=1.0)
    vector_weight: float = Field(0.5, description="向量检索权重", ge=0.0, le=1.0)
    context_entities: Optional[List[str]] = Field(None, description="上下文相关实体列表")


class EntityRequest(BaseModel):
    """实体检索请求模型"""
    entity_name: str = Field(..., description="实体名称")
    fuzzy_match: bool = Field(True, description="是否进行模糊匹配")
    limit: int = Field(5, description="结果数量限制")


class PathRequest(BaseModel):
    """路径查找请求模型"""
    source_entity_id: str = Field(..., description="起始实体ID")
    target_entity_id: str = Field(..., description="目标实体ID")
    max_depth: int = Field(3, description="最大路径长度", ge=1, le=5)
    relation_types: Optional[List[str]] = Field(None, description="要考虑的关系类型列表")


class CypherRequest(BaseModel):
    """Cypher查询请求模型"""
    query_description: str = Field(..., description="查询描述")
    schema_info: Optional[str] = Field(None, description="数据库模式信息")


class Text2CypherRequest(BaseModel):
    """Text2Cypher搜索请求模型"""
    query: str = Field(..., description="用户查询文本")
    force_cypher: Optional[str] = Field(None, description="可选的强制执行的Cypher查询")
    cypher_params: Optional[Dict[str, Any]] = Field(None, description="Cypher查询参数")
    use_cache: bool = Field(True, description="是否使用查询缓存")


class ExecuteCypherRequest(BaseModel):
    """执行Cypher查询请求模型"""
    cypher_query: str = Field(..., description="Cypher查询语句")
    params: Optional[Dict[str, Any]] = Field(None, description="查询参数")


@router.post("/query", response_model=Dict[str, Any])
async def retrieve_knowledge(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    执行知识检索查询

    根据用户查询，使用多策略检索从知识库中获取相关信息。
    可以指定检索策略和各种参数来优化结果。
    """
    try:
        start_time = time.time()
        logger.info(f"接收查询请求: {request.query}")

        # 选择策略并执行检索
        if request.strategy == "auto":
            # 自动选择策略
            results = await retrieval_service.multi_strategy_retrieval_async(
                query=request.query,
                limit=request.limit
            )
        else:
            # 使用指定的权重
            results = await retrieval_service.enhanced_hybrid_search_async(
                query=request.query,
                limit=request.limit,
                graph_weight=request.graph_weight,
                vector_weight=request.vector_weight,
                context_entities=request.context_entities
            )

        elapsed_time = time.time() - start_time
        logger.info(f"查询处理完成，耗时: {elapsed_time:.2f}秒")

        # 添加性能指标
        results["performance"] = {
            "total_time": elapsed_time,
            "query_analysis_time": results.get("elapsed_time", 0)
        }

        return results

    except Exception as e:
        logger.error(f"查询处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"检索处理失败: {str(e)}")


@router.post("/entities", response_model=List[Dict[str, Any]])
async def find_entities(
    request: EntityRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    查找与指定名称匹配的实体

    支持精确匹配和模糊匹配，返回符合条件的实体列表。
    """
    try:
        entities = await retrieval_service.find_entities_by_name_async(
            entity_name=request.entity_name,
            fuzzy_match=request.fuzzy_match,
            limit=request.limit
        )
        return entities
    except Exception as e:
        logger.error(f"实体查找失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"实体查找失败: {str(e)}")


@router.get("/entity/{entity_id}/neighbors", response_model=Dict[str, Any])
async def get_entity_neighbors(
    entity_id: str,
    hop: int = Query(1, ge=1, le=3, description="跳数(1-3)"),
    limit: int = Query(10, ge=1, le=50, description="结果数量限制"),
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    获取实体的邻居节点

    返回与指定实体直接或间接相连的其他实体及其关系信息。
    可以指定跳数(1-3)来控制搜索深度。
    """
    try:
        neighbors = await retrieval_service.get_entity_neighbors_async(
            entity_id=entity_id,
            hop=hop,
            limit=limit
        )
        return neighbors
    except Exception as e:
        logger.error(f"获取实体邻居失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取实体邻居失败: {str(e)}")


@router.post("/path", response_model=Dict[str, Any])
async def find_path(
    request: PathRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    查找两个实体之间的路径

    寻找连接两个实体的最短路径，可以指定最大深度和关系类型过滤器。
    """
    try:
        path = await retrieval_service.find_shortest_path_async(
            source_entity_id=request.source_entity_id,
            target_entity_id=request.target_entity_id,
            max_depth=request.max_depth,
            relation_types=request.relation_types
        )
        return path
    except Exception as e:
        logger.error(f"路径查找失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"路径查找失败: {str(e)}")


@router.post("/cypher/generate", response_model=Dict[str, Any])
async def generate_cypher(
    request: CypherRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    生成Cypher查询语句

    根据自然语言描述，生成与需求相匹配的Cypher查询语句。
    可选择性地提供数据库模式信息以提高生成质量。
    """
    try:
        cypher_query = await retrieval_service.generate_cypher_query_async(
            query_description=request.query_description,
            schema_info=request.schema_info
        )
        
        return {
            "query_description": request.query_description,
            "cypher_query": cypher_query
        }
    except Exception as e:
        logger.error(f"Cypher生成失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Cypher生成失败: {str(e)}")


@router.post("/analyze-query", response_model=Dict[str, Any])
async def analyze_query(
    query: str = Query(..., description="要分析的查询文本"),
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    分析查询文本

    对查询文本进行分析，提取实体、关键概念，并识别查询类型。
    """
    try:
        analysis_result = await retrieval_service.analyze_query_async(query)
        
        # 将分析结果转换为JSON可序列化格式
        return {
            "query": query,
            "entities": analysis_result.entities,
            "entity_names": analysis_result.get_entity_names(),
            "key_concepts": analysis_result.key_concepts,
            "relations": analysis_result.relations,
            "query_type": analysis_result.query_type,
            "context_constraints": analysis_result.context_constraints
        }
    except Exception as e:
        logger.error(f"查询分析失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"查询分析失败: {str(e)}")


@router.post("/search/async", response_model=Dict[str, Any])
async def async_search(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    异步搜索接口

    提交一个搜索请求并立即返回任务ID，搜索结果将在后台处理。
    客户端可以稍后通过任务ID获取结果。
    """
    try:
        task_id = await retrieval_service.submit_async_search_task(
            query=request.query,
            limit=request.limit,
            strategy=request.strategy,
            graph_weight=request.graph_weight,
            vector_weight=request.vector_weight,
            context_entities=request.context_entities
        )
        
        # 在后台任务中执行实际搜索
        background_tasks.add_task(
            retrieval_service.process_async_search_task,
            task_id=task_id,
            query=request.query,
            limit=request.limit,
            strategy=request.strategy,
            graph_weight=request.graph_weight,
            vector_weight=request.vector_weight,
            context_entities=request.context_entities
        )
        
        return {"task_id": task_id, "status": "submitted"}
    except Exception as e:
        logger.error(f"异步搜索提交失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"异步搜索提交失败: {str(e)}")


@router.get("/search/async/{task_id}", response_model=Dict[str, Any])
async def get_async_search_result(
    task_id: str,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    获取异步搜索结果

    通过任务ID获取之前提交的异步搜索任务的结果。
    """
    try:
        result = await retrieval_service.get_async_search_result(task_id)
        if result is None:
            return {"task_id": task_id, "status": "pending", "message": "任务正在处理中"}
        return result
    except Exception as e:
        logger.error(f"获取异步搜索结果失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"获取异步搜索结果失败: {str(e)}")


@router.post("/text2cypher", response_model=Dict[str, Any])
async def text2cypher_search(
    request: Text2CypherRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    使用Text2Cypher进行检索
    
    将自然语言查询自动转换为Cypher查询并执行，返回结果。
    可以选择性地提供强制执行的Cypher查询和参数。
    
    - 如果提供force_cypher，将直接执行指定的查询
    - 否则，系统会自动生成适合查询的Cypher语句
    - 如果查询生成或执行失败，将回退到标准检索方法
    """
    try:
        start_time = time.time()
        logger.info(f"接收Text2Cypher请求: {request.query}")
        
        results = await retrieval_service.text2cypher_search_async(
            query=request.query,
            force_cypher=request.force_cypher,
            cypher_params=request.cypher_params,
            use_cache=request.use_cache
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Text2Cypher处理完成，耗时: {elapsed_time:.2f}秒")
        
        # 添加性能指标
        if "performance" not in results:
            results["performance"] = {}
        
        results["performance"]["total_time"] = elapsed_time
        
        return results
        
    except Exception as e:
        logger.error(f"Text2Cypher处理失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Text2Cypher处理失败: {str(e)}")


@router.post("/cypher/execute", response_model=Dict[str, Any])
async def execute_cypher(
    request: ExecuteCypherRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service)
):
    """
    执行Cypher查询
    
    直接执行用户提供的Cypher查询语句和参数。
    适用于高级用户或已经有现成Cypher查询的情况。
    """
    try:
        start_time = time.time()
        logger.info(f"执行Cypher查询: {request.cypher_query}")
        
        results = await retrieval_service.execute_cypher_query_async(
            cypher_query=request.cypher_query,
            params=request.params
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"Cypher查询执行完成，耗时: {elapsed_time:.2f}秒")
        
        return {
            "cypher_query": request.cypher_query,
            "params": request.params,
            "results": results,
            "results_count": len(results),
            "elapsed_time": elapsed_time
        }
        
    except Exception as e:
        logger.error(f"执行Cypher查询失败: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"执行Cypher查询失败: {str(e)}")