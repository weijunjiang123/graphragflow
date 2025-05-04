"""
检索服务模块 - 提供同步和异步的图检索服务
"""
import asyncio
import uuid
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import functools
import time
import threading

from fastapi import Depends

from src.config import MODEL, DATABASE, APP
from src.core.graph_retrieval import GraphRetriever, QueryAnalysisResult
from src.core.model_provider import ModelProvider
from src.core.embeddings import EmbeddingsManager

# 设置日志
logger = logging.getLogger(__name__)

# 任务结果缓存
_task_cache = {}


class RetrievalService:
    """检索服务类 - 封装GraphRetriever并提供同步和异步接口"""
    
    def __init__(self):
        """初始化检索服务"""
        # 配置
        self.db_uri = DATABASE.URI
        self.db_user = DATABASE.USERNAME
        self.db_password = DATABASE.PASSWORD
        self.db_name = DATABASE.DATABASE_NAME
        
        # 初始化模型提供者
        self.model_provider = ModelProvider()
        self.llm = self.model_provider.get_llm()
        
        # 初始化嵌入模型 - 增强错误处理
        self.embeddings_manager = EmbeddingsManager()
        
        # 尝试不同的嵌入模型
        self.embeddings = None
        embedding_models = ["nomic-embed-text", "llama3", "all-MiniLM-L6-v2"]
        
        for model in embedding_models:
            try:
                logger.info(f"尝试初始化嵌入模型: {model}")
                self.embeddings = self.embeddings_manager.get_working_embeddings(
                    provider=MODEL.MODEL_PROVIDER,
                    model_name=model
                )
                if self.embeddings:
                    logger.info(f"成功初始化嵌入模型: {model}")
                    break
            except Exception as e:
                logger.warning(f"初始化嵌入模型 {model} 失败: {str(e)}")
        
        if not self.embeddings:
            logger.warning("所有嵌入模型初始化失败，将使用备用嵌入策略")
            # 实现备用嵌入策略 - 使用简单的平均词向量
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                logger.info("尝试使用HuggingFace嵌入模型")
                self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                logger.info("成功初始化HuggingFace嵌入模型")
            except Exception as e:
                logger.error(f"初始化HuggingFace嵌入模型失败: {str(e)}")
                self.embeddings = None
        
        # 初始化图检索器
        self.retriever = GraphRetriever(
            uri=self.db_uri,
            user=self.db_user,
            password=self.db_password,
            database=self.db_name,
            llm=self.llm,
            ner_model=None  # 使用LLM进行实体识别
        )
        
        # 初始化向量检索器
        self.vector_retriever = None
        if self.embeddings:
            try:
                self.vector_retriever = self.retriever.initialize_vector_retriever(
                    embeddings=self.embeddings,
                    index_name=APP.VECTOR_INDEX_NAME
                )
                logger.info("向量检索器初始化成功")
            except Exception as e:
                logger.error(f"向量检索器初始化失败: {str(e)}")
                self.vector_retriever = None
        else:
            logger.warning("由于嵌入模型不可用，向量检索器未初始化")
        
        logger.info("检索服务初始化完成")
        # 查询缓存
        self._query_cache = {}
    
    # 同步方法
    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """分析用户查询"""
        return self.retriever.analyze_query(query)
    
    def find_entities_by_name(self, entity_name: str, fuzzy_match: bool = True, limit: int = 5) -> List[Dict[str, Any]]:
        """根据名称查找实体"""
        return self.retriever.find_entities_by_name(entity_name, fuzzy_match, limit)
    
    def get_entity_neighbors(self, entity_id: str, hop: int = 1, limit: int = 10) -> Dict[str, Any]:
        """获取实体的邻居节点"""
        return self.retriever.get_entity_neighbors(entity_id, hop, limit)
    
    def find_shortest_path(self, source_entity_id: str, target_entity_id: str, 
                          max_depth: int = 4, relation_types: List[str] = None) -> Dict[str, Any]:
        """查找两个实体之间的最短路径"""
        return self.retriever.find_shortest_path(source_entity_id, target_entity_id, max_depth, relation_types)
    
    def generate_cypher_query(self, query_description: str, schema_info: str = None) -> tuple:
        """生成Cypher查询语句和参数
        
        Returns:
            tuple: (cypher_query, params)
        """
        return self.retriever.generate_cypher_query(query_description, schema_info)
        
    def execute_cypher_query(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行Cypher查询
        
        Args:
            cypher_query: Cypher查询语句
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        return self.retriever.execute_cypher(cypher_query, params)
        
    def text2cypher_search(self, query: str, force_cypher: str = None, 
                         cypher_params: Dict[str, Any] = None, use_cache: bool = True) -> Dict[str, Any]:
        """使用Text2Cypher进行检索
        
        Args:
            query: 用户查询
            force_cypher: 可选的强制执行的Cypher查询（优先于生成的查询）
            cypher_params: 与force_cypher配合使用的查询参数
            use_cache: 是否使用查询缓存
            
        Returns:
            包含检索结果和执行信息的字典
        """
        # 查询缓存键
        cache_key = f"text2cypher:{query}:{force_cypher}"
        
        # 如果使用缓存且存在缓存结果，直接返回
        if use_cache and cache_key in self._query_cache:
            logger.info(f"命中查询缓存: {cache_key}")
            return self._query_cache[cache_key]
            
        # 记录开始时间
        start_time = time.time()
        
        if force_cypher:
            # 使用提供的查询
            cypher_query = force_cypher
            params = cypher_params or {}
        else:
            # 调用LLM生成查询
            cypher_query, params = self.retriever.generate_cypher_query(query)
            
        # 如果生成查询失败，回退到多策略检索
        if not cypher_query:
            logger.warning(f"无法为查询生成Cypher: {query}，回退到多策略检索")
            return self.multi_strategy_retrieval(query, use_text2cypher=False)
            
        # 执行查询
        try:
            results = self.retriever.execute_cypher(cypher_query, params)
            
            # 构建结果
            result_data = {
                "query": query,
                "executed_cypher": cypher_query,
                "executed_params": params,
                "results_count": len(results),
                "results": results,
                "elapsed_time": time.time() - start_time,
                "source": "text2cypher"
            }
            
            # 缓存结果
            if use_cache:
                self._query_cache[cache_key] = result_data
                # 设置缓存过期时间（10分钟）
                self._schedule_cache_expiry(cache_key, 600)
                
            return result_data
            
        except Exception as e:
            logger.error(f"执行Text2Cypher查询失败: {str(e)}\nCypher: {cypher_query}\n参数: {params}")
            # 出错时回退到多策略检索
            return self.multi_strategy_retrieval(query, use_text2cypher=False)
    
    def multi_strategy_retrieval(self, query: str, limit: int = 5, use_text2cypher: bool = True,
                               force_cypher: str = None, cypher_params: Dict[str, Any] = None, 
                               force_strategy: str = None) -> Dict[str, Any]:
        """使用多策略进行检索 - 委托给GraphRetriever的multi_strategy_retrieval方法
        
        Args:
            query: 用户查询
            limit: 结果数量限制
            use_text2cypher: 是否允许使用Text2Cypher策略
            force_cypher: 可选的强制执行的Cypher查询
            cypher_params: 与force_cypher配合使用的查询参数
            force_strategy: 强制使用的策略名称
            
        Returns:
            检索结果字典
        """
        logger.info(f"执行多策略检索: {query}")
        return self.retriever.multi_strategy_retrieval(
            query=query,
            use_text2cypher=use_text2cypher and self.llm is not None,
            force_cypher=force_cypher,
            cypher_params=cypher_params,
            force_strategy=force_strategy
        )
    
    def enhanced_hybrid_search(self, query: str, limit: int = 5, graph_weight: float = 0.5, 
                             vector_weight: float = 0.5, context_entities: List[str] = None) -> Dict[str, Any]:
        """增强混合搜索 - 委托给GraphRetriever的enhanced_hybrid_search方法
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            graph_weight: 图检索结果权重
            vector_weight: 向量检索结果权重
            context_entities: 上下文相关实体列表
            
        Returns:
            混合搜索结果
        """
        logger.info(f"执行增强混合搜索: {query}, 图权重: {graph_weight}, 向量权重: {vector_weight}")
        return self.retriever.enhanced_hybrid_search(
            query=query,
            limit=limit,
            graph_weight=graph_weight,
            vector_weight=vector_weight,
            context_entities=context_entities
        )
        
    def _schedule_cache_expiry(self, cache_key: str, ttl_seconds: int):
        """设置缓存过期定时器"""
        def _expire_cache():
            if cache_key in self._query_cache:
                del self._query_cache[cache_key]
                logger.debug(f"缓存项已过期: {cache_key}")
                
        # 使用定时器设置过期
        timer = threading.Timer(ttl_seconds, _expire_cache)
        timer.daemon = True
        timer.start()
    
    # 异步方法 - 将同步方法包装成异步方法
    async def analyze_query_async(self, query: str) -> QueryAnalysisResult:
        """异步分析用户查询"""
        return await asyncio.to_thread(self.analyze_query, query)
    
    async def find_entities_by_name_async(self, entity_name: str, fuzzy_match: bool = True, limit: int = 5) -> List[Dict[str, Any]]:
        """异步根据名称查找实体"""
        return await asyncio.to_thread(self.find_entities_by_name, entity_name, fuzzy_match, limit)
    
    async def get_entity_neighbors_async(self, entity_id: str, hop: int = 1, limit: int = 10) -> Dict[str, Any]:
        """异步获取实体的邻居节点"""
        return await asyncio.to_thread(self.get_entity_neighbors, entity_id, hop, limit)
    
    async def find_shortest_path_async(self, source_entity_id: str, target_entity_id: str, 
                                      max_depth: int = 4, relation_types: List[str] = None) -> Dict[str, Any]:
        """异步查找两个实体之间的最短路径"""
        return await asyncio.to_thread(self.find_shortest_path, source_entity_id, target_entity_id, max_depth, relation_types)
    
    async def generate_cypher_query_async(self, query_description: str, schema_info: str = None) -> str:
        """异步生成Cypher查询语句"""
        return await asyncio.to_thread(self.generate_cypher_query, query_description, schema_info)
    
    async def execute_cypher_query_async(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """异步执行Cypher查询"""
        return await asyncio.to_thread(self.execute_cypher_query, cypher_query, params)
        
    async def text2cypher_search_async(self, query: str, force_cypher: str = None, 
                                     cypher_params: Dict[str, Any] = None, use_cache: bool = True) -> Dict[str, Any]:
        """异步使用Text2Cypher进行检索"""
        return await asyncio.to_thread(self.text2cypher_search, query, force_cypher, cypher_params, use_cache)
    
    async def enhanced_hybrid_search_async(self, query: str, limit: int = 5, graph_weight: float = 0.5, 
                                         vector_weight: float = 0.5, context_entities: List[str] = None) -> Dict[str, Any]:
        """异步增强混合搜索"""
        return await asyncio.to_thread(
            self.enhanced_hybrid_search, 
            query, limit, graph_weight, vector_weight, context_entities
        )
    
    async def multi_strategy_retrieval_async(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """异步多策略检索"""
        return await asyncio.to_thread(self.multi_strategy_retrieval, query, limit)
    
    # 异步任务相关方法
    async def submit_async_search_task(self, **kwargs) -> str:
        """提交异步搜索任务，返回任务ID"""
        task_id = str(uuid.uuid4())
        _task_cache[task_id] = {
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "params": kwargs
        }
        logger.info(f"提交异步搜索任务: {task_id}")
        return task_id
    
    async def process_async_search_task(self, task_id: str, **kwargs):
        """处理异步搜索任务"""
        try:
            logger.info(f"开始处理异步任务: {task_id}")
            _task_cache[task_id]["status"] = "processing"
            
            # 获取搜索参数
            query = kwargs.get("query")
            limit = kwargs.get("limit", 5)
            strategy = kwargs.get("strategy", "auto")
            graph_weight = kwargs.get("graph_weight", 0.5)
            vector_weight = kwargs.get("vector_weight", 0.5)
            context_entities = kwargs.get("context_entities")
            
            # 执行搜索
            start_time = time.time()
            
            if strategy == "auto":
                result = await self.multi_strategy_retrieval_async(query, limit)
            else:
                result = await self.enhanced_hybrid_search_async(
                    query, limit, graph_weight, vector_weight, context_entities
                )
                
            elapsed_time = time.time() - start_time
            
            # 添加任务信息
            result["task_id"] = task_id
            result["elapsed_time"] = elapsed_time
            result["status"] = "completed"
            
            # 更新缓存
            _task_cache[task_id]["result"] = result
            _task_cache[task_id]["status"] = "completed"
            _task_cache[task_id]["completed_at"] = datetime.now().isoformat()
            
            logger.info(f"异步任务完成: {task_id}, 耗时: {elapsed_time:.2f}秒")
            
            # 设置结果过期时间（1小时后过期）
            asyncio.create_task(self._expire_task_result(task_id, 3600))
            
        except Exception as e:
            logger.error(f"异步任务处理失败: {task_id}, 错误: {str(e)}", exc_info=True)
            _task_cache[task_id]["status"] = "failed"
            _task_cache[task_id]["error"] = str(e)
    
    async def get_async_search_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取异步搜索结果"""
        if task_id not in _task_cache:
            return {"task_id": task_id, "status": "not_found", "message": "未找到该任务"}
            
        task_info = _task_cache[task_id]
        if task_info["status"] == "completed":
            return task_info["result"]
        elif task_info["status"] == "failed":
            return {
                "task_id": task_id,
                "status": "failed",
                "error": task_info.get("error", "未知错误"),
                "created_at": task_info["created_at"]
            }
        else:
            return {
                "task_id": task_id,
                "status": task_info["status"],
                "message": "任务正在处理中",
                "created_at": task_info["created_at"]
            }
    
    async def _expire_task_result(self, task_id: str, ttl_seconds: int):
        """设置任务结果过期时间"""
        await asyncio.sleep(ttl_seconds)
        if task_id in _task_cache:
            del _task_cache[task_id]
            logger.info(f"任务结果已过期: {task_id}")
    
    def close(self):
        """关闭服务，释放资源"""
        if hasattr(self, 'retriever') and self.retriever:
            self.retriever.close()
            logger.info("检索服务已关闭")


# 单例服务
_service_instance = None

# 依赖注入函数
def get_retrieval_service() -> RetrievalService:
    """获取检索服务实例（依赖注入）"""
    global _service_instance
    if _service_instance is None:
        _service_instance = RetrievalService()
    return _service_instance