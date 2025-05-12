"""
服务依赖 - 提供FastAPI依赖注入
负责创建和管理服务所需的共享资源
"""
import logging
import os
from functools import lru_cache
from typing import Optional

from src.core.graph_retrieval import GraphRetriever
from src.core.model_provider import ModelProvider
from src.config import DATABASE

logger = logging.getLogger(__name__)

@lru_cache(maxsize=1)
def get_llm():
    """
    获取LLM模型实例（单例模式）
    
    Returns:
        LLM模型实例
    """
    # 读取环境变量配置或使用默认值
    provider = os.environ.get("LLM_PROVIDER", "ollama")
    model_name = os.environ.get("LLM_MODEL", "qwen2.5")
    
    logger.info(f"初始化LLM模型: {provider}/{model_name}")
    
    # 获取LLM模型
    llm = ModelProvider.get_llm(
        provider=provider,
        model_name=model_name
    )
    
    if not llm:
        logger.warning(f"初始化LLM模型失败: {provider}/{model_name}")
    else:
        logger.info(f"成功初始化LLM模型: {provider}/{model_name}")
    
    return llm

@lru_cache(maxsize=1)
def get_embeddings():
    """
    获取嵌入模型实例（单例模式）
    
    Returns:
        嵌入模型实例
    """
    # 读取环境变量配置或使用默认值
    provider = os.environ.get("EMBEDDING_PROVIDER", "ollama")
    model_name = os.environ.get("EMBEDDING_MODEL", "nomic-embed-text")
    
    logger.info(f"初始化嵌入模型: {provider}/{model_name}")
    
    # 获取嵌入模型
    embeddings = ModelProvider.get_embeddings(
        provider=provider,
        model_name=model_name
    )
    
    if not embeddings:
        logger.warning(f"初始化嵌入模型失败: {provider}/{model_name}")
    else:
        logger.info(f"成功初始化嵌入模型: {provider}/{model_name}")
    
    return embeddings

@lru_cache(maxsize=1)
def get_graph_retriever():
    """
    获取图检索器实例（单例模式）
    
    Returns:
        GraphRetriever实例
    """
    # 获取LLM模型
    llm = get_llm()    # 获取Neo4j连接信息
    uri = os.environ.get("NEO4J_URI", getattr(DATABASE, "URI", "neo4j://localhost:7687"))
    user = os.environ.get("NEO4J_USER", getattr(DATABASE, "USERNAME", "neo4j"))
    password = os.environ.get("NEO4J_PASSWORD", getattr(DATABASE, "PASSWORD", "password"))
    database = os.environ.get("NEO4J_DATABASE", getattr(DATABASE, "DATABASE_NAME", "neo4j"))
    
    logger.info(f"初始化图检索器，连接到 {uri}, 用户: {user}, 数据库: {database}")
    
    # 创建图检索器
    retriever = GraphRetriever(
        uri=uri,
        user=user,
        password=password,
        database=database,
        llm=llm
    )
    
    # 初始化向量检索器
    try:
        embeddings = get_embeddings()
        if embeddings:
            retriever.initialize_vector_retriever(embeddings)
            logger.info("成功初始化向量检索器")
    except Exception as e:
        logger.error(f"初始化向量检索器失败: {str(e)}")
    
    return retriever
