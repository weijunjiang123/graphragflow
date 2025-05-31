"""
Text2Cypher服务 - 提供全局的自然语言转Cypher查询服务
"""
import logging
from typing import Optional

from src.core.text2cypher import GraphRetriever, Text2CypherService
from src.core.model_provider import ModelProvider
from src.config import DATABASE, MODEL

logger = logging.getLogger(__name__)

# 全局服务实例
_text2cypher_service_instance = None
_retrieval_service_instance = None

def get_text2cypher_service() -> Text2CypherService:
    """获取或创建Text2Cypher服务的全局实例
    
    Returns:
        Text2CypherService: 配置好的Text2Cypher服务实例
    """
    global _text2cypher_service_instance
    
    if _text2cypher_service_instance is None:
        logger.info("初始化Text2Cypher服务")
        try:
            model_provider = ModelProvider()
            llm = model_provider.get_llm()
            
            # Text2CypherService已经正确使用参数名称
            _text2cypher_service_instance = Text2CypherService(
                uri=DATABASE.URI,
                user=DATABASE.USERNAME,
                password=DATABASE.PASSWORD,
                database=DATABASE.DATABASE_NAME,
                llm=llm
            )
            logger.info("Text2Cypher服务初始化成功")
        except Exception as e:
            logger.error(f"初始化Text2Cypher服务失败: {str(e)}")
            raise
    
    return _text2cypher_service_instance

def get_retrieval_service() -> GraphRetriever:
    """获取或创建检索服务的全局实例
    使用单例模式避免重复初始化。
    
    Returns:
        GraphRetriever: 配置好的GraphRetriever实例
    """
    global _retrieval_service_instance
    
    if _retrieval_service_instance is None:
        logger.info("初始化检索服务")
        try:
            model_provider = ModelProvider()
            llm = model_provider.get_llm()
            
            # 修复参数名称：将uri改为connection_uri等
            _retrieval_service_instance = GraphRetriever(
                connection_uri=DATABASE.URI,
                username=DATABASE.USERNAME,
                password=DATABASE.PASSWORD,
                database_name=DATABASE.DATABASE_NAME,
                llm=llm
            )
            
            # 确保为text2cypher创建/优化索引
            if hasattr(_retrieval_service_instance, 'ensure_fulltext_index'):
                _retrieval_service_instance.ensure_fulltext_index()
            
            if hasattr(_retrieval_service_instance, 'ensure_vector_index'):
                _retrieval_service_instance.ensure_vector_index()
            
            logger.info("检索服务初始化成功")
        except Exception as e:
            logger.error(f"初始化检索服务失败: {str(e)}")
            raise
    
    return _retrieval_service_instance