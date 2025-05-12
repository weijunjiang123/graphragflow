"""
Text2Cypher服务 - 业务逻辑层
处理text2Cypher查询请求，包装GraphRetriever的功能
"""
import logging
from typing import Dict, List, Any, Optional, Union

# 导入图检索器
from src.core.graph_retrieval import GraphRetriever

logger = logging.getLogger(__name__)

class Text2CypherService:
    """Text2Cypher服务类，封装图检索功能"""
    
    def __init__(self, graph_retriever: GraphRetriever):
        """
        初始化Text2Cypher服务
        
        Args:
            graph_retriever: 图检索器实例
        """
        self.graph_retriever = graph_retriever
        self.is_supported = self.graph_retriever.supports_text2cypher
        logger.info(f"初始化Text2Cypher服务，支持text2Cypher: {self.is_supported}")
    def search(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """
        执行text2Cypher搜索
        
        Args:
            query: 自然语言查询
            limit: 结果数量限制
            
        Returns:
            格式化的搜索结果
        """
        logger.info(f"执行text2Cypher搜索: {query}, limit={limit}")
        
        try:
            # 检查是否支持text2Cypher
            if not self.is_supported:
                logger.warning("当前模型不支持text2Cypher，使用混合搜索")
                results = self.graph_retriever.hybrid_search(query, limit)
                return {
                    "original_query": query,
                    "cypher_query": "不支持text2Cypher",
                    "results": results,
                    "result_count": len(results),
                    "fallback_method": "hybrid_search"
                }
            
            # 执行text2Cypher搜索
            result = self.graph_retriever.text2cypher_search(
                query=query, 
                limit=limit,
                return_formatted=True
            )
            logger.info(f"text2Cypher搜索完成，结果数: {result.get('result_count', 0)}")
            return result
        except Exception as e:
            logger.error(f"text2Cypher搜索失败: {str(e)}")
            try:
                # 出错时回退到混合搜索
                results = self.graph_retriever.hybrid_search(query, limit)
                return {
                    "original_query": query,
                    "cypher_query": f"执行失败: {str(e)}",
                    "results": results,
                    "result_count": len(results),
                    "error": str(e),
                    "fallback_method": "hybrid_search"
                }
            except Exception as inner_e:
                # 如果混合搜索也失败，返回一个空结果
                logger.error(f"混合搜索也失败: {str(inner_e)}")
                return {
                    "original_query": query,
                    "cypher_query": f"执行失败: {str(e)}",
                    "results": [],
                    "result_count": 0,
                    "error": f"{str(e)}; 混合搜索失败: {str(inner_e)}",
                    "fallback_method": "none"
                }
    
    def explain_results(self, results: List[Dict[str, Any]], query: str, cypher_query: str) -> str:
        """
        解释查询结果
        
        Args:
            results: 查询结果列表
            query: 原始查询
            cypher_query: Cypher查询
            
        Returns:
            结果解释
        """
        if not self.graph_retriever.llm:
            return "无法生成解释，未提供LLM模型"
        
        try:
            # 生成结果解释
            return self.graph_retriever.format_text2cypher_results(
                results=results,
                original_query=query,
                cypher_query=cypher_query
            ).get("explanation", "")
        except Exception as e:
            logger.error(f"解释结果失败: {str(e)}")
            return f"解释结果失败: {str(e)}"
    
    def get_examples(self) -> List[Dict[str, str]]:
        """
        获取text2Cypher示例查询
        
        Returns:
            示例查询列表
        """
        return self.graph_retriever.get_text2cypher_examples()
    
    def test_capability(self) -> Dict[str, Any]:
        """
        测试text2Cypher功能
        
        Returns:
            测试结果
        """
        return self.graph_retriever.test_text2cypher_capability()
