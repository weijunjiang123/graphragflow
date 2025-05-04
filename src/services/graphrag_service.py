"""
GraphRAG集成服务 - 将图检索与大语言模型集成
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union
import asyncio

from src.config import MODEL, DATABASE, APP
from src.core.graph_retrieval import GraphRetriever
from src.core.model_provider import ModelProvider
from src.core.llm_context_generator import LLMContextGenerator
from src.core.embeddings import EmbeddingsManager

logger = logging.getLogger(__name__)

class GraphRAGService:
    """GraphRAG集成服务 - 集成图检索和LLM生成"""
    
    def __init__(self, max_context_length: int = 4000):
        """初始化GraphRAG服务
        
        Args:
            max_context_length: LLM上下文最大长度
        """
        # 配置数据库连接
        self.db_uri = DATABASE.URI
        self.db_user = DATABASE.USERNAME
        self.db_password = DATABASE.PASSWORD
        self.db_name = DATABASE.DATABASE_NAME
        
        # 初始化模型提供者
        self.model_provider = ModelProvider()
        self.llm = self.model_provider.get_llm()
        
        # 初始化嵌入模型
        self.embeddings_manager = EmbeddingsManager()
        self.embeddings = self._initialize_embeddings()
        
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
            logger.warning("嵌入模型不可用，向量检索器未初始化")
        
        # 初始化LLM上下文生成器
        self.context_generator = LLMContextGenerator(
            graph_retriever=self.retriever,
            max_context_length=max_context_length
        )
        
        logger.info("GraphRAG服务初始化完成")
    
    def _initialize_embeddings(self):
        """初始化嵌入模型，尝试多种模型
        
        Returns:
            嵌入模型实例，如果初始化失败则返回None
        """
        # 尝试不同的嵌入模型
        embedding_models = ["nomic-embed-text", "llama3", "all-MiniLM-L6-v2"]
        
        for model in embedding_models:
            try:
                logger.info(f"尝试初始化嵌入模型: {model}")
                embeddings = self.embeddings_manager.get_working_embeddings(
                    provider=MODEL.MODEL_PROVIDER,
                    model_name=model
                )
                if embeddings:
                    logger.info(f"成功初始化嵌入模型: {model}")
                    return embeddings
            except Exception as e:
                logger.warning(f"初始化嵌入模型 {model} 失败: {str(e)}")
        
        # 如果所有Ollama模型都失败，尝试HuggingFace模型
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("尝试使用HuggingFace嵌入模型")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            logger.info("成功初始化HuggingFace嵌入模型")
            return embeddings
        except Exception as e:
            logger.error(f"初始化HuggingFace嵌入模型失败: {str(e)}")
            return None
    
    def process_query(self, query: str, system_prompt: Optional[str] = None,
                     retrieval_strategy: str = "auto", graph_weight: float = 0.5,
                     vector_weight: float = 0.5) -> Dict[str, Any]:
        """处理用户查询 - 检索 + 生成
        
        Args:
            query: 用户查询
            system_prompt: 系统提示词
            retrieval_strategy: 检索策略
            graph_weight: 图检索权重
            vector_weight: 向量检索权重
            
        Returns:
            包含响应的字典
        """
        start_time = time.time()
        
        # 第一步：执行检索
        try:
            if retrieval_strategy == "auto":
                retrieval_results = self.retriever.multi_strategy_retrieval(query)
            else:
                retrieval_results = self.retriever.enhanced_hybrid_search(
                    query=query,
                    graph_weight=graph_weight,
                    vector_weight=vector_weight
                )
            logger.info(f"检索完成，检索到 {len(retrieval_results.get('merged_results', []))} 条结果")
        except Exception as e:
            logger.error(f"检索失败: {str(e)}")
            return {
                "query": query,
                "response": f"检索失败: {str(e)}，无法提供准确回答。",
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
        
        # 第二步：生成LLM上下文
        try:
            llm_context = self.context_generator.generate_context(query, retrieval_results)
            logger.info(f"上下文生成完成，长度: {len(llm_context.get('context_text', ''))}")
        except Exception as e:
            logger.error(f"上下文生成失败: {str(e)}")
            return {
                "query": query,
                "response": f"上下文生成失败: {str(e)}，无法提供准确回答。",
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
        
        # 第三步：生成LLM响应
        if not self.llm:
            logger.error("LLM未初始化，无法生成响应")
            return {
                "query": query,
                "response": "系统错误：语言模型未初始化，无法生成响应。",
                "elapsed_time": time.time() - start_time,
                "error": "LLM未初始化"
            }
            
        try:
            generation_result = self.context_generator.generate_response(
                llm=self.llm,
                llm_context=llm_context,
                query=query,
                system_prompt=system_prompt
            )
            
            # 合并结果
            result = {
                "query": query,
                "response": generation_result.get("response", ""),
                "elapsed_time": time.time() - start_time,
                "retrieval_time": retrieval_results.get("elapsed_time", 0),
                "generation_time": generation_result.get("elapsed_time", 0),
                "context_length": generation_result.get("context_length", 0),
                "retrieval_strategy": retrieval_strategy if retrieval_strategy != "auto" else retrieval_results.get("retrieval_strategy", "auto"),
                "entity_count": len(retrieval_results.get("entities", [])),
            }
            
            logger.info(f"查询处理完成，总耗时: {result['elapsed_time']:.2f}秒")
            return result
            
        except Exception as e:
            logger.error(f"响应生成失败: {str(e)}")
            return {
                "query": query,
                "response": f"响应生成失败: {str(e)}，无法提供准确回答。",
                "elapsed_time": time.time() - start_time,
                "error": str(e)
            }
    
    async def process_query_async(self, query: str, system_prompt: Optional[str] = None,
                                retrieval_strategy: str = "auto", graph_weight: float = 0.5,
                                vector_weight: float = 0.5) -> Dict[str, Any]:
        """异步处理用户查询
        
        Args:
            query: 用户查询
            system_prompt: 系统提示词
            retrieval_strategy: 检索策略
            graph_weight: 图检索权重
            vector_weight: 向量检索权重
            
        Returns:
            包含响应的字典
        """
        # 将同步方法包装为异步
        return await asyncio.to_thread(
            self.process_query,
            query=query,
            system_prompt=system_prompt,
            retrieval_strategy=retrieval_strategy,
            graph_weight=graph_weight,
            vector_weight=vector_weight
        )
    
    def close(self):
        """关闭服务，释放资源"""
        if hasattr(self, 'retriever') and self.retriever:
            self.retriever.close()
            logger.info("GraphRAG服务已关闭")