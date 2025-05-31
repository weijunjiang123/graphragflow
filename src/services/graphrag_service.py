"""
GraphRAG集成服务 - 将图检索与大语言模型集成
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union
import asyncio

from src.config import MODEL, DATABASE, APP
from src.core.text2cypher import GraphRetriever
from src.core.model_provider import ModelProvider
from src.core.llm_context_generator import LLMContextGenerator
from src.core.embeddings import EmbeddingsManager
# Fix: Import Text2CypherIndexManager from the correct location
from src.core.text2cypher_manager import Text2CypherIndexManager

logger = logging.getLogger(__name__)

# Global service instance
_graphrag_service_instance = None

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
        
        # 初始化图检索器 - 修复参数名称
        self.retriever = GraphRetriever(
            connection_uri=self.db_uri,
            username=self.db_user,
            password=self.db_password,
            database_name=self.db_name,
            llm=self.llm,
            # ner_model=None  # 使用LLM进行实体识别
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
    
    def _optimize_text2cypher_indexes(self):
        """优化text2cypher索引以提高检索质量"""
        try:
            logger.info("开始优化text2cypher索引...")
            index_manager = Text2CypherIndexManager(
                uri=self.db_uri,
                user=self.db_user,
                password=self.db_password,
                database=self.db_name
            )
            
            # 创建和优化索引
            index_manager.optimize_indexes_for_text2cypher()
            index_manager.create_cypher_transformation_indexes()
            index_manager.create_text2cypher_schema_constraints()
            
            # 创建查询模板
            index_manager.create_text2cypher_query_templates()
            
            logger.info("text2cypher索引优化完成")
            return True
        except Exception as e:
            logger.error(f"优化text2cypher索引失败: {str(e)}")
            return False
    
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
                
            # 使用Text2CypherIndexManager处理和优化检索结果
            try:
                from src.core.text2cypher_manager import Text2CypherIndexManager
                index_manager = Text2CypherIndexManager(
                    uri=self.db_uri,
                    user=self.db_user,
                    password=self.db_password,
                    database=self.db_name
                )
                # 修复和优化检索结果中的merged_results
                retrieval_results = index_manager.optimize_and_repair_text_to_cypher_results(retrieval_results)
                index_manager.close()
            except Exception as e:
                logger.warning(f"优化检索结果失败，将使用原始结果: {str(e)}")
                
            merged_results_count = len(retrieval_results.get('merged_results', []))
            logger.info(f"检索完成，检索到 {merged_results_count} 条结果")
            
            # 如果没有检索到结果，尝试使用text2cypher进行回退检索
            if merged_results_count == 0 and hasattr(self.retriever, 'generate_cypher_query'):
                logger.info("使用text2cypher进行回退检索...")
                try:
                    # 生成Cypher查询
                    cypher_query, params = self.retriever.generate_cypher_query(query)
                    
                    if cypher_query:
                        # 执行查询
                        cypher_results = self.retriever.execute_cypher(cypher_query, params)
                        
                        if cypher_results:
                            logger.info(f"text2cypher回退检索成功，获取到 {len(cypher_results)} 条结果")
                            # 构建新的结果格式
                            graph_results = []
                            for result in cypher_results:
                                item_content = str(result)
                                item_id = f"cypher_{hash(item_content)}"
                                graph_results.append({
                                    "id": item_id,
                                    "content": item_content,
                                    "metadata": {"source": "text2cypher_fallback"},
                                    "score": 0.9,
                                    "sources": ["graph"]
                                })
                            
                            # 将结果加入到retrieval_results中
                            retrieval_results["graph_results"] = graph_results
                            retrieval_results["merged_results"] = graph_results
                except Exception as fallback_e:
                    logger.warning(f"text2cypher回退检索失败: {str(fallback_e)}")
            
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
            # 确保retrieval_results包含organized_context
            if "organized_context" not in retrieval_results or not retrieval_results["organized_context"]:
                # 如果检索结果中没有组织好的上下文且有merged_results，手动生成
                if "merged_results" in retrieval_results and retrieval_results["merged_results"]:
                    try:
                        from src.core.graph_retrieval import QueryAnalysisResult
                        # 构建简单的查询分析结果
                        query_analysis = QueryAnalysisResult(query)
                        query_analysis.query_type = "general"
                        query_analysis.key_concepts = []
                        
                        # 提取实体和执行的cypher查询
                        entities = retrieval_results.get("entities", [])
                        executed_cypher = retrieval_results.get("executed_cypher_query", "")
                        
                        # 调用组织上下文方法
                        retrieval_results["organized_context"] = self.retriever._organize_retrieval_context(
                            query=query,
                            query_analysis=query_analysis,
                            merged_results=retrieval_results["merged_results"],
                            entities=entities,
                            executed_cypher=executed_cypher
                        )
                        logger.info("已手动构建organized_context")
                    except Exception as ctx_e:
                        logger.warning(f"手动构建上下文失败: {str(ctx_e)}")
            
            # 生成LLM上下文
            llm_context = self.context_generator.generate_context(query, retrieval_results)
            context_length = len(llm_context.get('context_text', ''))
            logger.info(f"上下文生成完成，长度: {context_length}")
            
            # 如果上下文为空但有检索结果，直接使用检索结果作为上下文
            if context_length == 0 and "merged_results" in retrieval_results and retrieval_results["merged_results"]:
                raw_context = "\n\n".join([item.get("content", "") for item in retrieval_results["merged_results"]])
                llm_context["context_text"] = f"根据检索到的信息:\n\n{raw_context}"
                logger.info(f"使用原始检索结果作为上下文，长度: {len(llm_context['context_text'])}")
                
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
                "context_length": len(llm_context.get('context_text', '')),
                "retrieval_strategy": retrieval_strategy if retrieval_strategy != "auto" else retrieval_results.get("retrieval_strategy", "auto"),
                "entity_count": len(retrieval_results.get("entities", [])),
                "result_count": len(retrieval_results.get("merged_results", [])),
                "contexts": retrieval_results.get("merged_results", []),  # 添加检索到的上下文来源
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


def get_graphrag_service(max_context_length: int = 4000) -> GraphRAGService:
    """
    Get or create a GraphRAG service instance.
    Uses a singleton pattern to avoid re-initialization.
    
    Args:
        max_context_length: Maximum context length for LLM
        
    Returns:
        GraphRAGService instance
    """
    global _graphrag_service_instance
    
    if _graphrag_service_instance is None:
        logger.info("Initializing GraphRAG service")
        try:
            _graphrag_service_instance = GraphRAGService(max_context_length=max_context_length)
            logger.info("GraphRAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize GraphRAG service: {str(e)}")
            raise
    
    return _graphrag_service_instance