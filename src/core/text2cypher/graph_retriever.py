"""
GraphRetriever模块 - 提供自然语言到Cypher查询的转换

使用LangChain的GraphCypherQAChain实现Neo4j图数据库的自然语言查询
"""
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple

from langchain_neo4j import GraphCypherQAChain, Neo4jGraph
from langchain_core.prompts.prompt import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.language_models.llms import BaseLLM

from src.config import DATABASE, MODEL

logger = logging.getLogger(__name__)

# 默认Cypher生成提示模板
DEFAULT_CYPHER_GENERATION_TEMPLATE = """任务: 生成Cypher语句来查询图数据库。

说明:
仅使用提供的模式中的关系类型和属性。
不要使用未提供的其他关系类型或属性。
将复杂查询分解为更简单的查询。
为查询添加适当的限制条件，避免返回太多结果。

模式:
{schema}

注意: 不要在您的回复中包含任何解释或道歉。
不要回答除了构建Cypher语句之外的任何问题。
不要包含生成的Cypher语句以外的任何文本。

用户问题:
{question}"""


class GraphRetriever:
    """使用LangChain的GraphCypherQAChain实现的图检索器"""
    
    def __init__(self, 
                llm: Optional[BaseLLM] = None,
                connection_uri: Optional[str] = None,
                username: Optional[str] = None,
                password: Optional[str] = None,
                database_name: Optional[str] = None,
                cypher_template: Optional[str] = None,
                top_k: int = 10,
                return_direct: bool = False,
                verbose: bool = False,
                enhanced_schema: bool = True):
        """初始化图检索器
        
        Args:
            llm: 用于生成Cypher查询和回答的语言模型
            connection_uri: Neo4j数据库连接URI
            username: Neo4j用户名
            password: Neo4j密码
            database_name: Neo4j数据库名称
            cypher_template: 用于生成Cypher的提示模板
            top_k: 返回的最大结果数
            return_direct: 是否直接返回数据库结果
            verbose: 是否显示详细日志
            enhanced_schema: 是否使用增强的图模式信息
        """
        # 使用提供的参数或默认配置
        self.connection_uri = connection_uri or DATABASE.URI
        self.username = username or DATABASE.USERNAME
        self.password = password or DATABASE.PASSWORD
        self.database_name = database_name or DATABASE.DATABASE_NAME
        self.top_k = top_k
        self.return_direct = return_direct
        self.verbose = verbose
        self.enhanced_schema = enhanced_schema
        
        # 设置LLM
        self.llm = llm or self._initialize_llm()
        
        # 初始化Neo4j图
        self.graph = self._initialize_graph()
        
        # 设置Cypher生成提示
        cypher_template = cypher_template or DEFAULT_CYPHER_GENERATION_TEMPLATE
        self.cypher_prompt = PromptTemplate(
            input_variables=["schema", "question"], 
            template=cypher_template
        )
        
        # 初始化GraphCypherQAChain
        self.chain = self._initialize_chain()
        
        logger.info("GraphRetriever初始化完成")
    
    def _initialize_llm(self) -> BaseLLM:
        """根据配置初始化语言模型"""
        if MODEL.MODEL_PROVIDER in ["openai", "openaiapi"]:
            # 直接将 api_base 作为参数传递，不再放入 model_kwargs
            return ChatOpenAI(
                temperature=0,
                model=MODEL.OPENAI_MODEL,
                api_key=MODEL.OPENAI_API_KEY,
                base_url=MODEL.OPENAI_API_BASE if MODEL.OPENAI_API_BASE != "https://api.openai.com/v1" else None
            )
        elif MODEL.MODEL_PROVIDER == "ollama":
            # 如果使用Ollama，需要先导入OllamaLLM
            from langchain_community.llms import Ollama
            
            return Ollama(
                model=MODEL.OLLAMA_LLM_MODEL,
                base_url=MODEL.OLLAMA_BASE_URL,
                temperature=0
            )
        else:
            raise ValueError(f"不支持的模型提供商: {MODEL.MODEL_PROVIDER}")

    def _initialize_graph(self) -> Neo4jGraph:
        """初始化Neo4j图连接"""
        try:
            graph = Neo4jGraph(
                url=self.connection_uri,
                username=self.username,
                password=self.password,
                database=self.database_name,
                enhanced_schema=self.enhanced_schema
            )
            # 刷新图模式
            graph.refresh_schema()
            logger.info(f"成功连接到Neo4j数据库 ({self.connection_uri})")
            
            if self.verbose:
                logger.debug(f"图模式:\n{graph.schema}")
                
            return graph
        except Exception as e:
            logger.error(f"连接Neo4j数据库失败: {str(e)}")
            raise

    def _initialize_chain(self) -> GraphCypherQAChain:
        """初始化GraphCypherQAChain"""
        return GraphCypherQAChain.from_llm(
            llm=self.llm,
            graph=self.graph,
            cypher_prompt=self.cypher_prompt,
            top_k=self.top_k,
            return_direct=self.return_direct,
            return_intermediate_steps=True,
            verbose=self.verbose,
            allow_dangerous_requests=True  # 允许执行自定义Cypher
        )
    
    def query(self, query: str) -> Dict[str, Any]:
        """执行自然语言查询并返回结果
        
        Args:
            query: 自然语言查询
            
        Returns:
            包含查询结果的字典
        """
        start_time = time.time()
        
        try:
            # 执行查询
            result = self.chain.invoke({"query": query})
            
            # 从中间步骤中提取生成的Cypher查询
            intermediate_steps = result.get("intermediate_steps", [])
            generated_cypher = intermediate_steps[0].get("query") if intermediate_steps else "未生成Cypher查询"
            raw_results = intermediate_steps[1].get("context") if len(intermediate_steps) > 1 else []
            
            # 构建返回结果
            retrieval_result = {
                "query": query,
                "generated_cypher": generated_cypher,
                "raw_results": raw_results,
                "result": result.get("result", ""),
                "entities": self._extract_entities(raw_results),
                "metadata": {
                    "elapsed_time": time.time() - start_time,
                    "model": MODEL.OPENAI_MODEL if MODEL.MODEL_PROVIDER == "openai" else MODEL.OLLAMA_LLM_MODEL
                }
            }
            
            # 如果启用了直接返回，则保留原始结果
            if self.return_direct:
                retrieval_result["direct_results"] = result.get("result", [])
            
            logger.info(f"查询 '{query[:50]}...' 执行完成，耗时: {retrieval_result['metadata']['elapsed_time']:.2f}秒")
            return retrieval_result
            
        except Exception as e:
            logger.error(f"执行查询时出错: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "raw_results": [],
                "result": f"查询执行失败: {str(e)}",
                "metadata": {
                    "elapsed_time": time.time() - start_time,
                    "success": False
                }
            }
    
    def _extract_entities(self, raw_results: List[Dict]) -> List[Dict]:
        """从原始结果中提取实体信息
        
        Args:
            raw_results: 数据库查询原始结果
            
        Returns:
            提取的实体列表
        """
        entities = []
        
        for item in raw_results:
            # 提取第一个键值作为实体ID
            if not item:
                continue
                
            # 获取第一个键作为实体名称
            first_key = next(iter(item.keys()), None)
            if first_key:
                entity_value = item[first_key]
                entity = {
                    "id": entity_value,
                    "type": first_key.split('.')[-2] if '.' in first_key else "Entity",
                    "property": first_key.split('.')[-1] if '.' in first_key else first_key
                }
                entities.append(entity)
        
        return entities
    
    def get_schema(self) -> str:
        """获取当前图数据库的模式信息
        
        Returns:
            格式化的模式字符串
        """
        # 刷新模式以确保最新
        self.graph.refresh_schema()
        return self.graph.schema
    
    def execute_cypher(self, cypher_query: str) -> List[Dict]:
        """直接执行Cypher查询
        
        Args:
            cypher_query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        try:
            return self.graph.query(cypher_query)
        except Exception as e:
            logger.error(f"执行Cypher查询时出错: {str(e)}")
            raise