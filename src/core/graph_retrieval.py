import logging
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

from src.config import DOCUMENT, MODEL, DATABASE
from src.core.embeddings import EmbeddingsManager
from src.core.model_provider import ModelProvider
from src.core.progress_tracker import ProgressTracker
from src.main import Neo4jConnectionManager, create_fulltext_index, setup_entity_extraction

logger = logging.getLogger(__name__)

class GraphRetriever:
    """用于检索图数据库中知识的检索器类"""
    
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        """初始化图检索器
        
        Args:
            uri: Neo4j 数据库URI
            user: 用户名
            password: 密码
            database: 数据库名称(默认为"neo4j")
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.vector_store = None
        logger.info(f"初始化图检索器，连接到 {uri}")
        
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
            logger.info("关闭图数据库连接")
            
    def initialize_vector_retriever(self, embeddings, index_name: str = "document_vector"):
        """初始化向量检索器
        
        Args:
            embeddings: 嵌入模型
            index_name: 向量索引名称
            
        Returns:
            向量检索器对象
        """
        try:
            self.vector_store = Neo4jVector.from_existing_index(
                embeddings,
                url=self.uri,
                username=self.user,
                password=self.password,
                index_name=index_name,
                # 修改搜索类型为 vector 或添加 keyword_index 参数
                search_type="vector",  # 改为 vector 类型，避免需要 keyword_index
                # 或者使用：search_type="hybrid", keyword_index="fulltext_entity_id",
                node_label="Document",
                text_node_properties=["text"],
                embedding_node_property="embedding"
            )
            logger.info(f"成功初始化向量检索器，使用索引: {index_name}")
            return self.vector_store.as_retriever()
        except Exception as e:
            logger.error(f"初始化向量检索器失败: {str(e)}")
            raise
            
    def fulltext_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """使用全文索引搜索
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            
        Returns:
            搜索结果列表
        """
        cypher_query = """
        CALL db.index.fulltext.queryNodes("fulltext_entity_id", $query_text, {limit: $limit_val}) 
        YIELD node, score
        RETURN node.id AS id, labels(node) AS labels, score, 
            properties(node) AS properties
        ORDER BY score DESC
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                # 修改参数名称，避免与 Cypher 查询中的参数名冲突
                result = session.run(
                    cypher_query, 
                    query_text=query, 
                    limit_val=limit
                )
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"全文搜索失败: {str(e)}")
                return []
                
    def vector_search(self, query: str, limit: int = 5) -> List[Document]:
        """使用向量搜索
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            
        Returns:
            相关文档列表
        """
        if not self.vector_store:
            raise ValueError("向量存储未初始化，请先调用 initialize_vector_retriever()")
            
        try:
            retriever = self.vector_store.as_retriever(search_kwargs={"k": limit})
            return retriever.invoke(query)
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
            
    def hybrid_search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """使用混合搜索(图+向量)
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            
        Returns:
            混合搜索结果
        """
        # 提取可能的实体关键词
        keywords = query.split()
        entity_filter = " OR ".join([f"entity.id:*{keyword}*" for keyword in keywords])
        
        cypher_query = """
        // 第一部分：向量搜索
        CALL db.index.vector.queryNodes($index_name, $embedding, $limit) 
        YIELD node as doc, score as vector_score
        
        // 关联文档与实体
        OPTIONAL MATCH (doc)<-[:MENTIONED_IN|RELATED_TO]-(entity:__Entity__)
        WHERE doc.text IS NOT NULL
        
        // 收集结果
        WITH doc, entity, vector_score,
             CASE WHEN entity IS NOT NULL THEN 1 ELSE 0 END as has_entity
        
        RETURN doc.id AS id,
               labels(doc) AS type,
               doc.text AS content,
               vector_score AS relevance,
               COLLECT(DISTINCT {
                   id: entity.id, 
                   type: CASE WHEN entity IS NOT NULL THEN head(labels(entity)) ELSE NULL END
               }) AS related_entities,
               count(entity) as entity_count
        ORDER BY relevance DESC, entity_count DESC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            if not self.vector_store or not hasattr(self.vector_store, "_embeddings"):
                # 无法进行向量搜索，回退到全文搜索
                return self.fulltext_search(query, limit)
            
            try:
                # 获取查询的嵌入向量
                embedding = self.vector_store._embeddings.embed_query(query)
                
                # 执行混合查询
                result = session.run(
                    cypher_query, 
                    index_name="document_vector",
                    embedding=embedding,
                    limit=limit
                )
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"混合搜索失败: {str(e)}")
                # 出错时回退到全文搜索
                return self.fulltext_search(query, limit)
                
    def get_entity_relations(self, entity_id: str, depth: int = 1) -> Dict[str, Any]:
        """获取实体的关系图
        
        Args:
            entity_id: 实体ID
            depth: 图遍历深度
            
        Returns:
            包含节点和关系的图数据
        """
        cypher_query = """
        MATCH path = (e:__Entity__ {id: $entity_id})-[r*1..$depth]-(related)
        WHERE related:__Entity__ OR related:Document
        RETURN 
            COLLECT(DISTINCT e) + 
            COLLECT(DISTINCT related) AS nodes,
            COLLECT(DISTINCT relationships(path)) AS rels
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, entity_id=entity_id, depth=depth).single()
                if not result:
                    return {"nodes": [], "relationships": []}
                    
                # 处理节点
                nodes = []
                for node in result["nodes"]:
                    props = dict(node._properties)  # 获取所有属性
                    node_labels = list(node.labels)  # 获取节点标签
                    nodes.append({
                        "id": props.get("id", str(node.id)),
                        "label": node_labels[0] if node_labels else "Unknown",
                        "properties": props
                    })
                
                # 处理关系
                relationships = []
                for rel_list in result["rels"]:
                    for rel in rel_list:
                        relationships.append({
                            "source": rel.start_node["id"],
                            "target": rel.end_node["id"],
                            "type": rel.type,
                            "properties": dict(rel._properties)
                        })
                
                return {
                    "nodes": nodes,
                    "relationships": relationships
                }
                
            except Exception as e:
                logger.error(f"获取实体关系失败: {str(e)}")
                return {"nodes": [], "relationships": []}
            

def test_llm_retrieval():
    NEO4J_URL = DATABASE.URI
    NEO4J_USER = DATABASE.USERNAME
    NEO4J_PASSWORD = DATABASE.PASSWORD
    NEO4J_DATABASE = DATABASE.DATABASE_NAME

    progress = ProgressTracker(total_stages=7)

    fallback_provider = MODEL.MODEL_PROVIDER
    try:
        if fallback_provider == "ollama":
            llm = ModelProvider.get_llm(
                provider="ollama",
                model_name=DOCUMENT.OLLAMA_LLM_MODEL,
                base_url=MODEL.OLLAMA_BASE_URL,
                # temperature=0
            )
        else:
            llm = ModelProvider.get_llm(
                provider="openai",
                model_name=MODEL.OPENAI_MODEL,
                api_key=MODEL.OPENAI_API_KEY,
                api_base=MODEL.OPENAI_API_BASE,
                # temperature=0
            )
        
        if llm:
            print(f"✓ Successfully initialized fallback LLM with provider {fallback_provider}")
        else:
            raise ValueError(f"Failed to initialize fallback LLM with provider {fallback_provider}")
    except Exception as fallback_error:
        logger.error(f"Error initializing fallback LLM: {str(fallback_error)}")
        raise ValueError(f"Failed to initialize LLM with both primary and fallback providers")

    # 继续创建全文索引
    driver = Neo4jConnectionManager.get_instance(NEO4J_URL, (NEO4J_USER, NEO4J_PASSWORD))
    index_result = create_fulltext_index(driver)
    if index_result:
        print("✓ Fulltext index created/verified")
    
    # 创建图检索器对象
    graph_retriever = GraphRetriever(NEO4J_URL, NEO4J_USER, NEO4J_PASSWORD)
    
    # 初始化向量检索器
    try:
        # 重用之前创建的嵌入模型
        embeddings_manager = EmbeddingsManager()
        embeddings = embeddings_manager.get_working_embeddings(
            provider=MODEL.MODEL_PROVIDER, 
            api_key=MODEL.OPENAI_EMBEDDING_API_KEY, 
            api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
            model_name=MODEL.OPENAI_EMBEDDINGS_MODEL  # 添加模型名称参数
        )
        vector_retriever = graph_retriever.initialize_vector_retriever(embeddings)
        print("✓ 向量检索器初始化成功")
    except Exception as e:
        print(f"⚠️ 向量检索器初始化失败: {str(e)}")
        print("将使用全文搜索进行检索")
    # STAGE 7: Set up entity extraction
    progress.update("Setting up entity extraction")
    entity_chain = setup_entity_extraction(llm)
    
    # 示例查询
    print("\n演示图检索器能力:")
    try:
        query = "who is author of this？"
        print(f"查询: '{query}'")
        results = graph_retriever.hybrid_search(query, limit=3)
        
        print(f"找到 {len(results)} 个相关内容:")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result.get('content', '')[:100]}...")
            
            if result.get('related_entities'):
                print(f"     相关实体: {', '.join([e['id'] for e in result['related_entities'] if e['id']])}")
    except Exception as e:
        print(f"❌ 查询失败: {str(e)}")
    finally:
        # 记得在程序结束时关闭检索器
        graph_retriever.close()