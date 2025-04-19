import json
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

    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j", llm=None, ner_model=None):
        """初始化图检索器

        Args:
            uri: Neo4j 数据库URI
            user: 用户名
            password: 密码
            database: 数据库名称(默认为"neo4j")
            llm: 大语言模型 (optional)
            ner_model: 命名实体识别模型 (optional)
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.vector_store = None
        self.llm = llm
        self.ner_model = ner_model
        logger.info(f"初始化图检索器，连接到 {uri}")
    """用于检索图数据库中知识的检索器类"""
    
        
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
                text_node_property="text",
                embedding_node_property="embedding"
            )
            logger.info(f"成功初始化向量检索器，使用索引: {index_name}")
            return self.vector_store.as_retriever()
        except Exception as e:
            logger.error(f"初始化向量检索器失败: {str(e)}")
            raise

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """从文本中提取实体

        Args:
            text: 输入文本

        Returns:
            实体列表 [{"entity": "实体名", "type": "实体类型"}]
        """
        logger.info(f"从文本中提取实体: {text[:100]}...")
        extracted_entities = []

        if self.ner_model:
            # 使用专用NER模型提取实体
            entities = self.ner_model(text)
            extracted_entities = entities
            logger.info(f"NER模型提取的实体: {extracted_entities}")
            return entities
        elif self.llm:
            # 使用LLM提取实体
            prompt = f"""
            从以下文本中提取所有的命名实体（人物、组织、地点、产品等）:

            {text}

            以JSON数组格式返回，每个实体包含实体名称和实体类型:
            [
                {{"entity": "实体名称", "type": "实体类型"}}
            ]
            仅返回JSON数组，不要包含其他解释。
            """

            try:
                result = self.llm.invoke(prompt)

                # 处理LLM返回的JSON
                import re
                json_pattern = r'\[(.*?)\]'
                match = re.search(json_pattern, result, re.DOTALL)

                if match:
                    try:
                        json_str = f"[{match.group(1)}]"
                        entities = json.loads(json_str)
                        extracted_entities = entities
                        logger.info(f"LLM提取的实体: {extracted_entities}")
                        return entities
                    except json.JSONDecodeError:
                        try:
                            # 尝试直接解析整个结果
                            entities = json.loads(result)
                            if isinstance(entities, list):
                                return entities
                        except:
                            logger.warning(f"无法解析LLM返回的实体JSON: {result}")
                            return []
                else:
                    logger.warning("LLM未返回有效的实体JSON")
                    return []
            except Exception as e:
                logger.error(f"使用LLM提取实体失败: {str(e)}")
                return []
        else:
            logger.warning("未提供NER模型或LLM，无法提取实体")
            return []

        return extracted_entities
            
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
        # 提取实体
        entities = self.extract_entities(query)

        # 构建实体过滤器
        entity_ids = [entity["entity"] for entity in entities] if entities else []
        entity_filter = " OR ".join([f"entity.id:*{entity_id}*" for entity_id in entity_ids])
        cypher_query = f"""
        // 第一部分：向量搜索
        CALL db.index.vector.queryNodes($index_name, $embedding, $limit)
        YIELD node as doc, score as vector_score

        // 关联文档与实体
        OPTIONAL MATCH (doc)<-[:MENTIONED_IN|RELATED_TO]-(entity:__Entity__)
        WHERE doc.text IS NOT NULL
        {"AND " + entity_filter if entity_filter else ""}

        // 收集结果
        WITH doc, entity, vector_score,
             CASE WHEN entity IS NOT NULL THEN 1 ELSE 0 END as has_entity

        RETURN doc.id AS id,
               labels(doc) AS type,
               doc.text AS content,
               vector_score AS relevance,
               COLLECT(DISTINCT {{
                   id: entity.id,
                   type: CASE WHEN entity IS NOT NULL THEN head(labels(entity)) ELSE NULL END
               }}) AS related_entities,
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
            

