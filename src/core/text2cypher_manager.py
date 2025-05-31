"""
Text2Cypher索引管理器 - 管理Neo4j数据库中的Text2Cypher相关索引和约束
"""
import logging
from typing import Dict, Any, List, Optional

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)

class Text2CypherIndexManager:
    """Text2Cypher索引管理器 - 管理图数据库中的索引和约束"""
    
    def __init__(self, uri: str, user: str, password: str, database: str):
        """初始化索引管理器
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            database: 数据库名
        """
        self.uri = uri
        self.user = user
        self.password = password
        self.database = database
        
        # 初始化Neo4j驱动
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Neo4j索引管理器初始化成功 ({uri})")
    
    def optimize_indexes_for_text2cypher(self):
        """创建和优化用于text2cypher的全文索引"""
        # 全文索引创建查询
        fulltext_indexes = [
            # 内容全文索引
            """
            CREATE FULLTEXT INDEX content_fulltext_idx IF NOT EXISTS
            FOR (n:Document|Content|Entity|Concept|Paragraph|Section)
            ON EACH [n.content, n.title, n.text, n.name, n.description]
            OPTIONS {
                analyzer: 'standard',
                indexConfig: {
                    `fulltext.analyzer`: 'standard',
                    `fulltext.eventually_consistent`: true,
                    `fulltext.rebuild_on_startup`: false
                }
            }
            """,
            # 属性全文索引
            """
            CREATE FULLTEXT INDEX property_fulltext_idx IF NOT EXISTS
            FOR (n)
            ON EACH [n.name, n.title, n.description]
            OPTIONS {
                analyzer: 'standard'
            }
            """
        ]
        
        # 执行索引创建
        with self.driver.session(database=self.database) as session:
            for index_query in fulltext_indexes:
                try:
                    session.run(index_query)
                    logger.info(f"创建全文索引成功")
                except Exception as e:
                    logger.error(f"创建全文索引失败: {str(e)}")
    
    def create_cypher_transformation_indexes(self):
        """创建用于Cypher转换的索引"""
        # 节点和关系类型索引
        indexes = [
            # 节点标签索引
            """
            CREATE INDEX node_labels_idx IF NOT EXISTS
            FOR (n)
            ON n.__typename
            """,
            # 关系类型索引
            """
            CREATE INDEX relationship_types_idx IF NOT EXISTS
            FOR ()-[r]-()
            ON type(r)
            """
        ]
        
        # 执行索引创建
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                    logger.info(f"创建Cypher转换索引成功")
                except Exception as e:
                    logger.error(f"创建Cypher转换索引失败: {str(e)}")
    
    def create_text2cypher_schema_constraints(self):
        """创建Text2Cypher所需的模式约束"""
        # 唯一约束
        constraints = [
            # 实体唯一性约束
            """
            CREATE CONSTRAINT entity_id_uniqueness IF NOT EXISTS
            FOR (n:Entity)
            REQUIRE n.id IS UNIQUE
            """
        ]
        
        # 执行约束创建
        with self.driver.session(database=self.database) as session:
            for constraint_query in constraints:
                try:
                    session.run(constraint_query)
                    logger.info(f"创建模式约束成功")
                except Exception as e:
                    logger.error(f"创建模式约束失败: {str(e)}")
    
    def create_text2cypher_query_templates(self):
        """创建Text2Cypher查询模板"""
        # 在数据库中存储查询模板
        query_templates = [
            {
                "name": "content_search",
                "description": "搜索包含关键词的内容",
                "cypher": """
                CALL db.index.fulltext.queryNodes("content_fulltext_idx", $query, {limit: 5})
                YIELD node, score
                RETURN node.id as id, node.content as content, labels(node)[0] as type, 
                       score, node.title as title, node.created_at as created_at
                ORDER BY score DESC
                LIMIT 5
                """
            },
            {
                "name": "entity_retrieval",
                "description": "检索特定实体及其关系",
                "cypher": """
                MATCH (n)
                WHERE n.name CONTAINS $entity OR n.title CONTAINS $entity
                OPTIONAL MATCH (n)-[r]-(m)
                RETURN n, r, m
                LIMIT 10
                """
            }
        ]
        
        # 存储模板的Cypher查询
        store_template_query = """
        MERGE (t:QueryTemplate {name: $name})
        SET t.description = $description,
            t.cypher = $cypher,
            t.updated_at = datetime()
        RETURN t.name
        """
        
        # 执行模板存储
        with self.driver.session(database=self.database) as session:
            for template in query_templates:
                try:
                    session.run(
                        store_template_query,
                        name=template["name"],
                        description=template["description"],
                        cypher=template["cypher"]
                    )
                    logger.info(f"创建查询模板 '{template['name']}' 成功")
                except Exception as e:
                    logger.error(f"创建查询模板失败: {str(e)}")
    
    def optimize_and_repair_text_to_cypher_results(self, retrieval_results: Dict[str, Any]) -> Dict[str, Any]:
        """优化和修复text2cypher的检索结果
        
        Args:
            retrieval_results: 原始检索结果
            
        Returns:
            优化后的检索结果
        """
        # 如果没有merged_results，则初始化为空列表
        if "merged_results" not in retrieval_results:
            retrieval_results["merged_results"] = []
        
        # 确保每个merged_result项包含必要的字段
        for i, result in enumerate(retrieval_results.get("merged_results", [])):
            # 确保每个结果项都有id
            if "id" not in result:
                result["id"] = f"result_{i}"
            
            # 确保每个结果项都有content
            if "content" not in result:
                # 尝试从其他字段构建内容
                content_parts = []
                for field in ["text", "title", "description", "name"]:
                    if field in result and result[field]:
                        content_parts.append(str(result[field]))
                
                if content_parts:
                    result["content"] = "\n".join(content_parts)
                else:
                    result["content"] = f"Result {i+1}"
            
            # 确保每个结果都有metadata
            if "metadata" not in result:
                result["metadata"] = {}
            
            # 确保每个结果都有score
            if "score" not in result:
                result["score"] = 0.5  # 默认中等相关性
            
            # 确保每个结果都有sources
            if "sources" not in result:
                result["sources"] = ["graph"]
        
        return retrieval_results
    
    def close(self):
        """关闭连接"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j索引管理器已关闭")