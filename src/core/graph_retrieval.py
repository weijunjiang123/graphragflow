import json
import logging
from pathlib import Path
import sys
from typing import List, Dict, Any, Optional, Union, Tuple
from neo4j import GraphDatabase
from langchain_core.documents import Document
from langchain_community.vectorstores import Neo4jVector
import time

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

class QueryAnalysisResult:
    """查询分析结果类"""
    
    def __init__(self, 
                query: str,
                entities: List[Dict[str, str]] = None,
                key_concepts: List[str] = None,
                relations: List[str] = None,
                query_type: str = "general",
                context_constraints: Dict[str, Any] = None):
        """初始化查询分析结果
        
        Args:
            query: 原始查询文本
            entities: 提取的实体列表 [{"entity": "实体名称", "type": "实体类型"}]
            key_concepts: 关键概念列表
            relations: 可能的关系列表
            query_type: 查询类型 (general, entity_focused, relationship_focused, etc.)
            context_constraints: 上下文约束条件
        """
        self.query = query
        self.entities = entities or []
        self.key_concepts = key_concepts or []
        self.relations = relations or []
        self.query_type = query_type
        self.context_constraints = context_constraints or {}
        
    def get_entity_names(self) -> List[str]:
        """获取所有实体名称"""
        return [entity["entity"] for entity in self.entities]
    
    def get_entities_by_type(self, entity_type: str) -> List[Dict[str, str]]:
        """按类型获取实体"""
        return [entity for entity in self.entities if entity["type"].lower() == entity_type.lower()]
    
    def to_dict(self) -> Dict[str, Any]:
        """将查询分析结果转换为字典"""
        return {
            "query": self.query,
            "entities": self.entities,
            "key_concepts": self.key_concepts,
            "relations": self.relations,
            "query_type": self.query_type,
            "context_constraints": self.context_constraints
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"查询分析结果:\n"
                f"- 查询: {self.query}\n"
                f"- 实体: {', '.join(self.get_entity_names())}\n" 
                f"- 关键概念: {', '.join(self.key_concepts)}\n"
                f"- 查询类型: {self.query_type}")

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

    def analyze_query(self, query: str) -> QueryAnalysisResult:
        """对查询进行分析，提取实体、关键概念和识别查询类型
        
        Args:
            query: 用户查询
            
        Returns:
            查询分析结果对象
        """
        logger.info(f"开始分析查询: {query}")
        
        # 提取实体
        entities = self.extract_entities(query)
        
        # 提取关键概念和识别查询类型
        key_concepts = []
        relations = []
        query_type = "general"
        context_constraints = {}
        
        # 如果有LLM，使用它进行更深入的查询分析
        if self.llm:
            try:
                analysis_prompt = f"""
                分析以下用户查询，提取关键信息:

                查询: "{query}"

                请返回以下JSON格式:
                {{
                    "key_concepts": ["概念1", "概念2"...],
                    "relations": ["可能的关系1", "可能的关系2"...],
                    "query_type": "查询类型",
                    "context_constraints": {{"时间范围": "...", "数量限制": "..."}}
                }}

                查询类型可以是:
                - "general": 一般信息查询
                - "entity_focused": 主要关注特定实体
                - "relationship_focused": 主要关注实体间关系
                - "comparison": 比较多个实体
                - "temporal": 涉及时间序列或历史查询
                - "causal": 探究因果关系
                
                仅返回JSON，不要包含其他说明。
                """
                
                result = self.llm.invoke(analysis_prompt)
                
                # 处理AIMessage类型的返回值
                if hasattr(result, 'content'):
                    # 如果是AIMessage对象，获取其content属性
                    result_text = result.content
                elif isinstance(result, str):
                    # 如果直接是字符串
                    result_text = result
                else:
                    # 尝试转换为字符串
                    result_text = str(result)
                
                # 增强型JSON解析
                try:
                    # 先尝试直接解析整个响应
                    try:
                        analysis_result = json.loads(result_text)
                        
                        # 提取分析结果
                        key_concepts = analysis_result.get("key_concepts", [])
                        relations = analysis_result.get("relations", [])
                        query_type = analysis_result.get("query_type", "general")
                        context_constraints = analysis_result.get("context_constraints", {})
                        
                        logger.info(f"查询分析结果: 类型={query_type}, 概念={key_concepts}")
                    except json.JSONDecodeError:
                        # 尝试找到并解析JSON部分
                        import re
                        json_pattern = r'\{.*\}'
                        match = re.search(json_pattern, result_text, re.DOTALL)
                        
                        if match:
                            json_str = match.group(0)
                            analysis_result = json.loads(json_str)
                            
                            # 提取分析结果
                            key_concepts = analysis_result.get("key_concepts", [])
                            relations = analysis_result.get("relations", [])
                            query_type = analysis_result.get("query_type", "general")
                            context_constraints = analysis_result.get("context_constraints", {})
                            
                            logger.info(f"查询分析结果: 类型={query_type}, 概念={key_concepts}")
                        else:
                            logger.warning("无法从LLM响应中提取JSON")
                except Exception as e:
                    logger.error(f"解析查询分析结果时出错: {str(e)}")
            except Exception as e:
                logger.error(f"使用LLM分析查询时出错: {str(e)}")
        
        # 构建查询分析结果
        return QueryAnalysisResult(
            query=query,
            entities=entities,
            key_concepts=key_concepts,
            relations=relations,
            query_type=query_type,
            context_constraints=context_constraints
        )

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
                # 调用LLM获取结果
                result = self.llm.invoke(prompt)
                
                # 处理AIMessage类型的返回值
                if hasattr(result, 'content'):
                    # 如果是AIMessage对象，获取其content属性
                    result_text = result.content
                elif isinstance(result, str):
                    # 如果直接是字符串
                    result_text = result
                else:
                    # 尝试转换为字符串
                    result_text = str(result)
                
                # 处理LLM返回的JSON
                import re
                json_pattern = r'\[(.*?)\]'
                match = re.search(json_pattern, result_text, re.DOTALL)

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
                            entities = json.loads(result_text)
                            if isinstance(entities, list):
                                return entities
                        except:
                            logger.warning(f"无法解析LLM返回的实体JSON: {result_text}")
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
    
    def find_entities_by_name(self, entity_name: str, fuzzy_match: bool = True, limit: int = 5) -> List[Dict[str, Any]]:
        """根据名称查找实体(精确或模糊匹配)
        
        Args:
            entity_name: 实体名称
            fuzzy_match: 是否进行模糊匹配
            limit: 结果数量限制
            
        Returns:
            匹配的实体列表
        """
        if fuzzy_match:
            # 使用全文索引进行模糊匹配
            cypher_query = """
            CALL db.index.fulltext.queryNodes("fulltext_entity_id", $entity_name, {limit: $limit}) 
            YIELD node, score
            WHERE node:__Entity__
            RETURN node.id AS id, 
                   labels(node) AS labels, 
                   score, 
                   properties(node) AS properties
            ORDER BY score DESC
            """
        else:
            # 使用精确匹配
            cypher_query = """
            MATCH (node:__Entity__)
            WHERE node.id = $entity_name OR node.name = $entity_name
            RETURN node.id AS id, 
                   labels(node) AS labels, 
                   1.0 AS score, 
                   properties(node) AS properties
            LIMIT $limit
            """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, entity_name=entity_name, limit=limit)
                entities = [record.data() for record in result]
                logger.info(f"找到 {len(entities)} 个匹配实体")
                return entities
            except Exception as e:
                logger.error(f"查找实体失败: {str(e)}")
                return []
    
    def get_entity_neighbors(self, entity_id: str, hop: int = 1, limit: int = 10) -> Dict[str, Any]:
        """获取实体的N跳邻居
        
        Args:
            entity_id: 实体ID
            hop: 跳数(1-3)
            limit: 每种关系的结果限制
            
        Returns:
            包含邻居实体和关系的字典
        """
        if hop < 1 or hop > 3:
            logger.warning(f"跳数 {hop} 超出范围(1-3)，将使用默认值 1")
            hop = 1
            
        # 修正Cypher查询，避免使用已弃用的id()函数
        cypher_query = f"""
        MATCH path = (source:__Entity__ {{id: $entity_id}})-[r*1..{hop}]-(neighbor)
        WHERE neighbor:__Entity__ AND source <> neighbor
        WITH source, neighbor, [rel in relationships(path) | rel] AS rels, path
        RETURN neighbor.id AS id,
               labels(neighbor) AS labels,
               properties(neighbor) AS properties,
               [rel in rels | type(rel)] AS relationship_types,
               [rel in rels | properties(rel)] AS relationship_properties,
               length(path) AS distance
        ORDER BY distance ASC, id ASC
        LIMIT $limit
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, entity_id=entity_id, limit=limit)
                neighbors = [record.data() for record in result]
                
                # 组织结果
                return {
                    "seed_entity": entity_id,
                    "neighbor_count": len(neighbors),
                    "neighbors": neighbors
                }
            except Exception as e:
                logger.error(f"获取实体邻居失败: {str(e)}")
                return {
                    "seed_entity": entity_id,
                    "neighbor_count": 0,
                    "neighbors": []
                }
    
    def find_shortest_path(self, source_entity_id: str, target_entity_id: str, 
                          max_depth: int = 4, relation_types: List[str] = None) -> Dict[str, Any]:
        """查找两个实体之间的最短路径
        
        Args:
            source_entity_id: 起始实体ID
            target_entity_id: 目标实体ID
            max_depth: 最大路径长度
            relation_types: 要考虑的关系类型列表
            
        Returns:
            包含路径的字典
        """
        # 构建关系类型过滤条件
        rel_filter = ""
        if relation_types:
            rel_types = "|".join([f":{r}" for r in relation_types])
            rel_filter = f"[{rel_types}]"
        
        # 修正Cypher查询，使用正确的路径语法
        cypher_query = f"""
        MATCH path = shortestPath((source:__Entity__ {{id: $source_id}})-[{rel_filter}*1..{max_depth}]-(target:__Entity__ {{id: $target_id}}))
        WITH source, target, path, relationships(path) as rels, nodes(path) as nodes
        RETURN 
            source.id AS source_id,
            target.id AS target_id,
            length(path) AS path_length,
            [node IN nodes | {{
                id: node.id, 
                labels: labels(node),
                properties: properties(node)
            }}] AS path_nodes,
            [rel IN rels | {{
                type: type(rel),
                properties: properties(rel),
                source: startNode(rel).id,
                target: endNode(rel).id
            }}] AS path_relationships
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(
                    cypher_query, 
                    source_id=source_entity_id, 
                    target_id=target_entity_id
                ).single()
                
                if result:
                    return result.data()
                else:
                    logger.info(f"未找到从 {source_entity_id} 到 {target_entity_id} 的路径")
                    return {
                        "source_id": source_entity_id,
                        "target_id": target_entity_id,
                        "path_length": 0,
                        "path_nodes": [],
                        "path_relationships": []
                    }
            except Exception as e:
                logger.error(f"查找最短路径失败: {str(e)}")
                return {
                    "source_id": source_entity_id,
                    "target_id": target_entity_id,
                    "path_length": 0,
                    "path_nodes": [],
                    "path_relationships": [],
                    "error": str(e)
                }
    
    def pattern_match(self, pattern_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行模式匹配查询
        
        Args:
            pattern_query: Cypher查询或预定义模式名称
            params: 查询参数
            
        Returns:
            匹配结果列表
        """
        predefined_patterns = {
            "author_publication": """
                MATCH (author:__Entity__)-[:AUTHORED]->(publication:__Entity__)
                WHERE author.id = $author_id
                RETURN author.id AS author_id, 
                       publication.id AS publication_id,
                       publication.title AS title,
                       publication.year AS year
                ORDER BY publication.year DESC
                LIMIT $limit
            """,
            "topic_related": """
                MATCH (entity:__Entity__ {id: $entity_id})-[:RELATED_TO]-(related:__Entity__)
                RETURN entity.id AS source_id,
                       related.id AS related_id,
                       labels(related)[0] AS related_type,
                       properties(related) AS properties
                LIMIT $limit
            """,
            "citation_network": """
                MATCH (source:__Entity__)-[:CITES]->(cited:__Entity__)
                WHERE source.id = $publication_id
                RETURN source.id AS source_id,
                       cited.id AS cited_id,
                       cited.title AS cited_title,
                       cited.year AS cited_year
                LIMIT $limit
            """
        }
        
        # 确定要使用的查询
        if pattern_query in predefined_patterns:
            cypher_query = predefined_patterns[pattern_query]
        else:
            # 用户提供的自定义查询
            cypher_query = pattern_query
            
        # 确保参数字典存在
        if params is None:
            params = {}
            
        # 添加默认参数
        if "limit" not in params:
            params["limit"] = 10
            
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, **params)
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"模式匹配查询失败: {str(e)}")
                return []
                
    def generate_cypher_query(self, query_description: str, schema_info: str = None) -> Tuple[str, Dict[str, Any]]:
        """使用LLM生成Cypher查询语句和推断的参数
        
        Args:
            query_description: 自然语言查询描述
            schema_info: 数据库模式信息(可选)
            
        Returns:
            Tuple containing the generated Cypher query string and a dictionary of inferred parameters.
            Returns ("", {}) if generation fails.
        """
        if not self.llm:
            logger.error("无法生成Cypher查询: 未提供LLM")
            return "", {}
            
        # 如果未提供模式信息，尝试获取
        if not schema_info:
            try:
                # 获取更全面的模式信息，包括节点标签、关系类型、常见属性和示例数据
                with self.driver.session(database=self.database) as session:
                    # 获取所有节点标签
                    labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
                    labels = labels_result["labels"] if labels_result else []
                    
                    # 获取所有关系类型
                    rel_types_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types").single()
                    rel_types = rel_types_result["types"] if rel_types_result else []
                    
                    # 获取主要节点标签的属性和示例数据
                    schema_details = []
                    common_labels = [l for l in labels if l not in ["Document"]] 
                    selected_labels = common_labels[:5]  # 限制处理的标签数量
                    
                    for label in selected_labels:
                        # 获取标签的常见属性
                        props_query = f"""
                        MATCH (n:{label}) 
                        WITH n LIMIT 1
                        RETURN keys(n) AS properties
                        """
                        props_result = session.run(props_query).single()
                        if props_result:
                            properties = props_result["properties"]
                            property_str = ", ".join([f"`{p}`" for p in properties if p != "embedding"])
                            
                            # 获取节点示例
                            example_query = f"""
                            MATCH (n:{label})
                            RETURN n LIMIT 1
                            """
                            try:
                                example_result = session.run(example_query).single()
                                if example_result:
                                    node = example_result["n"]
                                    example_data = {k: v for k, v in dict(node._properties).items() 
                                                 if k != "embedding" and not isinstance(v, (list, dict))}
                                    schema_details.append(f"节点 `:{label}` 属性: {property_str}")
                                    schema_details.append(f"示例: {json.dumps(example_data, ensure_ascii=False)[:150]}")
                            except:
                                # 如果获取示例失败，只添加属性信息
                                schema_details.append(f"节点 `:{label}` 属性: {property_str}")
                    
                    # 获取关系的属性信息
                    for rel_type in rel_types[:3]:  # 限制处理的关系类型数量
                        rel_query = f"""
                        MATCH ()-[r:{rel_type}]->() 
                        WITH r LIMIT 1
                        RETURN keys(r) AS properties,
                               labels(startNode(r)) AS start_labels,
                               labels(endNode(r)) AS end_labels
                        """
                        try:
                            rel_result = session.run(rel_query).single()
                            if rel_result:
                                rel_props = rel_result["properties"]
                                start_labels = rel_result["start_labels"]
                                end_labels = rel_result["end_labels"]
                                
                                rel_prop_str = ", ".join([f"`{p}`" for p in rel_props]) if rel_props else "无属性"
                                schema_details.append(
                                    f"关系 `:{rel_type}` 从 `:{start_labels[0] if start_labels else '?'}` 到 `:{end_labels[0] if end_labels else '?'}`，属性: {rel_prop_str}"
                                )
                        except:
                            # 如果关系查询失败，添加简单信息
                            schema_details.append(f"关系类型: `:{rel_type}`")
                    
                    # 获取数据库大小和统计信息
                    try:
                        count_query = """
                        MATCH (n)
                        RETURN count(n) AS node_count LIMIT 1
                        """
                        count_result = session.run(count_query).single()
                        if count_result:
                            node_count = count_result["node_count"]
                            schema_details.append(f"数据库包含约 {node_count} 个节点")
                    except:
                        pass
                        
                    # 构建完整模式信息
                    all_labels = "节点标签: " + ", ".join([f"`:{l}`" for l in labels])
                    all_rels = "关系类型: " + ", ".join([f"`:{r}`" for r in rel_types])
                    schema_details_str = "\n".join(schema_details)
                    
                    schema_info = f"{all_labels}\n{all_rels}\n\n详细信息:\n{schema_details_str}"
            except Exception as e:
                logger.warning(f"获取数据库模式详情失败: {str(e)}")
                # 降级到基本模式信息
                try:
                    with self.driver.session(database=self.database) as session:
                        labels_result = session.run("CALL db.labels() YIELD label RETURN collect(label) AS labels").single()
                        labels = labels_result["labels"] if labels_result else []
                        
                        rel_types_result = session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS types").single()
                        rel_types = rel_types_result["types"] if rel_types_result else []
                        
                        schema_info = f"节点标签: {', '.join(labels)}\n关系类型: {', '.join(rel_types)}"
                except:
                    schema_info = "未能获取数据库模式信息"
        
        # 优化Cypher生成提示
        prompt = f"""
        你是一位Neo4j Cypher专家，现在需要将自然语言问题转换为精确的Cypher查询。

        数据库模式:
        ```
        {schema_info}
        ```

        用户问题: "{query_description}"

        请遵循以下规则生成Cypher查询:
        1. 生成匹配用户问题的最优Cypher查询
        2. 查询必须使用参数化形式，如 `$paramName` 而非硬编码值
        3. 查询应当对应模式信息中的标签和关系类型
        4. 使用模式中已存在的属性名称
        5. 实体节点通常使用 `__Entity__` 标签，通过 `id` 或 `name` 属性标识
        6. 返回有意义的结果字段，必要时包含属性信息
        7. 包含 LIMIT 子句限制结果数量（通常为10-20）
        8. 必要时使用模式推导，智能匹配标签和关系
        9. 写出尽可能高效的查询

        回复格式: 仅返回一个JSON对象，包含两个字段：
        - "query": 生成的Cypher查询字符串
        - "params": 包含查询参数的对象

        示例:
        ```json
        {{
          "query": "MATCH (p:Person {{name: $person_name}})-[:ACTED_IN]->(m:Movie) RETURN m.title AS movie, m.year AS year ORDER BY m.year DESC LIMIT $limit",
          "params": {{ "person_name": "Tom Hanks", "limit": 10 }}
        }}
        ```
        """
        
        try:
            # 调用LLM生成Cypher查询
            result = self.llm.invoke(prompt)
            
            # 处理返回结果
            if hasattr(result, 'content'):
                result_text = result.content
            elif isinstance(result, str):
                result_text = result
            else:
                result_text = str(result)

            # 增强型JSON解析
            cypher_query = ""
            params = {}
            
            # 尝试提取JSON
            import re
            json_pattern = r'\{(?:[^{}]|(?:\{[^{}]*\}))*\}'
            match = re.search(json_pattern, result_text, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                try:
                    parsed_json = json.loads(json_str)
                    cypher_query = parsed_json.get("query", "").strip()
                    params = parsed_json.get("params", {})
                    
                    # 验证查询和参数
                    if cypher_query and isinstance(params, dict):
                        # 净化查询（删除注释、额外引号等）
                        cypher_query = re.sub(r'\s+', ' ', cypher_query)  # 规范化空白
                        cypher_query = re.sub(r'//.*?$', '', cypher_query, flags=re.MULTILINE)  # 移除单行注释
                        cypher_query = re.sub(r'/\*.*?\*/', '', cypher_query, flags=re.DOTALL)  # 移除块注释
                        cypher_query = cypher_query.strip().rstrip(';')  # 删除尾部分号和空白
                        
                        # 验证参数在查询中被引用
                        for param_name in params.keys():
                            if f"${param_name}" not in cypher_query and not re.search(rf"\$`?{param_name}`?", cypher_query):
                                logger.warning(f"参数 '{param_name}' 在查询中未被引用，但仍将传递")
                        
                        logger.info(f"成功生成Cypher查询: {cypher_query}")
                        logger.info(f"查询参数: {params}")
                        return cypher_query, params
                    else:
                        logger.warning("解析出的JSON不包含有效的查询或参数")
                except json.JSONDecodeError as e:
                    logger.error(f"解析JSON失败: {e}")
            
            # 备用方法：直接提取Cypher查询
            cypher_pattern = r'(?:MATCH|CALL|RETURN|WITH|MERGE|CREATE|UNWIND)\s+.*?(?:RETURN|LIMIT).*?[^\s;]'
            cypher_matches = re.findall(cypher_pattern, result_text, re.IGNORECASE | re.DOTALL)
            
            if cypher_matches:
                cypher_query = cypher_matches[0].strip()
                # 清理查询
                cypher_query = re.sub(r'^```(?:cypher)?\s*|\s*```$', '', cypher_query, flags=re.MULTILINE)
                cypher_query = cypher_query.strip().rstrip(';')
                
                # 提取可能的参数
                param_pattern = r'\$([a-zA-Z0-9_]+)'
                param_matches = re.findall(param_pattern, cypher_query)
                
                for param in param_matches:
                    # 尝试提取参数值，使用占位符
                    if param.lower() == 'limit' or param.endswith('_limit'):
                        params[param] = 10
                    elif param.lower() == 'skip' or param.endswith('_skip'):
                        params[param] = 0
                    else:
                        # 为其他参数使用原始查询中的文本作为默认值
                        value_match = re.search(rf"{param}\s*[:=]\s*[\"']?([^\"',}}]+)[\"']?", result_text)
                        if value_match:
                            params[param] = value_match.group(1).strip()
                        else:
                            # 使用问题中可能的关键词
                            words = re.findall(r'\b\w+\b', query_description)
                            for word in words:
                                if len(word) > 3 and param.lower().find(word.lower()) >= 0:
                                    params[param] = word
                                    break
                            else:
                                params[param] = f"value_for_{param}"
                
                logger.warning(f"从非JSON结构中提取到Cypher查询: {cypher_query}")
                logger.warning(f"推断的参数: {params}")
                return cypher_query, params
            
            logger.error(f"无法从LLM响应中提取有效的Cypher查询: {result_text[:200]}...")
            return "", {}

        except Exception as e:
            logger.error(f"生成Cypher查询时发生错误: {str(e)}")
            return "", {}

    def execute_cypher(self, cypher_query: str, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行给定的 Cypher 查询"""
        if not cypher_query:
            return []
        if params is None:
            params = {}
            
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, **params)
                # Convert Neo4j Records to dictionaries, handling potential complex types
                data = []
                for record in result:
                    record_data = {}
                    for key, value in record.items():
                        if hasattr(value, '_properties'): # Handle Node/Relationship types
                            record_data[key] = dict(value._properties)
                            # Optionally add labels for nodes
                            if hasattr(value, 'labels'):
                                record_data[key]['labels'] = list(value.labels)
                        elif isinstance(value, list) and value and hasattr(value[0], '_properties'):
                             # Handle lists of Nodes/Relationships
                             record_data[key] = [dict(item._properties) for item in value]
                        else:
                            record_data[key] = value
                    data.append(record_data)

                logger.info(f"成功执行 Cypher 查询，返回 {len(data)} 条记录")
                return data
            except Exception as e:
                logger.error(f"执行 Cypher 查询失败: {str(e)}\n查询: {cypher_query}\n参数: {params}")
                return []

    def vector_search(self, query: str, limit: int = 5) -> List[Document]:
        """执行向量搜索
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            
        Returns:
            匹配的文档列表
        """
        if not self.vector_store:
            logger.warning("向量检索失败: 向量存储未初始化")
            return []
            
        try:
            logger.info(f"执行向量搜索: {query}")
            start_time = time.time()
            
            # 使用向量存储的检索器搜索
            retriever = self.vector_store.as_retriever(search_kwargs={"k": limit})
            docs = retriever.get_relevant_documents(query)
            
            elapsed_time = time.time() - start_time
            logger.info(f"向量搜索完成，耗时: {elapsed_time:.2f}秒，找到 {len(docs)} 条结果")
            
            return docs
        except Exception as e:
            logger.error(f"向量搜索出错: {str(e)}")
            return []

    def enhanced_hybrid_search(self, 
                             query: str, 
                             limit: int = 5, 
                             graph_weight: float = 0.5, 
                             vector_weight: float = 0.5,
                             context_entities: List[str] = None,
                             use_text2cypher: bool = True,
                             force_cypher: str = None,
                             cypher_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """增强混合搜索，融合图检索(包括Text2Cypher)与向量检索
        
        Args:
            query: 搜索查询
            limit: 结果数量限制
            graph_weight: 图检索结果权重
            vector_weight: 向量检索结果权重
            context_entities: 上下文相关实体列表
            use_text2cypher: 是否尝试使用 LLM 生成和执行 Cypher 查询
            force_cypher: 可选的强制执行的Cypher查询（优先于自动生成）
            cypher_params: 与force_cypher配合使用的查询参数
            
        Returns:
            混合搜索结果
        """
        start_time = time.time()
        
        # 分析查询
        query_analysis = self.analyze_query(query)
        logger.info(f"查询分析结果: {query_analysis}")
        
        # 提取实体并添加上下文实体
        entities = query_analysis.get_entity_names()
        if context_entities:
            entities.extend(context_entities)
        entities = list(set(entities)) # 去除重复
        
        # --- 图检索部分 ---
        graph_results = []
        executed_cypher_query = ""
        executed_cypher_params = {}

        # 1. 如果提供了强制执行的Cypher查询，优先使用它
        if force_cypher:
            logger.info(f"使用强制指定的Cypher查询: {force_cypher}")
            executed_cypher_query = force_cypher
            executed_cypher_params = cypher_params or {}
            cypher_execution_results = self.execute_cypher(force_cypher, executed_cypher_params)
            
            # 将Cypher执行结果转换为统一格式
            for record in cypher_execution_results:
                # 尝试提取主要节点或有意义的内容
                content = json.dumps(record, ensure_ascii=False, default=str)
                node_id = None
                labels = []
                
                # 启发式方法：查找记录中第一个像节点的字典
                for key, value in record.items():
                    if isinstance(value, dict) and ('id' in value or 'name' in value):
                        node_id = value.get('id') or value.get('name')
                        labels = value.get('labels', [])
                        content = self._extract_entity_content(value)
                        break
                
                # 如果未找到节点ID，使用备用ID
                if not node_id:
                    node_id = f"cypher_result_{hash(content)}"

                # 添加到图结果中
                graph_results.append({
                    "id": node_id,
                    "content": content,
                    "metadata": {
                        "source": "custom_cypher",
                        "labels": labels,
                        "raw_result": record
                    },
                    "score": 0.95,  # 为自定义查询赋予较高权重
                    "source": "graph"
                })
            
            logger.info(f"自定义Cypher查询返回 {len(cypher_execution_results)} 条结果")
        
        # 2. 如果未指定强制查询且启用Text2Cypher，尝试LLM生成查询
        elif use_text2cypher and self.llm:
            logger.info("尝试使用 Text2Cypher 生成查询...")
            generated_cypher, generated_params = self.generate_cypher_query(query)
            if generated_cypher:
                executed_cypher_query = generated_cypher
                executed_cypher_params = generated_params
                logger.info(f"执行生成的 Cypher 查询: {generated_cypher} with params: {generated_params}")
                cypher_execution_results = self.execute_cypher(generated_cypher, generated_params)
                
                # 将 Cypher 执行结果转换为 graph_results 格式
                for record in cypher_execution_results:
                    # 尝试提取节点或有意义的内容
                    content = json.dumps(record, ensure_ascii=False, default=str)
                    node_id = None
                    labels = []
                    
                    # 查找类节点的字典
                    for key, value in record.items():
                        if isinstance(value, dict) and ('id' in value or 'name' in value):
                            node_id = value.get('id') or value.get('name')
                            labels = value.get('labels', [])
                            content = self._extract_entity_content(value)
                            break
                            
                    # 如果未找到节点ID，使用备用ID
                    if not node_id:
                        node_id = f"cypher_result_{hash(content)}"

                    graph_results.append({
                        "id": node_id,
                        "content": content,
                        "metadata": {
                            "source": "text2cypher",
                            "labels": labels,
                            "raw_result": record
                        },
                        "score": 0.9, # 为自动生成查询结果赋予较高权重
                        "source": "graph"
                    })
                    
                logger.info(f"Text2Cypher 返回 {len(cypher_execution_results)} 条结果")

        # 3. 如果以上方法都未产生结果且有实体，执行基于实体的邻居搜索
        if not graph_results and entities:
            logger.info("未通过Cypher产生结果，执行基于实体的邻居搜索...")
            # 对查询中的每个实体执行图检索
            for entity_name in entities:
                entity_matches = self.find_entities_by_name(entity_name, fuzzy_match=True, limit=2)
                
                # 如果找到匹配实体，获取它的邻居信息
                for match in entity_matches:
                    entity_id = match.get("id")
                    if entity_id:
                        # 获取邻居（限制跳数和数量以保持结果可管理）
                        neighbors_data = self.get_entity_neighbors(entity_id, hop=1, limit=3) 
                        if neighbors_data and "neighbors" in neighbors_data:
                            # 将邻居数据转换为统一格式
                            for neighbor in neighbors_data["neighbors"]:
                                graph_results.append({
                                    "id": neighbor.get("id"),
                                    "content": self._extract_entity_content(neighbor),
                                    "metadata": {
                                        "source": "neighbor_search",
                                        "seed_entity": entity_id,
                                        "labels": neighbor.get("labels", []),
                                        "relationship_types": neighbor.get("relationship_types", []),
                                        "distance": neighbor.get("distance", 1)
                                    },
                                    "score": 1.0 - (neighbor.get("distance", 1) * 0.2), # 按距离计算分数
                                    "source": "graph"
                                })
        
        # --- 向量检索部分 ---
        vector_results = []
        if self.vector_store and vector_weight > 0: # 仅在需要时运行
            logger.info("执行向量搜索...")
            vector_docs = self.vector_search(query, limit=limit)
            
            # 将向量结果转换为统一格式
            for doc in vector_docs:
                # 安全获取分数，如果缺失则使用默认值
                score = getattr(doc, "metadata", {}).get("score", 0.7)
                vector_results.append({
                    "id": getattr(doc, "metadata", {}).get("id", f"doc_{hash(doc.page_content)}"),
                    "content": doc.page_content,
                    "metadata": getattr(doc, "metadata", {}),
                    "score": score, 
                    "source": "vector"
                })
                
            logger.info(f"向量搜索返回 {len(vector_results)} 条结果")
        
        # --- 结果融合 ---
        logger.info("开始融合搜索结果...")
        merged_results = self._merge_search_results(
            graph_results, 
            vector_results, 
            graph_weight=graph_weight,
            vector_weight=vector_weight,
            limit=limit
        )
        
        # --- 构建上下文 ---
        logger.info("组织检索上下文...")
        organized_context = self._organize_retrieval_context(
            query=query,
            query_analysis=query_analysis,
            merged_results=merged_results,
            entities=entities,
            executed_cypher=executed_cypher_query
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"混合搜索完成，耗时 {elapsed_time:.2f} 秒. 返回 {len(merged_results)} 条融合结果。")
        
        return {
            "query": query,
            "entities": entities,
            "graph_results_count": len(graph_results),
            "vector_results_count": len(vector_results),
            "merged_results_count": len(merged_results),
            "elapsed_time": elapsed_time,
            "organized_context": organized_context,
            "executed_cypher_query": executed_cypher_query,
            "executed_cypher_params": executed_cypher_params,
            "merged_results": merged_results  # 返回融合的结果列表以便更灵活使用
        }
        
    def _merge_search_results(self, 
                             graph_results: List[Dict[str, Any]], 
                             vector_results: List[Dict[str, Any]],
                             graph_weight: float = 0.5,
                             vector_weight: float = 0.5,
                             limit: int = 5) -> List[Dict[str, Any]]:
        """融合图检索和向量检索结果 (已更新以处理不同来源)
        
        Args:
            graph_results: 图检索结果 (来自 Text2Cypher 或邻居搜索)
            vector_results: 向量检索结果
            graph_weight: 图结果权重
            vector_weight: 向量结果权重
            limit: 结果数量限制
            
        Returns:
            融合排序后的结果列表
        """
        # 标准化权重
        total_weight = graph_weight + vector_weight
        if total_weight <= 0: # Avoid division by zero
             graph_weight = 0.5
             vector_weight = 0.5
        else:
            graph_weight = graph_weight / total_weight
            vector_weight = vector_weight / total_weight
        
        all_results_dict = {} # Use dict for easier deduplication and score update

        # 处理图结果 (Text2Cypher or Neighbors)
        for item in graph_results:
            item_id = item.get("id")
            if not item_id: continue # Skip items without ID

            # Calculate initial score based on source if not already present
            if 'score' not in item:
                 item['score'] = 0.9 if item.get('metadata',{}).get('source') == 'text2cypher' else 0.7 # Default scores

            final_score = item['score'] * graph_weight
            
            if item_id in all_results_dict:
                # Update existing entry if current score is higher
                if final_score > all_results_dict[item_id].get("final_score", 0):
                    all_results_dict[item_id]["final_score"] = final_score
                    # Merge metadata, prioritize graph source info
                    all_results_dict[item_id]["metadata"].update(item.get("metadata", {})) 
                    all_results_dict[item_id]["content"] = item.get("content") # Update content too
            else:
                item["final_score"] = final_score
                item["sources"] = [item.get("metadata", {}).get("source", "graph")] # Track sources
                all_results_dict[item_id] = item

        # 处理向量结果
        for item in vector_results:
            item_id = item.get("id")
            if not item_id: continue

            final_score = item.get("score", 0.7) * vector_weight # Use default score if missing

            if item_id in all_results_dict:
                # Found potential duplicate from graph search
                # Add vector score component and update sources
                # Option 1: Add scores (might inflate scores)
                # all_results_dict[item_id]["final_score"] += final_score 
                # Option 2: Average or weighted average (more complex)
                # Option 3: Take max (simple, prioritizes best single source score * weight)
                all_results_dict[item_id]["final_score"] = max(all_results_dict[item_id].get("final_score", 0), final_score)
                
                if "vector" not in all_results_dict[item_id].get("sources", []):
                     all_results_dict[item_id].setdefault("sources", []).append("vector")
                # Optionally merge metadata (e.g., keep vector metadata if needed)
                # all_results_dict[item_id]["metadata"].update(item.get("metadata", {})) 
            else:
                # New item from vector search
                item["final_score"] = final_score
                item["sources"] = ["vector"]
                all_results_dict[item_id] = item
        
        # Convert back to list and sort
        all_results = list(all_results_dict.values())
        sorted_results = sorted(all_results, key=lambda x: x.get("final_score", 0), reverse=True)
        
        logger.info(f"融合后得到 {len(sorted_results)} 条结果，将返回前 {limit} 条")
        return sorted_results[:limit]
        
    def _extract_entity_content(self, entity_data: Dict[str, Any]) -> str:
        """从实体数据中提取可读内容 (已更新以处理不同格式)
        
        Args:
            entity_data: 实体数据字典 (可能来自邻居搜索或直接 Cypher 结果)
            
        Returns:
            格式化的实体内容字符串
        """
        content_parts = []
        
        # Try to get ID and Label consistently
        entity_id = entity_data.get("id") or entity_data.get("name") or "未知ID"
        labels = entity_data.get("labels", [])
        label = labels[0] if labels and isinstance(labels, list) else entity_data.get("label", "Entity") # Handle direct 'label' key
        
        content_parts.append(f"{label}: {entity_id}")
        
        # Add properties, excluding common/internal ones
        properties = entity_data.get("properties", entity_data) # Use entity_data itself if 'properties' key is missing
        if isinstance(properties, dict):
            prop_strs = []
            for key, value in properties.items():
                # Exclude known internal/metadata keys and embeddings
                if key not in ["id", "embedding", "labels", "source", "score", "final_score", "sources", "raw_result", "relationship_types", "distance", "seed_entity"]:
                    prop_strs.append(f"{key}: {value}")
            if prop_strs:
                 content_parts.append("属性: " + ", ".join(prop_strs))

        # Add relationship info if available (from neighbor search)
        rel_types = entity_data.get("relationship_types", [])
        if rel_types:
            rel_str = "关系: " + ", ".join(rel_types)
            content_parts.append(rel_str)
        
        # Add distance if available (from neighbor search)
        distance = entity_data.get("distance")
        if distance is not None:
             content_parts.append(f"距离: {distance}")

        return "\n".join(content_parts)
        
    def _organize_retrieval_context(self,
                                  query: str,
                                  query_analysis: QueryAnalysisResult,
                                  merged_results: List[Dict[str, Any]],
                                  entities: List[str],
                                  executed_cypher: str = "") -> Dict[str, Any]: # Added executed_cypher
        """组织检索结果为结构化上下文 (已更新)
        
        Args:
            query: 原始查询
            query_analysis: 查询分析结果
            merged_results: 融合后的搜索结果
            entities: 相关实体列表
            executed_cypher: The Cypher query executed by Text2Cypher (if any)
            
        Returns:
            结构化上下文字典
        """
        # Initialize context structure
        context = {
            "query_information": {
                "original_query": query,
                "identified_entities": entities,
                "query_type": query_analysis.query_type,
                "key_concepts": query_analysis.key_concepts
            },
            "graph_knowledge": {
                "retrieved_entities": [], # Renamed for clarity
                "relationships": [], # Note: Relationships might be less explicit with direct Cypher
                "paths": [],
                "executed_cypher_query": executed_cypher # Store the executed query
            },
            "text_context": {
                "relevant_passages": []
            },
            "combined_context": "",
            # "suggested_cypher_query": "" # Removed, as we now execute directly
        }
        
        processed_node_ids = set()

        # Process merged results
        for item in merged_results:
            source_types = item.get("sources", [])
            
            # Process items originating from graph (Text2Cypher or Neighbors)
            if any(s in ["graph", "text2cypher", "neighbor_search"] for s in source_types):
                entity_id = item.get("id")
                if entity_id and entity_id not in processed_node_ids:
                    entity_info = {
                        "id": entity_id,
                        "label": item.get("metadata", {}).get("labels", ["Unknown"])[0],
                        "content_summary": item.get("content", ""), # Use the extracted content
                        "score": item.get("final_score", 0),
                        "retrieval_source": item.get("metadata", {}).get("source", "graph") # text2cypher or neighbor_search
                    }
                    context["graph_knowledge"]["retrieved_entities"].append(entity_info)
                    processed_node_ids.add(entity_id)
                
                # Attempt to extract relationship info if present (mainly from neighbor search)
                rel_types = item.get("metadata", {}).get("relationship_types", [])
                if rel_types:
                     # Basic relationship representation; more complex structure might be needed
                     relation = {
                         "type": ", ".join(rel_types),
                         "related_to": item.get("id"),
                         "from_seed": item.get("metadata", {}).get("seed_entity"),
                         "distance": item.get("metadata", {}).get("distance")
                     }
                     context["graph_knowledge"]["relationships"].append(relation)

            # Process items originating from vector search
            if "vector" in source_types:
                 passage = {
                    "content": item.get("content", ""),
                    "source_document_id": item.get("id", "unknown"), # Use item ID as potential doc ID
                    "relevance_score": item.get("final_score", 0) # Use final score reflecting vector contribution
                 }
                 context["text_context"]["relevant_passages"].append(passage)

        # Try finding shortest path if multiple relevant entities were identified
        if len(entities) >= 2:
            # Try to find paths between the top graph entities retrieved
            top_graph_entities = [e['id'] for e in context["graph_knowledge"]["retrieved_entities"][:2] if e.get('id')]
            if len(top_graph_entities) >= 2:
                 path = self.find_shortest_path(top_graph_entities[0], top_graph_entities[1])
                 if path and path.get("path_length", 0) > 0:
                     context["graph_knowledge"]["paths"].append(path)
        
        # --- Generate combined_context string ---
        combined_context_parts = []
        combined_context_parts.append(f"基于查询 '{query}' 的检索结果:")

        # Add Graph Knowledge Summary
        if context["graph_knowledge"]["retrieved_entities"]:
            combined_context_parts.append("\n--- 图知识 ---")
            if context["graph_knowledge"]["executed_cypher_query"]:
                 combined_context_parts.append(f"执行的Cypher查询: {context['graph_knowledge']['executed_cypher_query']}")
            
            combined_context_parts.append("相关图实体:")
            for entity in context["graph_knowledge"]["retrieved_entities"][:5]: # Limit summary
                combined_context_parts.append(f"- {entity.get('label', 'Entity')}: {entity.get('id', 'N/A')} (来源: {entity.get('retrieval_source', 'N/A')}, 分数: {entity.get('score', 0):.2f})")
                # Add a snippet of content if useful
                # content_snippet = entity.get('content_summary', '').split('\n')[1][:100] + "..." if '\n' in entity.get('content_summary', '') else entity.get('content_summary', '')[:100] + "..."
                # combined_context_parts.append(f"  摘要: {content_snippet}")


        # Add Path Information
        if context["graph_knowledge"]["paths"]:
            combined_context_parts.append("\n发现的实体间路径:")
            for path in context["graph_knowledge"]["paths"][:1]: # Limit to one path summary
                path_nodes = path.get("path_nodes", [])
                if path_nodes:
                    node_strs = [f"{node.get('labels', [''])[0]}:{node.get('id', '')}" for node in path_nodes]
                    combined_context_parts.append(f"- {' -> '.join(node_strs)} (长度: {path.get('path_length')})")

        # Add Text Context Summary
        if context["text_context"]["relevant_passages"]:
            combined_context_parts.append("\n--- 相关文本段落 ---")
            for i, passage in enumerate(context["text_context"]["relevant_passages"][:3]): # Limit summary
                content_snippet = passage.get("content", "")
                if len(content_snippet) > 200:
                    content_snippet = content_snippet[:197] + "..."
                combined_context_parts.append(f"{i+1}. (来源ID: {passage.get('source_document_id', 'N/A')}, 分数: {passage.get('relevance_score', 0):.2f})\n   {content_snippet}")

        context["combined_context"] = "\n".join(combined_context_parts)
        
        return context
        
    def multi_strategy_retrieval(self, query: str, use_text2cypher: bool = True, force_cypher: str = None, 
                              cypher_params: Dict[str, Any] = None, force_strategy: str = None) -> Dict[str, Any]:
        """多策略检索 - 智能选择并执行最佳检索策略
        
        Args:
            query: 用户查询
            use_text2cypher: 是否允许使用 Text2Cypher 策略
            force_cypher: 可选的强制执行的Cypher查询（优先于Text2Cypher生成）
            cypher_params: 与force_cypher配合使用的查询参数
            force_strategy: 强制使用的策略名称
            
        Returns:
            检索结果
        """
        start_time = time.time()
        
        # 分析查询
        query_analysis = self.analyze_query(query)
        
        # 使用策略选择器智能选择最佳检索策略
        strategy = RetrievalStrategySelector.select_strategy(
            query=query,
            query_analysis=query_analysis,
            use_text2cypher=use_text2cypher and self.llm is not None,
            force_strategy=force_strategy
        )
        
        # 提取策略参数
        strategy_name = strategy.get("name", "balanced")
        graph_weight = strategy.get("graph_weight", 0.5)
        vector_weight = strategy.get("vector_weight", 0.5)
        attempt_text2cypher = strategy.get("use_text2cypher", use_text2cypher)
        force_text2cypher = strategy.get("force_text2cypher", False)
        
        logger.info(f"选择的检索策略: {strategy_name}, 图权重: {graph_weight}, 向量权重: {vector_weight}, 使用Text2Cypher: {attempt_text2cypher}")
        
        # 如果有强制Cypher查询或强制要求使用Text2Cypher
        if force_cypher and force_text2cypher:
            # 确保Text2Cypher的权重较高
            graph_weight = max(graph_weight, 0.8)
            vector_weight = min(vector_weight, 0.2)
            
        # 执行增强混合搜索
        results = self.enhanced_hybrid_search(
            query=query, 
            graph_weight=graph_weight, 
            vector_weight=vector_weight,
            use_text2cypher=attempt_text2cypher,
            force_cypher=force_cypher,
            cypher_params=cypher_params
        )
            
        elapsed_time = time.time() - start_time
        logger.info(f"多策略检索完成，耗时 {elapsed_time:.2f} 秒")
        
        # 构建完整的结果包
        return {
            "query": query,
            "query_type": query_analysis.query_type,
            "retrieval_strategy": strategy_name,
            "graph_weight": graph_weight,
            "vector_weight": vector_weight,
            "elapsed_time": elapsed_time,
            "used_text2cypher": attempt_text2cypher,
            "results": results,
            "entities": results.get("entities", []),
            "executed_cypher": results.get("executed_cypher_query", ""),
            "executed_params": results.get("executed_cypher_params", {})
        }
            
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
    
class RetrievalStrategySelector:
    """检索策略选择器 - 基于查询类型和上下文智能选择最佳检索策略"""
    
    @staticmethod
    def select_strategy(query: str, query_analysis: QueryAnalysisResult = None, 
                       use_text2cypher: bool = True, force_strategy: str = None) -> Dict[str, float]:
        """
        基于查询类型和分析结果选择最佳检索策略
        
        Args:
            query: 用户查询
            query_analysis: 查询分析结果(如果已有)
            use_text2cypher: 是否允许使用Text2Cypher
            force_strategy: 强制使用的策略（优先级最高）
            
        Returns:
            包含策略名称和权重的字典
        """
        # 如果强制指定了策略，使用它
        if force_strategy:
            if force_strategy == "graph_only":
                return {"name": "graph_only", "graph_weight": 1.0, "vector_weight": 0.0, "use_text2cypher": use_text2cypher}
            elif force_strategy == "vector_only":
                return {"name": "vector_only", "graph_weight": 0.0, "vector_weight": 1.0, "use_text2cypher": False}
            elif force_strategy == "balanced":
                return {"name": "balanced", "graph_weight": 0.5, "vector_weight": 0.5, "use_text2cypher": use_text2cypher}
            elif force_strategy == "text2cypher_only":
                return {"name": "text2cypher_only", "graph_weight": 1.0, "vector_weight": 0.0, "use_text2cypher": True, "force_text2cypher": True}
        
        # 如果没有提供分析结果，创建一个默认的
        if not query_analysis:
            # 使用空分析，稍后会在multi_strategy_retrieval中进行真正的分析
            query_analysis = QueryAnalysisResult(query=query)
        
        # 基于查询类型选择策略
        query_type = query_analysis.query_type
        
        # 包含SQL、查询、数据库等术语的查询很可能适合用text2cypher
        has_database_terms = any(term in query.lower() for term in [
            "sql", "query", "database", "图数据库", "查询", "数据", "neo4j", 
            "cypher", "关系", "节点", "实体", "属性", "模式", "schema"
        ])
        
        # 命令式查询（"查找"、"列出"等）也适合text2cypher
        is_command_query = any(term in query.lower() for term in [
            "查找", "搜索", "列出", "展示", "显示", "获取", "返回", "统计", 
            "find", "search", "list", "show", "display", "get", "return", "count"
        ])
        
        # 检查查询中是否包含聚合或排序术语
        has_aggregation = any(term in query.lower() for term in [
            "平均", "最大", "最小", "总和", "计数", "排序", "前几个", "按", "分组",
            "average", "max", "min", "sum", "count", "sort", "order", "group", "top", "by"
        ])
        
        # 计算text2cypher适用性分数（0-1之间的值）
        text2cypher_score = 0.0
        if has_database_terms:
            text2cypher_score += 0.4
        if is_command_query:
            text2cypher_score += 0.3
        if has_aggregation:
            text2cypher_score += 0.3
        if query_type in ["relationship_focused", "comparison"]:
            text2cypher_score += 0.4
        if query_type == "entity_focused":
            text2cypher_score += 0.2
            
        # 归一化分数到0-1之间
        text2cypher_score = min(1.0, text2cypher_score)
        
        # 基于查询类型和text2cypher得分确定最终策略
        if query_type == "relationship_focused":
            # 关系聚焦查询优先使用图检索
            strategy = {
                "name": "graph_heavy", 
                "graph_weight": 0.8, 
                "vector_weight": 0.2,
                "use_text2cypher": use_text2cypher and text2cypher_score > 0.3
            }
        elif query_type == "entity_focused":
            # 实体聚焦查询平衡使用图检索和向量检索
            strategy = {
                "name": "entity_focused", 
                "graph_weight": 0.6, 
                "vector_weight": 0.4,
                "use_text2cypher": use_text2cypher and text2cypher_score > 0.3
            }
        elif query_type == "comparison":
            # 比较类查询通常适合使用图检索
            strategy = {
                "name": "comparison", 
                "graph_weight": 0.7, 
                "vector_weight": 0.3,
                "use_text2cypher": use_text2cypher and text2cypher_score > 0.4
            }
        elif text2cypher_score > 0.6 and use_text2cypher:
            # 对于高text2cypher得分的查询，优先使用text2cypher
            strategy = {
                "name": "text2cypher_preferred", 
                "graph_weight": 0.9, 
                "vector_weight": 0.1,
                "use_text2cypher": True
            }
        else:
            # 其他类型的查询使用平衡策略
            strategy = {
                "name": "balanced", 
                "graph_weight": 0.5, 
                "vector_weight": 0.5,
                "use_text2cypher": use_text2cypher and text2cypher_score > 0.5
            }
            
        logger.info(f"为查询'{query}'选择的检索策略: {strategy['name']}, " 
                   f"text2cypher得分: {text2cypher_score:.2f}, 使用text2cypher: {strategy['use_text2cypher']}")
        
        return strategy


