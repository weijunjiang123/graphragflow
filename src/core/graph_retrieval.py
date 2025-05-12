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
        self.supports_text2cypher = self._check_text2cypher_support() if llm else False
        logger.info(f"初始化图检索器，连接到 {uri}")
        if self.supports_text2cypher:
            logger.info("大语言模型支持text2Cypher功能")
        else:
            logger.info("大语言模型不支持text2Cypher功能或未提供LLM")
    
    def _check_text2cypher_support(self) -> bool:
        """检查LLM是否支持text2Cypher功能
        
        Returns:
            是否支持text2Cypher
        """
        try:
            # 简单测试LLM生成Cypher的能力
            test_prompt = """
            生成一个简单的Cypher查询，查找名为'Alice'的用户节点。
            """
            
            result = self.llm.invoke(test_prompt)
            
            # 检查结果中是否包含MATCH和RETURN等Cypher关键字
            import re
            cypher_pattern = r'(MATCH|RETURN|WHERE)'
            match = re.search(cypher_pattern, result, re.IGNORECASE)
            
            return match is not None
            
        except Exception as e:
            logger.warning(f"检查text2Cypher支持失败: {str(e)}")
            return False

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
        
        try:
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
        except Exception as e:
            logger.error(f"实体提取过程中发生异常: {str(e)}")
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
        try:
            # 提取实体，添加异常处理
            try:
                entities = self.extract_entities(query)
            except Exception as e:
                logger.warning(f"实体提取失败，忽略实体过滤: {str(e)}")
                entities = []

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
        except Exception as e:
            logger.error(f"混合搜索过程中发生异常: {str(e)}")
            # 所有方法都失败时，返回空结果
            return []
                
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
    
    def get_schema_info(self) -> str:
        """获取数据库模式信息
        
        Returns:
            数据库模式信息字符串
        """
        cypher_query = """
        CALL apoc.meta.schema()
        YIELD value
        RETURN value
        """
        
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query).single()
                if not result:
                    logger.warning("无法获取数据库模式信息")
                    return ""
                
                schema_data = result["value"]
                
                # 提取节点标签和属性
                node_labels = []
                for label, data in schema_data.items():
                    if "properties" in data:
                        properties = list(data["properties"].keys())
                        node_labels.append({
                            "label": label,
                            "properties": properties
                        })
                
                # 提取关系类型和属性
                relationship_types = []
                rel_query = """
                MATCH ()-[r]->()
                WITH type(r) AS relType, keys(r) AS relKeys
                RETURN DISTINCT relType, relKeys
                """
                
                rel_results = session.run(rel_query)
                for record in rel_results:
                    relationship_types.append({
                        "type": record["relType"],
                        "properties": record["relKeys"]
                    })
                
                # 格式化为人类可读的字符串
                schema_info = "节点标签和属性:\n"
                for node in node_labels:
                    schema_info += f"- 标签: {node['label']}, 属性: {', '.join(node['properties'])}\n"
                
                schema_info += "\n关系类型和属性:\n"
                for rel in relationship_types:
                    schema_info += f"- 类型: {rel['type']}, 属性: {', '.join(rel['properties'])}\n"
                
                return schema_info
                
            except Exception as e:
                logger.error(f"获取数据库模式信息失败: {str(e)}")
                return ""
    
    def text_to_cypher(self, query: str) -> str:
        """将自然语言查询转换为Cypher查询
        
        Args:
            query: 自然语言查询
            
        Returns:
            Cypher查询语句
        """
        if not self.llm:
            logger.error("未提供LLM模型，无法执行text2Cypher转换")
            return ""
        
        # 获取数据库模式信息
        schema_info = self.get_schema_info()
        
        # 创建提示
        prompt = f"""
        你是一位专业的Neo4j数据库查询专家。我需要你将以下自然语言查询转换为Cypher查询语句。
        请确保生成的Cypher查询语法正确，并能够准确捕捉查询意图。

        数据库模式信息:
        {schema_info}

        自然语言查询: {query}

        仅返回Cypher查询语句，不要包含任何解释或其他内容。查询应当包含LIMIT子句以限制返回结果数量。
        """
        
        try:
            # 调用LLM生成Cypher查询
            result = self.llm.invoke(prompt)
            
            # 提取Cypher查询（假设返回的可能包含其他文本）
            import re
            # 尝试匹配带有MATCH, RETURN等关键字的Cypher查询
            cypher_pattern = r'(?:MATCH|CALL|CREATE|MERGE)[\s\S]+?(?:RETURN|YIELD)[\s\S]+?(?:;|\Z)'
            match = re.search(cypher_pattern, result, re.IGNORECASE)
            
            if match:
                cypher_query = match.group(0).strip()
            else:
                # 如果没有匹配到典型的Cypher查询模式，直接使用整个结果
                cypher_query = result.strip()
            
            logger.info(f"生成的Cypher查询: {cypher_query}")
            return cypher_query            
        except Exception as e:
            logger.error(f"text2Cypher转换失败: {str(e)}")
            return ""
            
    def execute_cypher(self, cypher_query: str, parameters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """执行Cypher查询
        
        Args:
            cypher_query: Cypher查询语句
            parameters: 查询参数
            
        Returns:
            查询结果列表
        """
        if not cypher_query:
            logger.error("Cypher查询为空")
            return []
            
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(cypher_query, parameters or {})
                return [record.data() for record in result]
            except Exception as e:
                logger.error(f"执行Cypher查询失败: {str(e)}")
                return []
                
    def text2cypher_search(self, query: str, limit: int = 5, return_formatted: bool = False) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """使用text2Cypher进行搜索
        
        此方法将自然语言查询转换为Cypher查询语句，然后执行查询并返回结果。
        它利用大语言模型(LLM)的能力，将用户的自然语言问题直接转换为Neo4j的Cypher查询语言。
        这种方法比传统的关键词搜索或向量搜索更精确，因为它可以捕捉查询的语义意图和复杂关系。
        
        示例查询:
        - "查找与人工智能相关的所有文档"
        - "找出引用最多的前5个作者"
        - "返回包含'图数据库'关键词的文档及其相关实体"
        
        Args:
            query: 自然语言查询
            limit: 结果数量限制
            return_formatted: 是否返回格式化的结果（包含原始查询、Cypher查询和结果解释）
            
        Returns:
            当return_formatted=False时: 搜索结果列表
            当return_formatted=True时: 包含格式化结果的字典
        """        # 如果未提供LLM或不支持text2cypher，回退到混合搜索
        if not self.llm or not self.supports_text2cypher:
            logger.warning("LLM不支持text2Cypher功能，回退到混合搜索")
            results = self.hybrid_search(query, limit)
            return results
            
        try:
            # 1. 将自然语言转换为Cypher
            cypher_query = self.text_to_cypher(query)
            
            if not cypher_query:
                logger.warning("无法生成有效的Cypher查询，回退到混合搜索")
                results = self.hybrid_search(query, limit)
                if return_formatted:
                    return {
                        "original_query": query,
                        "cypher_query": "无法生成有效的Cypher查询",
                        "results": results,
                        "fallback_method": "hybrid_search",
                        "result_count": len(results)
                    }
                return results
                
            # 确保查询中包含LIMIT子句
            if "LIMIT" not in cypher_query.upper():
                if ";" in cypher_query:
                    cypher_query = cypher_query.replace(";", f" LIMIT {limit};")
                else:
                    cypher_query = f"{cypher_query} LIMIT {limit}"
                    
            logger.info(f"执行Cypher查询: {cypher_query}")
                
            # 2. 执行Cypher查询
            results = self.execute_cypher(cypher_query)
            
            # 3. 如果结果为空，回退到混合搜索
            if not results:
                logger.warning("Cypher查询未返回结果，回退到混合搜索")
                fallback_results = self.hybrid_search(query, limit)
                if return_formatted:
                    return {
                        "original_query": query,
                        "cypher_query": cypher_query,
                        "results": fallback_results,
                        "fallback_method": "hybrid_search",
                        "result_count": len(fallback_results)
                    }
                return fallback_results
            
            # 4. 返回结果
            if return_formatted:
                return self.format_text2cypher_results(results, query, cypher_query)
            return results
            
        except Exception as e:
            logger.error(f"text2Cypher搜索失败: {str(e)}")
            # 发生错误时回退到混合搜索
            fallback_results = self.hybrid_search(query, limit)
            if return_formatted:
                return {
                    "original_query": query,
                    "error": str(e),
                    "results": fallback_results,
                    "fallback_method": "hybrid_search",
                    "result_count": len(fallback_results)
                }
            return fallback_results
    
    def format_text2cypher_results(self, results: List[Dict[str, Any]], original_query: str, cypher_query: str) -> Dict[str, Any]:
        """格式化text2Cypher查询结果为用户友好的格式
        
        Args:
            results: 查询结果列表
            original_query: 原始自然语言查询
            cypher_query: 生成的Cypher查询
            
        Returns:
            格式化后的结果
        """
        # 提取结果中的实体和关系
        entities = set()
        relationships = []
        
        # 处理结果字段
        all_fields = set()
        for record in results:
            all_fields.update(record.keys())
            
            # 尝试找出记录中的节点和关系
            for key, value in record.items():
                # 检测可能的节点（具有id和标签属性的对象）
                if isinstance(value, dict) and "id" in value and ("label" in value or "labels" in value):
                    entities.add(value["id"])
                # 检测可能的关系（具有source和target属性的对象）
                elif isinstance(value, dict) and "source" in value and "target" in value:
                    relationships.append(value)
        
        # 使用LLM生成结果解释（如果有LLM可用）
        explanation = ""
        if self.llm:
            try:
                # 构建结果摘要
                results_summary = {}
                # 限制结果数量以避免提示太长
                max_results = min(3, len(results))
                for i, record in enumerate(results[:max_results]):
                    results_summary[f"结果_{i+1}"] = record
                
                # 创建解释提示
                explain_prompt = f"""
                请分析以下查询和结果，提供简洁明了的解释：
                
                原始查询: {original_query}
                
                转换后的Cypher查询: {cypher_query}
                
                查询结果(部分): {json.dumps(results_summary, ensure_ascii=False, indent=2)}
                
                请解释:
                1. 这个查询的目的是什么
                2. 返回的结果代表什么
                3. 结果中的主要发现或见解
                
                请使用简洁的语言，不超过100字。
                """
                
                explanation = self.llm.invoke(explain_prompt)
            except Exception as e:
                logger.error(f"生成结果解释失败: {str(e)}")
                explanation = "无法生成结果解释。"
        
        return {
            "original_query": original_query,
            "cypher_query": cypher_query,
            "results": results,
            "result_fields": list(all_fields),
            "entities_count": len(entities),
            "relationships_count": len(relationships),
            "explanation": explanation,
            "result_count": len(results)
        }
    
    def get_text2cypher_examples(self) -> List[Dict[str, str]]:
        """获取text2Cypher的示例查询
        
        Returns:
            示例查询列表
        """
        return [
            {
                "description": "基础查询 - 查找文档",
                "query": "查找包含'图数据库'关键词的文档",
                "explanation": "展示如何基于关键词查找文档"
            },
            {
                "description": "关系查询 - 实体关联",
                "query": "找出与人工智能相关的所有实体及其关联的文档",
                "explanation": "展示如何查询实体之间的关系"
            },
            {
                "description": "统计查询 - 聚合数据",
                "query": "统计每种类型的实体数量",
                "explanation": "展示如何使用聚合函数进行统计分析"
            },
            {
                "description": "路径查询 - 知识图谱遍历",
                "query": "查找从'机器学习'到'深度学习'的所有路径",
                "explanation": "展示如何在知识图谱中查询路径"
            },
            {
                "description": "复杂查询 - 组合条件",
                "query": "查找2023年之后发布的，与大语言模型相关，并且被引用次数大于10的文档",
                "explanation": "展示如何组合多个条件进行复杂查询"
            }
        ]
        
    def test_text2cypher_capability(self) -> Dict[str, Any]:
        """测试text2Cypher功能
        
        Returns:
            测试结果字典
        """
        if not self.llm:
            return {
                "success": False,
                "reason": "未提供LLM模型",
                "supports_text2cypher": False
            }
            
        try:
            # 使用简单查询测试LLM生成Cypher的能力
            test_query = "查找所有文档节点"
            cypher_query = self.text_to_cypher(test_query)
            
            if not cypher_query:
                return {
                    "success": False,
                    "reason": "LLM未能生成有效的Cypher查询",
                    "supports_text2cypher": False
                }
                
            # 执行查询测试
            try:
                results = self.execute_cypher(cypher_query)
                return {
                    "success": True,
                    "query": test_query,
                    "cypher": cypher_query,
                    "result_count": len(results),
                    "supports_text2cypher": True,
                    "examples": self.get_text2cypher_examples()
                }
            except Exception as e:
                return {
                    "success": False,
                    "reason": f"执行Cypher查询失败: {str(e)}",
                    "cypher": cypher_query,
                    "supports_text2cypher": True,  # 仍然支持，但执行失败
                    "examples": self.get_text2cypher_examples()
                }
                
        except Exception as e:
            return {
                "success": False,
                "reason": f"测试text2Cypher功能失败: {str(e)}",
                "supports_text2cypher": False
            }


