import logging
from typing import Optional, Dict, Any, Tuple, List

from neo4j import GraphDatabase
try:
    # 尝试导入新版本的Neo4jGraph
    from langchain_neo4j import Neo4jGraph
except ImportError:
    # 降级使用旧版(会产生警告)
    from langchain_community.graphs import Neo4jGraph
    logging.warning("请安装 langchain-neo4j 包以解决弃用警告: pip install langchain-neo4j")
import neo4j
from neo4j.exceptions import ClientError

logger = logging.getLogger(__name__)

# 常量定义
BASE_ENTITY_LABEL = "__Entity__"
BASE_KG_BUILDER_LABEL = "__KGBuilder__"
EXCLUDED_LABELS = ["Dataset", "Bloom"]
EXCLUDED_RELS = ["CONTAINS", "HAS_DATASET"]

# 更新查询语句以处理不同版本的Neo4j
NODE_PROPERTIES_QUERY = """
CALL db.schema.nodeTypeProperties()
YIELD nodeType, nodeLabels, propertyName, propertyTypes, mandatory
WHERE NOT any(label IN nodeLabels WHERE label IN $EXCLUDED_LABELS)
WITH nodeLabels, collect({property: propertyName, type: propertyTypes[0], required: mandatory}) as properties
UNWIND nodeLabels as label
RETURN {label: label, properties: properties} as output
"""

REL_PROPERTIES_QUERY = """
CALL db.schema.relTypeProperties()
YIELD relType, propertyName, propertyTypes, mandatory
WHERE NOT relType IN $EXCLUDED_LABELS
WITH relType, collect({property: propertyName, type: propertyTypes[0], required: mandatory}) as properties
RETURN {type: relType, properties: properties} as output
"""

# 使用更简单的关系查询，避免使用可能不存在的属性
REL_QUERY = """
MATCH (a)-[r]->(b)
WITH labels(a) AS startLabels, type(r) AS relType, labels(b) AS endLabels
RETURN {start: startLabels[0], type: relType, end: endLabels[0]} AS output
LIMIT 100
"""

# 更新索引查询以适应不同版本的Neo4j
INDEX_QUERY = """
SHOW INDEXES
YIELD name, type, labelsOrTypes, properties
RETURN {
  label: CASE 
           WHEN size(labelsOrTypes) > 0 THEN labelsOrTypes[0] 
           ELSE "Unknown" 
         END,
  properties: properties,
  type: type,
  size: 0,
  valuesSelectivity: 0.0,
  distinctValues: 0.0
}
"""

def format_schema(structured_schema: Dict[str, Any], is_enhanced: bool = False) -> str:
    """
    Format the structured schema into a string representation.
    
    Args:
        structured_schema (Dict[str, Any]): The structured schema dictionary
        is_enhanced (bool): Whether to include enhanced details in the output
        
    Returns:
        str: The formatted schema as a string
    """
    result = []
    
    # Add node properties
    result.append("Node properties:")
    if structured_schema["node_props"]:
        for label, properties in structured_schema["node_props"].items():
            if isinstance(properties, list):
                # 处理属性列表的情况
                props_str = ", ".join([f"{prop.get('property', 'unknown')}: {prop.get('type', 'unknown')}" for prop in properties])
            elif isinstance(properties, dict) and "properties" in properties:
                # 处理增强模式的情况，属性在properties键中
                props_list = properties.get("properties", [])
                props_str = ", ".join([f"{prop.get('property', 'unknown')}: {prop.get('type', 'unknown')}" for prop in props_list])
            else:
                props_str = "No properties"
            result.append(f"{label} {{{props_str}}}")
    else:
        result.append("No node properties found")
    
    # Add relationship properties
    result.append("\nRelationship properties:")
    if structured_schema["rel_props"]:
        for rel_type, properties in structured_schema["rel_props"].items():
            if isinstance(properties, list):
                props_str = ", ".join([f"{prop.get('property', 'unknown')}: {prop.get('type', 'unknown')}" for prop in properties])
            elif isinstance(properties, dict) and "properties" in properties:
                props_list = properties.get("properties", [])
                props_str = ", ".join([f"{prop.get('property', 'unknown')}: {prop.get('type', 'unknown')}" for prop in props_list])
            else:
                props_str = "No properties"
            result.append(f"{rel_type} {{{props_str}}}")
    else:
        result.append("No relationship properties found")
    
    # Add relationships
    result.append("\nThe relationships:")
    if structured_schema["relationships"]:
        for rel in structured_schema["relationships"]:
            if isinstance(rel, dict) and "start" in rel and "type" in rel and "end" in rel:
                result.append(f"(:{rel['start']})-[:{rel['type']}]->(:{rel['end']})")
            else:
                logger.warning(f"Invalid relationship format: {rel}")
    else:
        result.append("No relationships found")
    
    # Add enhanced details if requested
    if is_enhanced and "metadata" in structured_schema:
        result.append("\nMetadata:")
        if "constraint" in structured_schema["metadata"]:
            result.append("Constraints:")
            for constraint in structured_schema["metadata"]["constraint"]:
                result.append(f"  {constraint['name']}: {constraint['type']} on {constraint['labelsOrTypes']} ({', '.join(constraint['properties'])})")
                
        if "index" in structured_schema["metadata"]:
            result.append("Indexes:")
            for index in structured_schema["metadata"]["index"]:
                result.append(f"  {index['type']} on :{index['label']} ({', '.join(index['properties'])})")
    
    return "\n".join(result)

class Neo4jConnectionManager:
    """Singleton class to manage Neo4j driver connections"""
    _instance: Optional[GraphDatabase.driver] = None
    
    @classmethod
    def get_instance(cls, uri: str, auth: Tuple[str, str], **kwargs) -> GraphDatabase.driver:
        """Get or create a Neo4j driver instance
        
        Args:
            uri: Neo4j connection URI
            auth: Tuple of (username, password)
            **kwargs: Additional driver configuration
            
        Returns:
            Neo4j driver instance
        """
        if cls._instance is None:
            # Set reasonable defaults if not provided
            config = {
                'max_connection_lifetime': 3600,
                'max_connection_pool_size': 50,
                'connection_acquisition_timeout': 60
            }
            config.update(kwargs)
            
            cls._instance = GraphDatabase.driver(
                uri=uri, 
                auth=auth,
                **config
            )
            logger.info("Created new Neo4j driver connection")
        return cls._instance
    
    @classmethod
    def close(cls) -> None:
        """Close the Neo4j driver connection"""
        if cls._instance:
            cls._instance.close()
            cls._instance = None
            logger.info("Closed Neo4j driver connection")


class Neo4jManager:
    """Manager class for Neo4j operations"""
    
    def __init__(self, url: str, username: str, password: str):
        """Initialize Neo4j manager
        
        Args:
            url: Neo4j connection URL
            username: Neo4j username
            password: Neo4j password
        """
        self.url = url
        self.username = username
        self.password = password
        self.driver = Neo4jConnectionManager.get_instance(url, (username, password))
        try:
            # 尝试使用新版Neo4jGraph
            self.graph = Neo4jGraph(url=url, username=username, password=password)
        except TypeError:
            # 如果参数不匹配，尝试使用不同的参数名称
            self.graph = Neo4jGraph(url, username, password)
        
    def create_fulltext_index(self, index_name: str = "fulltext_entity_id") -> bool:
        """Create fulltext index if it doesn't exist
        
        Args:
            index_name: Name of the fulltext index
            
        Returns:
            True if index was created or already exists, False otherwise
        """
        query = f'''
        CREATE FULLTEXT INDEX `{index_name}` 
        IF NOT EXISTS
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        
        try:
            with self.driver.session() as session:
                session.run(query)
                logger.info(f"Fulltext index '{index_name}' created successfully.")
                return True
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Index '{index_name}' already exists, skipping creation.")
                return True
            else:
                logger.error(f"Error creating fulltext index: {str(e)}")
                return False
                
    def add_graph_documents(self, graph_documents, **kwargs):
        """Add graph documents to Neo4j
        
        Args:
            graph_documents: List of graph documents to add
            **kwargs: Additional parameters for add_graph_documents
        """
        return self.graph.add_graph_documents(
            graph_documents,
            **kwargs
        )
        
    def drop_index(self, index_name: str) -> bool:
        """Drop an index if it exists
        
        Args:
            index_name: Name of the index to drop
            
        Returns:
            True if index was dropped, False otherwise
        """
        try:
            with self.driver.session() as session:
                # Check if index exists
                result = session.run(
                    f"SHOW VECTOR INDEXES WHERE name = $name",
                    name=index_name
                ).single()
                
                if result:
                    logger.info(f"Found existing vector index '{index_name}' - dropping it...")
                    session.run(f"DROP VECTOR INDEX {index_name}")
                    logger.info(f"Dropped existing vector index '{index_name}'")
                    return True
                else:
                    logger.info(f"No index named '{index_name}' found to drop")
                    return False
        except Exception as e:
            logger.warning(f"Error when trying to drop vector index: {str(e)}")
            return False

    def get_schema(
        self,
        driver: neo4j.Driver,
        is_enhanced: bool = False,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
    ) -> str:
        """
        Returns the schema of the graph as a string with following format:

        .. code-block:: text

            Node properties:
            Person {id: INTEGER, name: STRING}
            Relationship properties:
            KNOWS {fromDate: DATE}
            The relationships:
            (:Person)-[:KNOWS]->(:Person)

        Args:
            driver (neo4j.Driver): Neo4j Python driver instance.
            is_enhanced (bool): Flag indicating whether to format the schema with
                detailed statistics (True) or in a simpler overview format (False).
            database (Optional[str]): The name of the database to connect to. Default is 'neo4j'.
            timeout (Optional[float]): The timeout for transactions in seconds.
                    Useful for terminating long-running queries.
                    By default, there is no timeout set.
            sanitize (bool): A flag to indicate whether to remove lists with
                    more than 128 elements from results. Useful for removing
                    embedding-like properties from database responses. Default is False.


        Returns:
            str: the graph schema information in a serialized format.
        """
        structured_schema = self.get_structured_schema(
            driver=driver,
            is_enhanced=is_enhanced,
            database=database,
            timeout=timeout,
            sanitize=sanitize,
        )
        return format_schema(structured_schema, is_enhanced)

    def query_database(
        self,
        driver: neo4j.Driver,
        query: str,
        params: Dict[str, Any] = {},
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Execute a query against the Neo4j database.
        
        Args:
            driver: Neo4j driver instance
            query: Cypher query to execute
            params: Query parameters
            database: Database name (optional)
            timeout: Query timeout in seconds
            sanitize: Whether to sanitize the response
            
        Returns:
            List of result records as dictionaries
        """
        with driver.session(database=database) as session:
            result = session.run(query, params, timeout=timeout)
            records = []
            for record in result:
                if sanitize:
                    record = self._sanitize_record(record)
                records.append(dict(record))
            return records
            
    def _sanitize_record(self, record):
        """Sanitize a record to remove large lists (e.g. embeddings)"""
        sanitized = {}
        for key, value in record.items():
            if isinstance(value, list) and len(value) > 128:
                sanitized[key] = f"List with {len(value)} items (sanitized)"
            else:
                sanitized[key] = value
        return sanitized

    def enhance_schema(
        self,
        driver: neo4j.Driver,
        structured_schema: Dict[str, Any],
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
    ) -> None:
        """
        Enhance the schema with additional statistics and metadata.
        
        Args:
            driver: Neo4j driver instance
            structured_schema: Schema to enhance
            database: Database name
            timeout: Query timeout
            sanitize: Whether to sanitize the response
        """
        try:
            # Get node counts
            node_counts_query = """
            MATCH (n)
            WITH labels(n) as labels, count(*) as count
            UNWIND labels as label
            RETURN label, sum(count) as count
            """
            
            node_counts = {}
            for record in self.query_database(
                driver=driver,
                query=node_counts_query,
                database=database,
                timeout=timeout,
                sanitize=sanitize
            ):
                node_counts[record["label"]] = record["count"]
                
            # Add counts to schema
            for label, props in structured_schema["node_props"].items():
                count = node_counts.get(label, 0)
                structured_schema["node_props"][label] = {
                    "properties": props,
                    "count": count
                }
                
            # Could add more enhancements here (relationship counts, etc.)
        except Exception as e:
            logger.warning(f"Error enhancing schema: {str(e)}")

    def get_structured_schema(
        self,
        driver: neo4j.Driver,
        is_enhanced: bool = False,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
    ) -> Dict[str, Any]:
        """
        Returns the structured schema of the graph.

        Returns a dict with following format:

        .. code:: python

            {
                'node_props': {
                    'Person': [{'property': 'id', 'type': 'INTEGER'}, {'property': 'name', 'type': 'STRING'}]
                },
                'rel_props': {
                    'KNOWS': [{'property': 'fromDate', 'type': 'DATE'}]
                },
                'relationships': [
                    {'start': 'Person', 'type': 'KNOWS', 'end': 'Person'}
                ],
                'metadata': {
                    'constraint': [
                        {'id': 7, 'name': 'person_id', 'type': 'UNIQUENESS', 'entityType': 'NODE', 'labelsOrTypes': ['Persno'], 'properties': ['id'], 'ownedIndex': 'person_id', 'propertyType': None},
                    ],
                    'index': [
                        {'label': 'Person', 'properties': ['name'], 'size': 2, 'type': 'RANGE', 'valuesSelectivity': 1.0, 'distinctValues': 2.0},
                    ]
                }
            }

        Note:
            The internal structure of the returned dict depends on the apoc.meta.data
            and apoc.schema.nodes procedures.

        Warning:
            Some labels are excluded from the output schema:

            - The `__Entity__` and `__KGBuilder__` node labels which are created by the KG Builder pipeline within this package
            - Some labels related to Bloom internals.

        Args:
            driver (neo4j.Driver): Neo4j Python driver instance.
            is_enhanced (bool): Flag indicating whether to format the schema with
                detailed statistics (True) or in a simpler overview format (False).
            database (Optional[str]): The name of the database to connect to. Default is 'neo4j'.
            timeout (Optional[float]): The timeout for transactions in seconds.
                Useful for terminating long-running queries.
                By default, there is no timeout set.
            sanitize (bool): A flag to indicate whether to remove lists with
                more than 128 elements from results. Useful for removing
                embedding-like properties from database responses. Default is False.

        Returns:
            dict[str, Any]: the graph schema information in a structured format.
        """
        try:
            # 获取节点属性
            node_properties = []
            try:
                node_props_result = self.query_database(
                    driver=driver,
                    query=NODE_PROPERTIES_QUERY,
                    params={
                        "EXCLUDED_LABELS": EXCLUDED_LABELS
                        + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL]
                    },
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize,
                )
                
                for data in node_props_result:
                    if isinstance(data, dict) and "output" in data:
                        node_properties.append(data["output"])
            except Exception as e:
                logger.warning(f"获取节点属性时出错: {e}")

            # 获取关系属性
            rel_properties = []
            try:
                rel_props_result = self.query_database(
                    driver=driver,
                    query=REL_PROPERTIES_QUERY,
                    params={"EXCLUDED_LABELS": EXCLUDED_RELS},
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize,
                )
                
                for data in rel_props_result:
                    if isinstance(data, dict) and "output" in data:
                        rel_properties.append(data["output"])
            except Exception as e:
                logger.warning(f"获取关系属性时出错: {e}")

            # 获取关系
            relationships = []
            try:
                relationships_result = self.query_database(
                    driver=driver,
                    query=REL_QUERY,
                    params={
                        "EXCLUDED_LABELS": EXCLUDED_LABELS
                        + [BASE_ENTITY_LABEL, BASE_KG_BUILDER_LABEL]
                    },
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize,
                )
                
                for item in relationships_result:
                    if isinstance(item, dict):
                        # 有些版本可能直接返回数据，有些可能嵌套在output键中
                        if "output" in item:
                            relationships.append(item["output"])
                        # 如果数据已经包含需要的键，直接使用
                        elif all(k in item for k in ["start", "type", "end"]):
                            relationships.append(item)
                        else:
                            logger.warning(f"未知的关系数据结构: {item}")
            except Exception as e:
                logger.warning(f"获取关系时出错: {e}")

            # 获取约束和索引
            constraint = []
            index = []
            try:
                constraint = self.query_database(
                    driver=driver,
                    query="SHOW CONSTRAINTS",
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize,
                )
            except Exception as e:
                logger.warning(f"获取约束时出错: {e}")
                
            try:
                index = self.query_database(
                    driver=driver,
                    query=INDEX_QUERY,
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize,
                )
            except Exception as e:
                logger.warning(f"获取索引时出错: {e}")

            # 构建结构化模式
            structured_schema = {
                "node_props": {},
                "rel_props": {},
                "relationships": relationships,
                "metadata": {"constraint": constraint, "index": index},
            }
            
            # 安全地添加节点属性
            for el in node_properties:
                if isinstance(el, dict) and "label" in el and "properties" in el:
                    structured_schema["node_props"][el["label"]] = el["properties"]
            
            # 安全地添加关系属性
            for el in rel_properties:
                if isinstance(el, dict) and "type" in el and "properties" in el:
                    structured_schema["rel_props"][el["type"]] = el["properties"]
            
            if is_enhanced:
                try:
                    self.enhance_schema(
                        driver=driver,
                        structured_schema=structured_schema,
                        database=database,
                        timeout=timeout,
                        sanitize=sanitize,
                    )
                except Exception as e:
                    logger.warning(f"增强模式时出错: {e}")
                    
            return structured_schema
            
        except Exception as e:
            logger.error(f"获取结构化模式时出错: {str(e)}")
            # 返回空结构作为默认值
            return {
                "node_props": {},
                "rel_props": {},
                "relationships": [],
                "metadata": {"constraint": [], "index": []},
            }
        
    def test_schema_retrieval(
        self, 
        is_enhanced: bool = False,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        print_output: bool = True
    ) -> Dict[str, Any]:
        """
        测试模式获取功能，并返回测试结果
        
        Args:
            is_enhanced: 是否增强模式信息
            database: 数据库名称
            timeout: 查询超时时间
            sanitize: 是否净化结果
            print_output: 是否打印输出结果
            
        Returns:
            包含测试结果的字典
        """
        results = {
            "success": False,
            "structured_schema": None,
            "formatted_schema": None,
            "error": None
        }
        
        try:
            logger.info("开始测试模式获取功能...")
            
            # 获取结构化模式
            try:
                structured_schema = self.get_structured_schema(
                    driver=self.driver,
                    is_enhanced=is_enhanced,
                    database=database,
                    timeout=timeout,
                    sanitize=sanitize
                )
                results["structured_schema"] = structured_schema
            except Exception as schema_error:
                logger.error(f"获取结构化模式失败: {str(schema_error)}")
                # 使用空结构作为备用方案
                results["structured_schema"] = {
                    "node_props": {},
                    "rel_props": {},
                    "relationships": [],
                    "metadata": {"constraint": [], "index": []},
                }
            
            # 获取格式化模式字符串
            try:
                if results["structured_schema"]:
                    # 直接使用已获取的结构化模式而不是再次查询
                    formatted_schema = format_schema(results["structured_schema"], is_enhanced)
                    results["formatted_schema"] = formatted_schema
                else:
                    results["formatted_schema"] = "无法获取模式信息"
            except Exception as format_error:
                logger.error(f"格式化模式失败: {str(format_error)}")
                results["formatted_schema"] = "模式格式化过程中出错"
            
            results["success"] = True
            logger.info("模式获取测试成功完成")
            
            # 打印结果(如果需要)
            if print_output:
                print("\n" + "="*50)
                print("Neo4j Schema 获取结果:")
                print("="*50)
                print("\n## 格式化模式:\n")
                print(results["formatted_schema"])
                print("\n## 结构化模式摘要:")
                
                schema = results["structured_schema"]
                if schema:
                    print(f"- 节点类型: {len(schema.get('node_props', {}))}")
                    print(f"- 关系类型: {len(schema.get('rel_props', {}))}")
                    print(f"- 关系连接: {len(schema.get('relationships', []))}")
                    
                    metadata = schema.get("metadata", {})
                    print(f"- 约束条件: {len(metadata.get('constraint', []))}")
                    print(f"- 索引: {len(metadata.get('index', []))}")
                else:
                    print("- 无可用模式数据")
                print("="*50)
                
                if schema and not schema.get("node_props") and not schema.get("rel_props"):
                    print("\n提示: 数据库似乎是空的。这是正常的，您可以先导入一些数据然后再测试。")
        
        except Exception as e:
            error_msg = f"模式获取测试失败: {str(e)}"
            logger.error(error_msg)
            results["error"] = error_msg
            
        return results
        
    @classmethod
    def run_schema_test(
        cls, 
        url: str, 
        username: str, 
        password: str,
        is_enhanced: bool = False,
        database: Optional[str] = None,
        timeout: Optional[float] = 30.0
    ):
        """
        快速测试Neo4j模式获取的静态方法
        
        Args:
            url: Neo4j连接URL
            username: 用户名
            password: 密码
            is_enhanced: 是否获取增强模式
            database: 数据库名称
            timeout: 查询超时时间(秒)
        """
        try:
            print(f"连接到Neo4j数据库: {url}")
            manager = cls(url=url, username=username, password=password)
            print("连接成功，开始获取数据库模式...")
            result = manager.test_schema_retrieval(
                is_enhanced=is_enhanced,
                database=database,
                timeout=timeout
            )
            
            if not result["success"]:
                print(f"测试失败: {result['error']}")
                print("\n提示: 如果数据库是空的或者刚初始化，这些警告可能是正常的。")
                print("您可以尝试先导入一些数据后再测试模式获取功能。")
                
        except Exception as e:
            print(f"模式测试运行失败: {str(e)}")
        finally:
            # 确保Neo4j连接被正确关闭
            Neo4jConnectionManager.close()
            print("Neo4j连接已关闭")
            
# 示例用法
def test_neo4j_schema(url=None, user=None, password=None, database=None):
    """
    提供一个更简单的函数测试Neo4j模式
    """
    import os
    import dotenv
    
    # 尝试加载环境变量
    dotenv.load_dotenv()
    
    # 优先使用参数，否则从环境变量获取
    url = url or os.getenv("NEO4J_URL", "bolt://localhost:7687")
    user = user or os.getenv("NEO4J_USER", "neo4j") 
    password = password or os.getenv("NEO4J_PASSWORD", "your_password")
    database = database or os.getenv("NEO4J_DATABASE", "neo4j")
    
    print(f"使用连接: {url}, 用户: {user}, 数据库: {database}")
    
    try:
        # 执行测试
        Neo4jManager.run_schema_test(
            url=url,
            username=user,
            password=password,
            database=database,
            is_enhanced=True
        )
    except Exception as e:
        print(f"测试过程中出现错误: {str(e)}")
        print("\n==== 错误调试信息 ====")
        import traceback
        traceback.print_exc()
        print("\n==== 建议 ====")
        print("1. 确认Neo4j数据库已启动并可访问")
        print("2. 确认用户名和密码正确")
        print("3. 如果数据库是空的，可以先导入一些样本数据")
        print("4. 检查Neo4j版本是否与查询兼容")
        print(f"5. 尝试使用其他连接方式，如 bolt:// 替代 neo4j://")

# 如果作为主程序运行，可以直接执行测试
if __name__ == "__main__":
    import os
    import sys
    import dotenv
    
    # 添加错误处理
    try:
        # 尝试从.env文件加载环境变量
        dotenv.load_dotenv()
        
        # 从环境变量获取连接信息或使用默认值
        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "your_password")
        neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")
        
        print("启动Neo4j模式测试...")
        
        # 检查命令行参数
        if len(sys.argv) > 1:
            if sys.argv[1] == "--debug":
                print(f"调试模式: 将使用bolt://协议连接")
                # 在调试模式下使用bolt://协议
                if neo4j_url.startswith("neo4j://"):
                    neo4j_url = neo4j_url.replace("neo4j://", "bolt://")
                    print(f"已修改连接URL为: {neo4j_url}")
            elif sys.argv[1] == "--help":
                print("用法: python neo4j_manager.py [选项]")
                print("选项:")
                print("  --debug     使用bolt://协议连接")
                print("  --help      显示此帮助信息")
                print("  --empty-ok  忽略空数据库警告")
                sys.exit(0)
        
        test_neo4j_schema(
            url=neo4j_url,
            user=neo4j_user,
            password=neo4j_password,
            database=neo4j_database
        )
    except ImportError as ie:
        print(f"导入错误: {str(ie)}")
        print("请确保已安装所有依赖: pip install python-dotenv neo4j langchain-neo4j")
    except Exception as e:
        print(f"程序运行出错: {str(e)}")
        print("请查看上方错误信息并进行修复")
