import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from models.node import Node
from models.relationship import Relationship

# 配置日志
logger = logging.getLogger(__name__)

class Neo4jImporter:
    """处理向 Neo4j 数据库导入节点和关系。"""
    
    def __init__(self, uri: str, username: str, password: str):
        """初始化 Neo4j 连接。
        
        Args:
            uri: Neo4j 连接 URI
            username: Neo4j 用户名
            password: Neo4j 密码
            
        Raises:
            ConnectionError: 无法连接到 Neo4j 数据库时抛出
            AuthError: 认证失败时抛出
        """
        try:
            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            # 验证连接
            self.driver.verify_connectivity()
            print(f"✅ 已连接到 Neo4j 数据库 ({uri})")
            logger.info(f"已连接到 Neo4j 数据库: {uri}")
        except AuthError as e:
            error_msg = f"Neo4j 认证失败: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print("\n可能的解决方案:")
            print("1. 检查用户名和密码是否正确")
            print("2. 确保 .env 文件中包含正确的凭据")
            print("3. 如果是首次使用，可能需要通过 Neo4j Browser 重置密码")
            raise
        except ServiceUnavailable as e:
            error_msg = f"Neo4j 服务不可用: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            print("\n可能的解决方案:")
            print("1. 确保 Neo4j 数据库正在运行")
            print("2. 检查数据库 URI 是否正确 (例如: neo4j://localhost:7687)")
            print("3. 验证防火墙设置是否允许连接到该端口")
            raise ConnectionError(error_msg)
        except Exception as e:
            error_msg = f"连接到 Neo4j 时出错: {str(e)}"
            logger.error(error_msg)
            print(f"❌ {error_msg}")
            raise
        
    def close(self):
        """关闭 Neo4j 连接。"""
        if hasattr(self, 'driver'):
            self.driver.close()
            logger.info("Neo4j 连接已关闭")
            print("📌 Neo4j 连接已关闭")
        
    def import_nodes(self, nodes: List[Node]) -> Dict[str, int]:
        """导入节点到 Neo4j。
        
        Args:
            nodes: Node 对象列表
            
        Returns:
            包含导入统计信息的字典
        """
        if not nodes:
            logger.warning("没有节点可导入")
            return {"imported": 0, "failed": 0}
            
        total = len(nodes)
        print(f"📥 正在导入 {total} 个节点...")
        logger.info(f"开始导入 {total} 个节点")
        
        # 导入统计
        imported = 0
        failed = 0
        
        # 处理批次以提高性能
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = nodes[i:i+batch_size]
            batch_stats = self._import_node_batch(batch)
            imported += batch_stats["imported"]
            failed += batch_stats["failed"]
            
            # 打印进度
            progress = min(i + batch_size, total)
            percent = (progress / total) * 100
            print(f"\r导入进度: {progress}/{total} 节点 ({percent:.1f}%) - 成功: {imported}, 失败: {failed}", end="")
            
        print(f"\n✅ 节点导入完成 - 成功: {imported}, 失败: {failed}")
        logger.info(f"节点导入完成 - 成功: {imported}, 失败: {failed}")
        return {"imported": imported, "failed": failed}
            
    def _import_node_batch(self, nodes: List[Node]) -> Dict[str, int]:
        """导入一批节点。
        
        Args:
            nodes: 一批 Node 对象
            
        Returns:
            包含导入统计信息的字典
        """
        if not nodes:
            return {"imported": 0, "failed": 0}
        
        imported = 0
        failed = 0
            
        with self.driver.session() as session:
            # 处理批次中的每个节点
            for node in nodes:
                try:
                    # 确保使用正确的节点类型属性名
                    # 注意：这里假设 Node 类使用 type 而不是 label
                    node_type = getattr(node, 'type', None) or getattr(node, 'label', 'Entity')
                    
                    # 过滤掉空值属性
                    properties = {k: v for k, v in node.properties.items() if v is not None}
                    
                    # Cypher 查询以合并节点 - 防止重复
                    query = f"""
                    MERGE (n:{node_type} {{id: $id}})
                    SET n += $properties
                    RETURN n.id
                    """
                    
                    result = session.run(query, id=node.id, properties=properties)
                    if result.single():
                        imported += 1
                    else:
                        logger.warning(f"节点 {node.id} 导入未返回结果")
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"导入节点 {node.id} 出错: {str(e)}")
                    failed += 1
                    
        return {"imported": imported, "failed": failed}
    
    def import_relationships(self, relationships: List[Relationship]) -> Dict[str, int]:
        """导入关系到 Neo4j。
        
        Args:
            relationships: Relationship 对象列表
            
        Returns:
            包含导入统计信息的字典
        """
        if not relationships:
            logger.warning("没有关系可导入")
            return {"imported": 0, "failed": 0}
            
        total = len(relationships)
        print(f"📥 正在导入 {total} 个关系...")
        logger.info(f"开始导入 {total} 个关系")
        
        # 导入统计
        imported = 0
        failed = 0
        
        # 处理批次以提高性能
        batch_size = 100
        for i in range(0, total, batch_size):
            batch = relationships[i:i+batch_size]
            batch_stats = self._import_relationship_batch(batch)
            imported += batch_stats["imported"]
            failed += batch_stats["failed"]
            
            # 打印进度
            progress = min(i + batch_size, total)
            percent = (progress / total) * 100
            print(f"\r导入进度: {progress}/{total} 关系 ({percent:.1f}%) - 成功: {imported}, 失败: {failed}", end="")
            
        print(f"\n✅ 关系导入完成 - 成功: {imported}, 失败: {failed}")
        logger.info(f"关系导入完成 - 成功: {imported}, 失败: {failed}")
        return {"imported": imported, "failed": failed}
            
    def _import_relationship_batch(self, relationships: List[Relationship]) -> Dict[str, int]:
        """导入一批关系。
        
        Args:
            relationships: 一批 Relationship 对象
            
        Returns:
            包含导入统计信息的字典
        """
        if not relationships:
            return {"imported": 0, "failed": 0}
            
        imported = 0
        failed = 0
            
        with self.driver.session() as session:
            # 处理批次中的每个关系
            for rel in relationships:
                try:
                    # 确保使用正确的属性名
                    # 注意：这里处理两种可能的命名规范
                    source_id = getattr(rel, 'source', None) or getattr(rel, 'source_id', '')
                    target_id = getattr(rel, 'target', None) or getattr(rel, 'target_id', '')
                    
                    if not source_id or not target_id:
                        logger.warning(f"关系缺少源节点或目标节点ID: {rel}")
                        failed += 1
                        continue
                    
                    # 过滤掉空值属性
                    properties = {k: v for k, v in rel.properties.items() if v is not None}
                    
                    # Cypher 查询以合并关系 - 需要已存在的节点
                    query = f"""
                    MATCH (source {{id: $source_id}})
                    MATCH (target {{id: $target_id}})
                    MERGE (source)-[r:{rel.type}]->(target)
                    SET r += $properties
                    RETURN type(r)
                    """
                    
                    result = session.run(
                        query, 
                        source_id=source_id, 
                        target_id=target_id, 
                        properties=properties
                    )
                    
                    if result.single():
                        imported += 1
                    else:
                        logger.warning(f"关系导入未返回结果: {source_id}-[{rel.type}]->{target_id}")
                        failed += 1
                        
                except Exception as e:
                    logger.error(f"导入关系出错: {str(e)}")
                    failed += 1
                    
        return {"imported": imported, "failed": failed}
        
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行自定义 Cypher 查询。
        
        Args:
            query: Cypher 查询字符串
            params: 查询参数
            
        Returns:
            查询结果列表
        """
        params = params or {}
        results = []
        
        with self.driver.session() as session:
            try:
                result = session.run(query, **params)
                results = [record.data() for record in result]
                logger.debug(f"执行查询成功: {len(results)} 条结果")
                return results
            except Exception as e:
                logger.error(f"执行查询出错: {str(e)}")
                raise