from src.core.text2cypher.graph_retriever import GraphRetriever

# Simple wrapper service for Text2Cypher functionality
class Text2CypherService:
    """Text2Cypher服务 - 将自然语言转换为Cypher查询"""
    
    def __init__(self, uri: str, user: str, password: str, database: str, llm=None):
        """初始化Text2Cypher服务
        
        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
            database: 数据库名
            llm: 语言模型
        """
        self.retriever = GraphRetriever(
            connection_uri=uri,
            username=user,
            password=password,
            database_name=database,
            llm=llm
        )
    def query(self, question: str):
        """执行自然语言查询
        
        Args:
            question: 自然语言问题
            
        Returns:
            查询结果
        """
        return self.retriever.query(question)
    def execute_cypher(self, cypher_query: str):
        """直接执行Cypher查询
        
        Args:
            cypher_query: Cypher查询语句
            
        Returns:
            查询结果列表
        """
        return self.retriever.execute_cypher(cypher_query)
    
    def get_schema(self) -> str:
        """获取图数据库模式
        
        Returns:
            格式化的数据库模式字符串
        """
        return self.retriever.get_schema()
    
    def close(self):
        """关闭连接"""
        if hasattr(self, "retriever") and self.retriever:
            try:
                self.retriever.graph.close()
            except Exception:
                pass

__all__ = [
    "GraphRetriever",
    "Text2CypherService",
]