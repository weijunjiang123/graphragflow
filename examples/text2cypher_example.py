"""
Text2Cypher模块使用示例

此脚本演示如何使用text2cypher模块将自然语言转换为Cypher查询并执行
"""
import logging
import sys
from pathlib import Path
import time

# 添加项目根目录到Python路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

# 导入必要的模块
from src.core.text2cypher import GraphRetriever
from src.config import DATABASE, MODEL

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """主函数: 演示text2cypher模块的使用"""
    # 初始化GraphRetriever
    logger.info("初始化GraphRetriever...")
    retriever = GraphRetriever(
        connection_uri=DATABASE.URI,
        username=DATABASE.USERNAME,
        password=DATABASE.PASSWORD,
        database_name=DATABASE.DATABASE_NAME,
        top_k=10,
        return_direct=False,
        verbose=True
    )
    
    # 获取并打印图数据库模式
    schema = retriever.get_schema()
    logger.info(f"图数据库模式:\n{schema}")
    
    # 示例查询
    example_queries = [
        "列出所有Person节点及其名称",
        "找出所有年龄大于30的Person",
        "谁认识最多的人?",
        "查找所有具有'admin'角色的用户"
    ]
    
    # 执行查询
    for query in example_queries:
        logger.info(f"\n执行查询: '{query}'")
        start_time = time.time()
        result = retriever.query(query)
        
        # 打印结果
        logger.info(f"原始查询: {result['query']}")
        logger.info(f"生成的Cypher: {result['generated_cypher']}")
        logger.info(f"查询结果: {result['result']}")
        logger.info(f"耗时: {result['metadata']['elapsed_time']:.2f}秒")
        
        # 休息一下，避免请求过快
        time.sleep(1)
    
    # 直接执行Cypher查询示例
    cypher_query = "MATCH (n:Person) RETURN n.name AS name, n.age AS age LIMIT 5"
    logger.info(f"\n直接执行Cypher: '{cypher_query}'")
    results = retriever.execute_cypher(cypher_query)
    logger.info(f"结果: {results}")

if __name__ == "__main__":
    main()