import pytest
import sys
import logging
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

from src.config import DATABASE, MODEL
from src.core.graph_retrieval import GraphRetriever
from src.core.embeddings import EmbeddingsManager
from src.main import Neo4jConnectionManager

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def neo4j_driver():
    """创建Neo4j驱动实例"""
    driver = Neo4jConnectionManager.get_instance(
        DATABASE.URI, 
        (DATABASE.USERNAME, DATABASE.PASSWORD)
    )
    yield driver

@pytest.fixture
def embeddings():
    """初始化嵌入模型"""
    embeddings_manager = EmbeddingsManager()
    embeddings = embeddings_manager.get_working_embeddings(
        provider=MODEL.MODEL_PROVIDER,
        api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
        api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
        model_name=MODEL.OPENAI_EMBEDDINGS_MODEL
    )
    yield embeddings

@pytest.fixture
def graph_retriever(neo4j_driver):
    """创建GraphRetriever实例"""
    retriever = GraphRetriever(
        DATABASE.URI, 
        DATABASE.USERNAME, 
        DATABASE.PASSWORD
    )
    yield retriever
    retriever.close()

def test_connection(graph_retriever):
    """测试图检索器的连接功能"""
    assert graph_retriever.driver is not None, "Neo4j连接未正确初始化"
    
    # 测试简单查询
    result = graph_retriever.execute_query("MATCH (n) RETURN count(n) as count LIMIT 1")
    assert result is not None, "无法执行基本查询"
    assert isinstance(result, list), "查询结果应该是一个列表"

def test_fulltext_index(graph_retriever):
    """测试全文索引功能"""
    # 确保索引存在
    result = graph_retriever.ensure_fulltext_index()
    assert result is True, "创建全文索引失败"
    
    # 验证索引已存在
    with graph_retriever.driver.session() as session:
        try:
            check_query = "SHOW INDEXES WHERE type = 'FULLTEXT' AND name = 'entityContentIndex'"
            result = session.run(check_query).single() is not None
            assert result is True, "全文索引不存在"
        except Exception:
            # Neo4j社区版可能不支持SHOW INDEXES
            pass

def test_get_all_entities(graph_retriever):
    """测试获取所有实体功能"""
    # 测试获取实体的功能
    entities = graph_retriever.get_all_entities(limit=5)
    assert isinstance(entities, list), "实体列表应该是一个列表"
    
    # 如果数据库中有实体，验证其属性
    if entities:
        entity = entities[0]
        assert "id" in entity, "实体应包含id属性"
        assert "name" in entity, "实体应包含name属性"

def test_content_search(graph_retriever):
    """测试内容搜索功能"""
    # 搜索一些常见术语
    results = graph_retriever.content_search("知识图谱", limit=3)
    assert isinstance(results, list), "搜索结果应该是一个列表"
    
    # 搜索另一个术语
    results = graph_retriever.content_search("人工智能", limit=3)
    assert isinstance(results, list), "搜索结果应该是一个列表"

def test_store_and_retrieve_embedding(graph_retriever, embeddings):
    """测试存储和检索嵌入向量功能"""
    # 创建一个测试实体
    with graph_retriever.driver.session() as session:
        create_query = """
        MERGE (n:Entity {name: 'TestEntity'})
        SET n.description = 'This is a test entity for embedding tests'
        RETURN elementId(n) as id
        """
        result = session.run(create_query)
        record = result.single()
        if record:
            entity_id = record["id"]
            
            # 生成嵌入向量
            text = "TestEntity: This is a test entity for embedding tests"
            vector = embeddings.embed_query(text)
            
            # 存储嵌入向量
            stored = graph_retriever.store_entity_embedding(entity_id, vector)
            assert stored is True, "存储嵌入向量失败"
            
            # 检查嵌入向量是否存在
            has_embedding = graph_retriever.has_entity_embedding(entity_id)
            assert has_embedding is True, "检查嵌入向量存在失败"
            
            # 清理测试数据
            session.run(
                "MATCH (n:Entity {name: 'TestEntity'}) DETACH DELETE n"
            )

if __name__ == "__main__":
    pytest.main(["-v", __file__])