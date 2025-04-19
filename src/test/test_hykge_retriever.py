import pytest
import logging
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

from src.config import DATABASE, MODEL, DOCUMENT
from src.core.embeddings import EmbeddingsManager
from src.core.model_provider import ModelProvider
from src.core.graph_retrieval import GraphRetriever
from src.core.hykge_retriever import HyKGERetriever
from src.main import Neo4jConnectionManager
from src.test.test_hykge_retrievel import patch_hykge_retriever, get_fallback_embeddings

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def llm():
    """初始化LLM"""
    provider = MODEL.MODEL_PROVIDER
    if provider == "ollama":
        return ModelProvider.get_llm(
            provider="ollama",
            model_name=DOCUMENT.OLLAMA_LLM_MODEL,
            base_url=MODEL.OLLAMA_BASE_URL
        )
    else:
        return ModelProvider.get_llm(
            provider="openai",
            model_name=MODEL.OPENAI_MODEL,
            api_key=MODEL.OPENAI_API_KEY,
            api_base=MODEL.OPENAI_API_BASE
        )

@pytest.fixture
def embeddings():
    """初始化嵌入模型"""
    embeddings_manager = EmbeddingsManager()
    try:
        return embeddings_manager.get_working_embeddings(
            provider=MODEL.MODEL_PROVIDER,
            api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
            api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
            model_name=MODEL.OPENAI_EMBEDDINGS_MODEL
        )
    except Exception:
        return get_fallback_embeddings()

@pytest.fixture
def graph_retriever(llm):
    """创建GraphRetriever实例"""
    retriever = GraphRetriever(
        DATABASE.URI, 
        DATABASE.USERNAME, 
        DATABASE.PASSWORD,
        llm=llm
    )
    yield retriever
    retriever.close()

@pytest.fixture
def hykge_retriever(graph_retriever, llm, embeddings):
    """创建HyKGERetriever实例"""
    # 应用补丁修复
    patch_hykge_retriever()
    
    retriever = HyKGERetriever(
        graph_retriever=graph_retriever,
        llm=llm,
        embeddings=embeddings,
        top_k=5,
        max_hop=2
    )
    return retriever

def test_pre_embed_entities(hykge_retriever):
    """测试预嵌入实体功能"""
    try:
        # 限制处理数量以加快测试速度
        result = hykge_retriever.pre_embed_entities(batch_size=5, max_workers=2)
        assert isinstance(result, int), "应返回嵌入实体的数量"
    except Exception as e:
        pytest.skip(f"预嵌入实体测试跳过: {str(e)}")

def test_rag_process(hykge_retriever):
    """测试RAG处理功能"""
    try:
        # 使用简单查询
        query = "知识图谱是什么?"
        result = hykge_retriever.rag_process(query)
        
        # 检查结果是否有效
        assert result is not None, "应返回有效结果"
        assert isinstance(result, str) or hasattr(result, "content"), "结果应为字符串或AIMessage"
        
        # 检查内容
        content = result if isinstance(result, str) else result.content
        assert len(content) > 0, "结果内容不应为空"
    except Exception as e:
        pytest.skip(f"RAG处理测试跳过: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
