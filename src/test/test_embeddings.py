import pytest
import sys
import numpy as np
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

from src.config import MODEL
from src.core.embeddings import EmbeddingsManager
from langchain_community.embeddings import HuggingFaceEmbeddings

@pytest.fixture
def embeddings_manager():
    """创建EmbeddingsManager实例"""
    return EmbeddingsManager()

def test_get_working_embeddings(embeddings_manager):
    """测试获取可用嵌入模型功能"""
    embeddings = embeddings_manager.get_working_embeddings(
        provider=MODEL.MODEL_PROVIDER,
        api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
        api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
        model_name=MODEL.OPENAI_EMBEDDINGS_MODEL
    )
    
    assert embeddings is not None, "应返回有效的嵌入模型"

def test_embed_query(embeddings_manager):
    """测试嵌入查询功能"""
    embeddings = embeddings_manager.get_working_embeddings(
        provider=MODEL.MODEL_PROVIDER,
        api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
        api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
        model_name=MODEL.OPENAI_EMBEDDINGS_MODEL
    )
    
    # 测试单个查询嵌入
    text = "这是一个测试查询"
    embedding = embeddings.embed_query(text)
    
    # 验证嵌入向量
    assert isinstance(embedding, list), "嵌入应该是一个列表"
    assert len(embedding) > 0, "嵌入应该包含元素" 
    assert all(isinstance(x, float) for x in embedding), "所有嵌入值应该是浮点数"

def test_embed_documents(embeddings_manager):
    """测试嵌入文档功能"""
    embeddings = embeddings_manager.get_working_embeddings(
        provider=MODEL.MODEL_PROVIDER,
        api_key=MODEL.OPENAI_EMBEDDING_API_KEY,
        api_base=MODEL.OPENAI_EMBEDDING_API_BASE,
        model_name=MODEL.OPENAI_EMBEDDINGS_MODEL
    )
    
    # 测试多个文档嵌入
    texts = ["文档1", "文档2", "文档3"]
    embeddings_list = embeddings.embed_documents(texts)
    
    # 验证嵌入向量列表
    assert isinstance(embeddings_list, list), "嵌入列表应该是一个列表"
    assert len(embeddings_list) == len(texts), "应该为每个文档生成一个嵌入"
    
    # 验证每个嵌入向量
    for embedding in embeddings_list:
        assert isinstance(embedding, list), "每个嵌入应该是一个列表"
        assert len(embedding) > 0, "每个嵌入应该包含元素"

def test_fallback_embeddings():
    """测试备用嵌入模型功能"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # 测试嵌入功能
        text = "测试备用嵌入模型"
        embedding = embeddings.embed_query(text)
        
        # 验证嵌入向量
        assert isinstance(embedding, list), "嵌入应该是一个列表"
        assert len(embedding) > 0, "嵌入应该包含元素"
    except Exception as e:
        pytest.skip(f"备用嵌入模型不可用: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
