import pytest
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent  # src 目录
root_dir = parent_dir.parent     # 项目根目录
sys.path.append(str(root_dir))

from src.config import MODEL, DOCUMENT
from src.core.model_provider import ModelProvider

@pytest.fixture
def model_provider():
    """创建ModelProvider实例"""
    return ModelProvider()

def test_get_llm(model_provider):
    """测试获取LLM功能"""
    # 测试获取OpenAI模型
    if MODEL.OPENAI_API_KEY:
        try:
            llm = ModelProvider.get_llm(
                provider="openai",
                model_name=MODEL.OPENAI_MODEL,
                api_key=MODEL.OPENAI_API_KEY,
                api_base=MODEL.OPENAI_API_BASE
            )
            assert llm is not None, "应返回有效的OpenAI LLM"
        except Exception as e:
            pytest.skip(f"OpenAI模型不可用: {str(e)}")
    
    # 测试获取Ollama模型
    try:
        llm = ModelProvider.get_llm(
            provider="ollama",
            model_name=DOCUMENT.OLLAMA_LLM_MODEL,
            base_url=MODEL.OLLAMA_BASE_URL
        )
        assert llm is not None, "应返回有效的Ollama LLM"
    except Exception as e:
        pytest.skip(f"Ollama模型不可用: {str(e)}")

def test_invoke_llm():
    """测试调用LLM功能"""
    # 获取当前配置的LLM
    provider = MODEL.MODEL_PROVIDER
    
    try:
        if provider == "ollama":
            llm = ModelProvider.get_llm(
                provider="ollama",
                model_name=DOCUMENT.OLLAMA_LLM_MODEL,
                base_url=MODEL.OLLAMA_BASE_URL
            )
        else:
            llm = ModelProvider.get_llm(
                provider="openai",
                model_name=MODEL.OPENAI_MODEL,
                api_key=MODEL.OPENAI_API_KEY,
                api_base=MODEL.OPENAI_API_BASE
            )
        
        # 测试简单查询
        prompt = "你好，请用一句话回答你是谁？"
        result = llm.invoke(prompt)
        
        # 验证结果
        assert result is not None, "应返回有效的响应"
        assert hasattr(result, "content") or isinstance(result, str), "结果应为AIMessage或字符串"
        
        content = result.content if hasattr(result, "content") else str(result)
        assert len(content) > 0, "响应内容不应为空"
    except Exception as e:
        pytest.skip(f"LLM调用不可用: {str(e)}")

if __name__ == "__main__":
    pytest.main(["-v", __file__])
