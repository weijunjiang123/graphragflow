"""
GraphRAG API - 数据模型
定义API的请求和响应模型
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 通用模型
class BaseResponse(BaseModel):
    """基础响应模型"""
    success: bool = Field(True, description="是否成功")
    message: Optional[str] = Field(None, description="消息")

# Text2Cypher模型
class Text2CypherRequest(BaseModel):
    """Text2Cypher请求模型"""
    query: str = Field(..., description="自然语言查询")
    limit: int = Field(5, description="结果数量限制")

class Text2CypherResponse(BaseResponse):
    """Text2Cypher响应模型"""
    original_query: str = Field(..., description="原始查询")
    cypher_query: str = Field(..., description="生成的Cypher查询")
    results: List[Dict[str, Any]] = Field(..., description="查询结果")
    result_count: int = Field(..., description="结果数量")
    explanation: Optional[str] = Field(None, description="结果解释")
    fallback_method: Optional[str] = Field(None, description="回退方法")

# 搜索模型
class SearchRequest(BaseModel):
    """搜索请求模型"""
    query: str = Field(..., description="搜索查询")
    limit: int = Field(5, description="结果数量限制")
    search_type: str = Field("text2cypher", description="搜索类型: text2cypher, hybrid, vector, fulltext")

class SearchResponse(BaseResponse):
    """搜索响应模型"""
    original_query: str = Field(..., description="原始查询")
    search_type: str = Field(..., description="搜索类型")
    results: List[Dict[str, Any]] = Field(..., description="查询结果")
    result_count: int = Field(..., description="结果数量")
    cypher_query: Optional[str] = Field(None, description="生成的Cypher查询")
    explanation: Optional[str] = Field(None, description="结果解释")
    fallback_method: Optional[str] = Field(None, description="回退方法")

# 示例查询模型
class ExampleQuery(BaseModel):
    """示例查询模型"""
    description: str = Field(..., description="示例描述")
    query: str = Field(..., description="示例查询")
    explanation: str = Field(..., description="示例解释")

# 测试响应模型
class TestResponse(BaseResponse):
    """测试响应模型"""
    api_status: str = Field(..., description="API状态")
    text2cypher_supported: bool = Field(..., description="是否支持text2Cypher")
    test_result: Dict[str, Any] = Field(..., description="测试结果")
