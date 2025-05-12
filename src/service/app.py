"""
GraphRAG API服务 - 入口文件
提供基于FastAPI的后端服务，用于处理图检索和text2Cypher查询
"""
import logging
import os
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# 导入服务模块
from src.service.text2cypher_service import Text2CypherService
from src.service.dependencies import get_graph_retriever

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG API",
    description="图检索和text2Cypher查询服务",
    version="1.0.0"
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 请求和响应模型
class Text2CypherRequest(BaseModel):
    query: str = Field(..., description="自然语言查询")
    limit: int = Field(5, description="结果数量限制")

class Text2CypherResponse(BaseModel):
    original_query: str = Field(..., description="原始查询")
    cypher_query: str = Field(..., description="生成的Cypher查询")
    results: List[Dict[str, Any]] = Field(..., description="查询结果")
    result_count: int = Field(..., description="结果数量")
    explanation: Optional[str] = Field(None, description="结果解释")
    fallback_method: Optional[str] = Field(None, description="回退方法")

class SearchRequest(BaseModel):
    query: str = Field(..., description="搜索查询")
    limit: int = Field(5, description="结果数量限制")
    search_type: str = Field("text2cypher", description="搜索类型: text2cypher, hybrid, vector, fulltext")

# 路由
@app.get("/")
async def root():
    """API根路径，返回服务状态"""
    return {"status": "online", "service": "GraphRAG API"}

@app.post("/api/text2cypher", response_model=Text2CypherResponse)
async def text2cypher(
    request: Text2CypherRequest,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    使用text2Cypher进行搜索
    """
    try:
        service = Text2CypherService(graph_retriever)
        result = service.search(request.query, request.limit)
        return result
    except Exception as e:
        logger.error(f"text2Cypher搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.post("/api/search")
async def search(
    request: SearchRequest,
    graph_retriever = Depends(get_graph_retriever)
):
    """
    统一搜索接口，支持多种搜索类型
    """
    try:
        service = Text2CypherService(graph_retriever)
        
        if request.search_type == "text2cypher":
            return service.search(request.query, request.limit)
        elif request.search_type == "hybrid":
            results = graph_retriever.hybrid_search(request.query, request.limit)
            return {
                "original_query": request.query,
                "search_type": "hybrid",
                "results": results,
                "result_count": len(results)
            }
        elif request.search_type == "vector":
            results = graph_retriever.vector_search(request.query, request.limit)
            return {
                "original_query": request.query,
                "search_type": "vector",
                "results": [doc.dict() for doc in results] if hasattr(results[0], "dict") else results,
                "result_count": len(results)
            }
        elif request.search_type == "fulltext":
            results = graph_retriever.fulltext_search(request.query, request.limit)
            return {
                "original_query": request.query,
                "search_type": "fulltext",
                "results": results,
                "result_count": len(results)
            }
        else:
            raise HTTPException(status_code=400, detail=f"不支持的搜索类型: {request.search_type}")
    except Exception as e:
        logger.error(f"搜索失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"搜索失败: {str(e)}")

@app.get("/api/examples")
async def get_examples(graph_retriever = Depends(get_graph_retriever)):
    """获取text2Cypher示例查询"""
    try:
        return graph_retriever.get_text2cypher_examples()
    except Exception as e:
        logger.error(f"获取示例查询失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取示例查询失败: {str(e)}")

# 测试API服务
@app.get("/api/test")
async def test_api(graph_retriever = Depends(get_graph_retriever)):
    """测试API服务状态和text2Cypher功能"""
    try:
        test_result = graph_retriever.test_text2cypher_capability()
        return {
            "api_status": "online",
            "text2cypher_supported": test_result["supports_text2cypher"],
            "test_result": test_result
        }
    except Exception as e:
        logger.error(f"测试API失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"测试失败: {str(e)}")

# 启动服务
if __name__ == "__main__":
    # 读取环境变量或使用默认值
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    # 启动Uvicorn服务器
    uvicorn.run("src.service.app:app", host=host, port=port, reload=True)
