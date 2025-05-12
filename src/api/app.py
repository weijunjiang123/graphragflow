"""
GraphRAG API服务 - 主应用文件
提供基于FastAPI的后端服务，采用RESTful规范
"""
import logging
import os
import sys
import time
from typing import List, Dict, Any
from fastapi import FastAPI, Request, Response, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRouter
from fastapi.openapi.utils import get_openapi
from starlette.responses import RedirectResponse

# 导入中间件和工具
from src.api.middleware import LoggingMiddleware, RateLimiterMiddleware
from src.api.versioning import APIVersioning, VersionedAPIRoute
from src.api.cache import text2cypher_cache, example_cache
from src.api.monitoring import performance_monitor, PerformanceMetric

# 导入路由
from src.api.routes.text2cypher import router as text2cypher_router
from src.api.routes.search import router as search_router
from src.api.routes.examples import router as examples_router

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join("logs", "api.log"), encoding="utf-8"),
    ]
)
logger = logging.getLogger(__name__)

# 创建FastAPI应用
app = FastAPI(
    title="GraphRAG API",
    description="图检索和text2Cypher查询服务",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    default_response_class=JSONResponse,
    # 使用自定义的路由类，支持版本控制
    route_class=VersionedAPIRoute
)

# 添加中间件
app.add_middleware(LoggingMiddleware)  # 日志中间件
app.add_middleware(
    RateLimiterMiddleware,
    requests_limit=200,  # 每分钟最多200个请求
    window_seconds=60
)

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有方法
    allow_headers=["*"],  # 允许所有头
)

# 性能监控中间件
@app.middleware("http")
async def performance_monitoring_middleware(request: Request, call_next):
    """收集请求性能指标的中间件"""
    start_time = time.time()
    request.state.start_time = start_time
    
    try:
        response = await call_next(request)
        
        # 计算请求处理时间
        process_time = (time.time() - start_time) * 1000  # 毫秒
        
        # 收集性能指标
        metric = PerformanceMetric(
            endpoint=request.url.path,
            method=request.method,
            status_code=response.status_code,
            response_time=process_time,
            timestamp=start_time,
            request_id=request.state.request_id if hasattr(request.state, "request_id") else None,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None
        )
        performance_monitor.add_metric(metric)
        
        return response
    except Exception as e:
        process_time = (time.time() - start_time) * 1000
        
        # 收集异常指标
        metric = PerformanceMetric(
            endpoint=request.url.path,
            method=request.method,
            status_code=500,
            response_time=process_time,
            timestamp=start_time,
            request_id=request.state.request_id if hasattr(request.state, "request_id") else None,
            user_agent=request.headers.get("User-Agent"),
            ip_address=request.client.host if request.client else None
        )
        performance_monitor.add_metric(metric)
        
        raise

# 自定义OpenAPI文档
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="GraphRAG API",
        version="1.0.0",
        description="图检索和text2Cypher查询服务API文档",
        routes=app.routes,
    )
    
    # 添加API版本信息
    openapi_schema["info"]["x-api-version"] = "v1"
    
    # 添加联系信息
    openapi_schema["info"]["contact"] = {
        "name": "GraphRAG Team",
        "email": "support@graphrag.example.com",
        "url": "https://github.com/your-username/GraphRAG-with-Llama-3.1",
    }
    
    # 添加服务条款
    openapi_schema["info"]["termsOfService"] = "https://graphrag.example.com/terms/"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# 异常处理
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """处理请求验证错误"""
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "请求参数错误",
            "detail": exc.errors(),
            "timestamp": time.time()
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """处理HTTP异常"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """处理通用异常"""
    logger.exception(f"未处理的异常: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "服务器内部错误",
            "detail": str(exc),
            "timestamp": time.time()
        }
    )

# 根路由重定向到API文档
@app.get("/", include_in_schema=False)
async def root_redirect():
    """将根路径重定向到API文档"""
    return RedirectResponse(url="/api/docs")

# API根路径
@app.get("/api", tags=["状态"])
async def api_root():
    """API根路径，返回服务状态"""
    return {
        "success": True,
        "status": "online",
        "service": "GraphRAG API",
        "version": app.version,
        "timestamp": time.time()
    }

# 健康检查
@app.get("/api/health", tags=["状态"])
async def health_check():
    """API健康检查"""
    return {
        "success": True,
        "status": "healthy",
        "version": app.version,
        "timestamp": time.time(),
        "cache": {
            "text2cypher": len(text2cypher_cache.cache),
            "examples": len(example_cache.cache)
        }
    }

# 缓存控制
@app.post("/api/cache/clear", tags=["管理"])
async def clear_cache():
    """清空所有缓存"""
    text2cypher_cache.clear()
    example_cache.clear()
    return {
        "success": True,
        "message": "缓存已清空",
        "timestamp": time.time()
    }

# 性能监控端点
@app.get("/api/metrics", tags=["管理"])
async def get_metrics(endpoint: str = None, method: str = None, time_range: int = None):
    """
    获取API性能指标
    
    参数:
    - endpoint: 过滤特定端点
    - method: 过滤特定方法
    - time_range: 时间范围（秒），如过去60秒
    """
    stats = performance_monitor.get_stats(endpoint, method, time_range)
    return {
        "success": True,
        "metrics": [stat.dict() for stat in stats],
        "timestamp": time.time()
    }

@app.post("/api/metrics/clear", tags=["管理"])
async def clear_metrics():
    """清空所有性能指标"""
    performance_monitor.clear()
    return {
        "success": True,
        "message": "性能指标已清空",
        "timestamp": time.time()
    }

# 注册路由，添加版本前缀
text2cypher_router.prefix = "/api/v1"
search_router.prefix = "/api/v1"
examples_router.prefix = "/api/v1"

app.include_router(text2cypher_router, tags=["text2cypher"])
app.include_router(search_router, tags=["search"])
app.include_router(examples_router, tags=["examples"])

# 兼容性路由 - 不带版本前缀，保持向后兼容
app.include_router(text2cypher_router, prefix="/api", tags=["text2cypher-compat"])
app.include_router(search_router, prefix="/api", tags=["search-compat"])
app.include_router(examples_router, prefix="/api", tags=["examples-compat"])

# 启动事件
@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    logger.info(f"GraphRAG API服务启动, 版本: {app.version}")

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时执行"""
    logger.info("GraphRAG API服务关闭")
