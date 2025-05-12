"""
GraphRAG API - 中间件模块
实现各类中间件，如日志记录、性能跟踪等
"""
import time
import logging
import uuid
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    请求日志中间件
    记录每个请求的处理时间、方法、路径和状态码
    """
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
        
    async def dispatch(self, request: Request, call_next):
        """处理请求并记录日志"""
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # 记录请求开始
        start_time = time.time()
        logger.info(f"[{request_id}] 开始处理请求: {request.method} {request.url.path}")
        
        # 获取客户端IP
        client_host = request.client.host if request.client else "unknown"
        
        try:
            # 处理请求
            response = await call_next(request)
            
            # 计算处理时间
            process_time = (time.time() - start_time) * 1000
            response.headers["X-Process-Time"] = f"{process_time:.2f}ms"
            response.headers["X-Request-ID"] = request_id
            
            # 记录请求结束
            logger.info(
                f"[{request_id}] 完成请求: {request.method} {request.url.path} "
                f"状态: {response.status_code} 处理时间: {process_time:.2f}ms "
                f"客户端: {client_host}"
            )
            
            return response
            
        except Exception as e:
            # 请求处理过程中出现异常
            process_time = (time.time() - start_time) * 1000
            logger.error(
                f"[{request_id}] 请求异常: {request.method} {request.url.path} "
                f"错误: {str(e)} 处理时间: {process_time:.2f}ms "
                f"客户端: {client_host}"
            )
            raise

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    请求限流中间件
    限制单个IP的请求频率
    """
    
    def __init__(self, app: ASGIApp, requests_limit: int = 100, window_seconds: int = 60):
        """
        初始化限流中间件
        
        Args:
            app: ASGI应用
            requests_limit: 时间窗口内的最大请求数
            window_seconds: 时间窗口大小（秒）
        """
        super().__init__(app)
        self.requests_limit = requests_limit
        self.window_seconds = window_seconds
        self.requests = {}  # 记录请求次数的字典 {ip: [(timestamp, request_path), ...]}
        
    async def dispatch(self, request: Request, call_next):
        """处理请求并进行限流"""
        # 获取客户端IP
        client_ip = request.client.host if request.client else "unknown"
        current_time = time.time()
        
        # 清理过期的请求记录
        if client_ip in self.requests:
            self.requests[client_ip] = [
                (ts, path) for ts, path in self.requests[client_ip] 
                if current_time - ts < self.window_seconds
            ]
        else:
            self.requests[client_ip] = []
        
        # 检查是否超过限制
        if len(self.requests[client_ip]) >= self.requests_limit:
            logger.warning(f"限流: IP {client_ip} 超过请求限制 {self.requests_limit}/{self.window_seconds}s")
            return Response(
                content="请求频率过高，请稍后再试",
                status_code=429,
                media_type="text/plain"
            )
        
        # 记录当前请求
        self.requests[client_ip].append((current_time, request.url.path))
        
        # 处理请求
        return await call_next(request)