"""
GraphRAG API - 工具类
"""
import logging
from fastapi import Request
from typing import Dict, Any

logger = logging.getLogger(__name__)

class APILogger:
    """API日志记录工具"""
    
    @staticmethod
    async def log_request(request: Request) -> None:
        """记录请求信息"""
        body = await request.body()
        logger.info(f"请求: {request.method} {request.url.path}")
        logger.debug(f"请求体: {body.decode() if body else 'empty'}")
        
    @staticmethod
    def log_response(response: Dict[str, Any]) -> None:
        """记录响应信息"""
        logger.debug(f"响应: {response}")

class ErrorHandler:
    """错误处理工具"""
    
    @staticmethod
    def format_error(status_code: int, detail: str) -> Dict[str, Any]:
        """格式化错误响应"""
        return {
            "success": False,
            "status_code": status_code,
            "detail": detail
        }
