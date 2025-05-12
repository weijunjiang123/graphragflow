"""
GraphRAG API - 版本控制模块
提供API版本控制功能
"""
import re
from typing import Dict, Any, Optional, Callable
from fastapi import Request, Response, Depends
from fastapi.routing import APIRoute

class VersionedAPIRoute(APIRoute):
    """
    版本化的API路由
    支持在路径中包含版本号，如：/api/v1/resource
    """
    
    def __init__(self, *args, **kwargs):
        self.version: Optional[str] = kwargs.pop("version", None)
        super().__init__(*args, **kwargs)
        
        # 如果提供了版本，修改路径
        if self.version:
            self.path = self._add_version_to_path(self.path, self.version)
    
    def _add_version_to_path(self, path: str, version: str) -> str:
        """向路径添加版本前缀"""
        # 检查是否已经有版本前缀
        if re.match(r'^/api/v\d+', path):
            return path
            
        # 向路径添加版本
        if path.startswith('/api/'):
            return path.replace('/api/', f'/api/{version}/')
        elif path.startswith('/'):
            return f'/api/{version}{path}'
        else:
            return f'/api/{version}/{path}'


class APIVersioning:
    """API版本控制工具"""
    
    @staticmethod
    def version_header(
        request: Request,
        default_version: str = "v1"
    ) -> str:
        """
        从请求头获取API版本
        
        Args:
            request: FastAPI请求对象
            default_version: 默认版本
            
        Returns:
            API版本，如：v1、v2等
        """
        return request.headers.get("X-API-Version", default_version)
    
    @staticmethod
    def version_response(response: Response, version: str) -> Response:
        """
        向响应头添加API版本
        
        Args:
            response: FastAPI响应对象
            version: API版本
            
        Returns:
            修改后的响应对象
        """
        response.headers["X-API-Version"] = version
        return response