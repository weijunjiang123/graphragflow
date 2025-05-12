"""
GraphRAG API - 响应格式化模块
提供统一的API响应格式
"""
import time
from typing import Dict, List, Any, Optional, Union
from fastapi.responses import JSONResponse

class APIResponse:
    """
    API响应工具类
    提供统一的响应格式
    """
    
    @staticmethod
    def success(data: Any = None, message: str = "操作成功", 
                meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成成功响应
        
        Args:
            data: 响应数据
            message: 响应消息
            meta: 元数据，如分页信息
            
        Returns:
            格式化的响应字典
        """
        response = {
            "success": True,
            "message": message,
            "timestamp": time.time()
        }
        
        if data is not None:
            response["data"] = data
            
        if meta is not None:
            response["meta"] = meta
            
        return response
    
    @staticmethod
    def error(message: str = "操作失败", 
              detail: Optional[Union[str, List[Dict[str, Any]]]] = None, 
              status_code: int = 400) -> JSONResponse:
        """
        生成错误响应
        
        Args:
            message: 错误消息
            detail: 详细错误信息
            status_code: HTTP状态码
            
        Returns:
            JSONResponse对象
        """
        response = {
            "success": False,
            "message": message,
            "timestamp": time.time()
        }
        
        if detail is not None:
            response["detail"] = detail
            
        return JSONResponse(
            status_code=status_code,
            content=response
        )
    
    @staticmethod
    def pagination(data: List[Any], total: int, page: int, page_size: int) -> Dict[str, Any]:
        """
        生成分页响应
        
        Args:
            data: 分页数据
            total: 总记录数
            page: 当前页码
            page_size: 每页记录数
            
        Returns:
            格式化的分页响应
        """
        return APIResponse.success(
            data=data,
            meta={
                "pagination": {
                    "total": total,
                    "page": page,
                    "page_size": page_size,
                    "pages": (total + page_size - 1) // page_size
                }
            }
        )
