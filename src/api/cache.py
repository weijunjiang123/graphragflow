"""
GraphRAG API - 缓存模块
提供缓存功能，减少重复计算和数据库查询
"""
import time
import logging
import json
import hashlib
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理不可序列化的对象"""
    def default(self, obj):
        try:
            return super().default(obj)
        except TypeError:
            # 对于不可序列化的对象，返回其类名和id
            return f"{obj.__class__.__name__}_{id(obj)}"

class Cache:
    """简单的内存缓存实现"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        """
        初始化缓存
        
        Args:
            max_size: 缓存最大条目数
            ttl: 缓存条目生存时间（秒）
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.max_size = max_size
        self.ttl = ttl
        logger.info(f"初始化缓存，最大条目数: {max_size}，生存时间: {ttl}秒")
    
    def _generate_key(self, *args, **kwargs) -> str:
        """
        根据参数生成缓存键
        
        Args:
            *args: 位置参数
            **kwargs: 关键字参数
            
        Returns:
            缓存键
        """
        # 将参数转换为JSON字符串，然后计算哈希值
        key_data = {
            "args": args,
            "kwargs": kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True, cls=CustomJSONEncoder)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            缓存值，如果不存在或已过期则返回None
        """
        if key not in self.cache:
            return None
        
        cache_item = self.cache[key]
        # 检查是否过期
        if time.time() > cache_item["expires_at"]:
            del self.cache[key]
            return None
            
        return cache_item["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 单独设置此项的生存时间（秒）
        """
        # 如果缓存已满，删除一个最旧的条目
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = min(self.cache, key=lambda k: self.cache[k]["expires_at"])
            del self.cache[oldest_key]
        
        # 设置缓存条目
        expires_at = time.time() + (ttl if ttl is not None else self.ttl)
        self.cache[key] = {
            "value": value,
            "expires_at": expires_at
        }
    
    def invalidate(self, key: str) -> bool:
        """
        使缓存条目失效
        
        Args:
            key: 缓存键
            
        Returns:
            是否成功使缓存失效
        """
        if key in self.cache:
            del self.cache[key]
            return True
        return False
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        logger.info("缓存已清空")


# 创建全局缓存实例
text2cypher_cache = Cache(max_size=500, ttl=1800)  # 30分钟缓存
example_cache = Cache(max_size=100, ttl=86400)  # 24小时缓存


def cached(cache_instance: Cache, prefix: str = ""):
    """
    函数装饰器，为函数结果提供缓存
    
    Args:
        cache_instance: 缓存实例
        prefix: 缓存键前缀
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{prefix}:{cache_instance._generate_key(*args, **kwargs)}"
            
            # 尝试从缓存获取
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 缓存未命中，执行函数
            logger.debug(f"缓存未命中: {cache_key}")
            result = func(*args, **kwargs)
            
            # 将结果存入缓存
            cache_instance.set(cache_key, result)
            return result
        
        return wrapper
    
    return decorator


def async_cached(cache_instance: Cache, prefix: str = ""):
    """
    异步函数装饰器，为异步函数结果提供缓存
    
    Args:
        cache_instance: 缓存实例
        prefix: 缓存键前缀
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{prefix}:{cache_instance._generate_key(*args, **kwargs)}"
            
            # 尝试从缓存获取
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                logger.debug(f"缓存命中: {cache_key}")
                return cached_result
            
            # 缓存未命中，执行函数
            logger.debug(f"缓存未命中: {cache_key}")
            result = await func(*args, **kwargs)
            
            # 将结果存入缓存
            cache_instance.set(cache_key, result)
            return result
        
        return wrapper
    
    return decorator