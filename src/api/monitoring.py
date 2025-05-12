"""
GraphRAG API - 性能监控模块
提供API性能监控和指标收集功能
"""
import time
import logging
import statistics
from typing import Dict, List, Optional, Any
from functools import wraps
from pydantic import BaseModel
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class PerformanceMetric(BaseModel):
    """性能指标模型"""
    endpoint: str
    method: str
    status_code: int
    response_time: float  # 毫秒
    timestamp: float
    request_id: Optional[str] = None
    user_agent: Optional[str] = None
    ip_address: Optional[str] = None
    
class PerformanceStats(BaseModel):
    """性能统计模型"""
    endpoint: str
    method: str
    request_count: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    p50_response_time: float
    p90_response_time: float
    p95_response_time: float
    p99_response_time: float
    error_count: int
    error_rate: float

class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self, max_metrics: int = 10000):
        """
        初始化性能监控
        
        Args:
            max_metrics: 最大存储的指标数量
        """
        self.metrics: List[PerformanceMetric] = []
        self.max_metrics = max_metrics
        logger.info(f"初始化性能监控，最大指标数量: {max_metrics}")
    
    def add_metric(self, metric: PerformanceMetric) -> None:
        """
        添加性能指标
        
        Args:
            metric: 性能指标
        """
        self.metrics.append(metric)
        # 如果超过最大数量，删除最早的指标
        if len(self.metrics) > self.max_metrics:
            self.metrics = self.metrics[-self.max_metrics:]
    
    def get_stats(self, endpoint: Optional[str] = None, method: Optional[str] = None, 
                  time_range: Optional[int] = None) -> List[PerformanceStats]:
        """
        获取性能统计数据
        
        Args:
            endpoint: 过滤特定端点
            method: 过滤特定方法
            time_range: 时间范围（秒），如过去60秒
            
        Returns:
            性能统计列表
        """
        # 筛选指标
        filtered_metrics = self.metrics
        current_time = time.time()
        
        if endpoint:
            filtered_metrics = [m for m in filtered_metrics if m.endpoint == endpoint]
        if method:
            filtered_metrics = [m for m in filtered_metrics if m.method == method]
        if time_range:
            min_time = current_time - time_range
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= min_time]
        
        # 按端点和方法分组
        grouped_metrics: Dict[str, List[PerformanceMetric]] = {}
        for metric in filtered_metrics:
            key = f"{metric.endpoint}:{metric.method}"
            if key not in grouped_metrics:
                grouped_metrics[key] = []
            grouped_metrics[key].append(metric)
        
        # 计算统计数据
        stats = []
        for key, metrics in grouped_metrics.items():
            if not metrics:
                continue
                
            endpoint, method = key.split(":")
            response_times = [m.response_time for m in metrics]
            error_count = sum(1 for m in metrics if m.status_code >= 400)
            
            # 计算百分位数
            p50 = statistics.median(response_times) if response_times else 0
            p90 = statistics.quantiles(response_times, n=10)[-1] if len(response_times) >= 10 else max(response_times) if response_times else 0
            p95 = statistics.quantiles(response_times, n=20)[-1] if len(response_times) >= 20 else max(response_times) if response_times else 0
            p99 = statistics.quantiles(response_times, n=100)[-1] if len(response_times) >= 100 else max(response_times) if response_times else 0
            
            stats.append(PerformanceStats(
                endpoint=endpoint,
                method=method,
                request_count=len(metrics),
                avg_response_time=sum(response_times) / len(response_times) if response_times else 0,
                min_response_time=min(response_times) if response_times else 0,
                max_response_time=max(response_times) if response_times else 0,
                p50_response_time=p50,
                p90_response_time=p90,
                p95_response_time=p95,
                p99_response_time=p99,
                error_count=error_count,
                error_rate=error_count / len(metrics) if metrics else 0
            ))
        
        return stats
    
    def clear(self) -> None:
        """清空所有指标"""
        self.metrics.clear()
        logger.info("性能指标已清空")

# 创建全局性能监控实例
performance_monitor = PerformanceMonitor()

@contextmanager
def measure_performance():
    """
    测量代码块执行时间的上下文管理器
    
    Returns:
        开始时间
    """
    start_time = time.time()
    try:
        yield start_time
    finally:
        end_time = time.time()
        elapsed_ms = (end_time - start_time) * 1000
        logger.debug(f"代码块执行时间: {elapsed_ms:.2f}ms")

def performance_logged(func):
    """
    性能日志装饰器
    记录函数执行时间
    
    Args:
        func: 要装饰的函数
    
    Returns:
        装饰后的函数
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        with measure_performance() as start_time:
            result = await func(*args, **kwargs)
        return result
    
    return wrapper
