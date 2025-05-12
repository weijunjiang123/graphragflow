"""
服务工具模块 - 提供辅助功能
"""
import logging
import json
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class ResultFormatter:
    """
    结果格式化工具类
    用于格式化图检索结果，使其更易于前端展示
    """
    
    @staticmethod
    def format_cypher_result(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        格式化Cypher查询结果
        
        Args:
            results: Cypher查询结果
            
        Returns:
            格式化后的结果
        """
        formatted = {
            "records": results,
            "summary": {
                "count": len(results),
            }
        }
        
        # 提取字段信息
        if results and len(results) > 0:
            fields = list(results[0].keys())
            formatted["summary"]["fields"] = fields
            
            # 分析结果类型
            field_types = {}
            for field in fields:
                sample = results[0].get(field)
                if sample is None:
                    field_types[field] = "null"
                elif isinstance(sample, dict):
                    field_types[field] = "object"
                elif isinstance(sample, list):
                    field_types[field] = "array"
                else:
                    field_types[field] = type(sample).__name__
            
            formatted["summary"]["field_types"] = field_types
        
        return formatted
    
    @staticmethod
    def convert_to_graph_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        将结果转换为图可视化数据格式
        
        Args:
            results: 查询结果
            
        Returns:
            图可视化数据
        """
        nodes = []
        links = []
        node_ids = set()
        
        for record in results:
            for key, value in record.items():
                # 处理节点
                if isinstance(value, dict) and "id" in value and ("label" in value or "labels" in value):
                    node_id = value.get("id")
                    if node_id not in node_ids:
                        node_ids.add(node_id)
                        nodes.append({
                            "id": node_id,
                            "label": value.get("label") or (value.get("labels", [""])[0] if value.get("labels") else ""),
                            "properties": value.get("properties", {})
                        })
                
                # 处理关系
                if isinstance(value, dict) and "source" in value and "target" in value:
                    links.append({
                        "source": value.get("source"),
                        "target": value.get("target"),
                        "type": value.get("type", "RELATED_TO"),
                        "properties": value.get("properties", {})
                    })
                
                # 处理关系列表
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict) and "source" in item and "target" in item:
                            links.append({
                                "source": item.get("source"),
                                "target": item.get("target"),
                                "type": item.get("type", "RELATED_TO"),
                                "properties": item.get("properties", {})
                            })
        
        return {
            "nodes": nodes,
            "links": links
        }

class LogUtils:
    """
    日志工具类
    """
    
    @staticmethod
    def setup_logging(level=logging.INFO):
        """
        设置日志
        
        Args:
            level: 日志级别
        """
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
