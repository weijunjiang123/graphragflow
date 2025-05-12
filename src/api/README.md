# GraphRAG API 文档

GraphRAG API是一个基于FastAPI构建的RESTful API服务，专为图检索和自然语言到Cypher查询（text2Cypher）设计。该API提供高性能、可扩展的接口，支持多种检索方式，包括text2Cypher、向量搜索、全文搜索和混合搜索。

## 特性

- **强大的自然语言处理**：将自然语言转换为Cypher查询语句
- **多种搜索模式**：支持text2Cypher、向量搜索、全文搜索和混合搜索
- **高性能设计**：使用缓存、异步处理和性能监控
- **健壮的异常处理**：统一的错误响应格式
- **版本控制**：支持API版本控制，确保兼容性
- **全面的监控**：请求日志、性能指标和健康状态检查
- **跨域支持**：内置CORS支持
- **请求限流**：防止API滥用
- **OpenAPI文档**：自动生成的API文档

## API端点

### 核心功能

- **POST /api/v1/text2cypher**：执行Text2Cypher查询
- **POST /api/v1/search**：执行统一搜索（支持多种搜索类型）
- **GET /api/v1/examples**：获取示例查询

### 管理与监控

- **GET /api/health**：API健康检查
- **POST /api/cache/clear**：清空缓存
- **GET /api/metrics**：获取性能指标
- **POST /api/metrics/clear**：清空性能指标

## 使用示例

### Text2Cypher查询

```python
import requests
import json

# API端点
url = "http://localhost:8000/api/v1/text2cypher"

# 请求数据
data = {
    "query": "找出所有与知识图谱相关的文档",
    "limit": 5
}

# 发送请求
response = requests.post(url, json=data)
result = response.json()

# 打印结果
print(json.dumps(result, indent=2, ensure_ascii=False))
```
  "original_query": "图数据库知识",
  "search_type": "text2cypher",
  "results": [...],
  "result_count": 5,
  "cypher_query": "MATCH (d:Document) WHERE d.text CONTAINS '图数据库' RETURN d LIMIT 5"
}
```

### 3. 示例查询 API

获取示例查询列表

**URL:** `/api/examples`  
**方法:** GET  
**响应:**
```json
[
  {
    "description": "基础查询 - 查找文档",
    "query": "查找包含'图数据库'关键词的文档",
    "explanation": "展示如何基于关键词查找文档"
  },
  ...
]
```

### 4. 测试 API

测试API服务状态和text2Cypher功能

**URL:** `/api/test`  
**方法:** GET  
**响应:**
```json
{
  "success": true,
  "api_status": "online",
  "text2cypher_supported": true,
  "test_result": {...}
}
```

## 错误处理

API使用标准HTTP状态码表示请求的结果：

- **200 OK:** 请求成功
- **400 Bad Request:** 请求参数错误
- **422 Unprocessable Entity:** 请求体验证失败
- **500 Internal Server Error:** 服务器内部错误

错误响应格式：
```json
{
  "success": false,
  "detail": "错误描述"
}
```

## 启动服务

可以使用以下方式启动API服务：

1. 使用run_api.py脚本：
```bash
python -m src.service.run_api --host 0.0.0.0 --port 8000
```

2. 使用main_api.py：
```bash
python main_api.py
```

3. 直接使用uvicorn：
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

## API文档

在服务启动后，可以通过以下URL访问API文档：

- Swagger UI: `http://localhost:8000/api/docs`
- ReDoc: `http://localhost:8000/api/redoc`
