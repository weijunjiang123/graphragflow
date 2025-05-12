# GraphRAG API服务

这是一个基于FastAPI的GraphRAG API服务，提供text2Cypher检索和图数据库查询功能。

## 功能特点

- **Text2Cypher检索**：将自然语言查询转换为Neo4j的Cypher查询语句
- **多种检索方式**：支持text2Cypher、混合检索、向量检索和全文检索
- **结果解释**：自动解释查询结果，提供更好的用户体验
- **RESTful API**：提供标准化的RESTful API接口
- **示例查询**：提供常见查询场景的示例

## 安装依赖

确保已安装所有必要的依赖：

```bash
pip install fastapi uvicorn requests pydantic
```

## 启动服务

你可以使用以下命令启动API服务：

```bash
python -m src.service.run_api --host 0.0.0.0 --port 8000
```

参数说明：
- `--host`：服务主机地址，默认为0.0.0.0
- `--port`：服务端口，默认为8000
- `--reload`：是否启用热重载
- `--llm-provider`：LLM提供商，可选值为ollama或openai，默认为ollama
- `--llm-model`：LLM模型名称，默认为qwen2.5
- `--embedding-provider`：嵌入提供商，可选值为ollama或openai，默认为ollama
- `--embedding-model`：嵌入模型名称，默认为nomic-embed-text

## API接口

### 1. Text2Cypher搜索

将自然语言查询转换为Cypher查询并执行。

- **URL**: `/api/text2cypher`
- **方法**: POST
- **请求体**:
  ```json
  {
    "query": "查找与人工智能相关的文档",
    "limit": 5
  }
  ```
- **响应**:
  ```json
  {
    "original_query": "查找与人工智能相关的文档",
    "cypher_query": "MATCH (d:Document) WHERE d.text CONTAINS '人工智能' RETURN d LIMIT 5",
    "results": [...],
    "result_count": 3,
    "explanation": "这个查询查找了文本内容中包含'人工智能'关键词的文档节点。"
  }
  ```

### 2. 统一搜索接口

支持多种搜索类型的统一接口。

- **URL**: `/api/search`
- **方法**: POST
- **请求体**:
  ```json
  {
    "query": "图数据库知识",
    "search_type": "text2cypher",
    "limit": 5
  }
  ```
- **search_type可选值**:
  - `text2cypher`: 使用text2Cypher搜索
  - `hybrid`: 使用混合搜索
  - `vector`: 使用向量搜索
  - `fulltext`: 使用全文搜索

### 3. 获取示例查询

获取示例查询列表。

- **URL**: `/api/examples`
- **方法**: GET
- **响应**:
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

### 4. 测试API服务

测试API服务状态和text2Cypher功能。

- **URL**: `/api/test`
- **方法**: GET
- **响应**:
  ```json
  {
    "api_status": "online",
    "text2cypher_supported": true,
    "test_result": {...}
  }
  ```

## 客户端示例

你可以使用提供的客户端示例脚本来测试API：

```bash
# 获取示例查询
python -m src.service.client_example --examples

# 执行text2Cypher搜索
python -m src.service.client_example --query "查找与人工智能相关的文档" --type text2cypher

# 执行混合搜索
python -m src.service.client_example --query "图数据库" --type hybrid --limit 10
```

## 环境变量

服务支持通过环境变量配置：

- `NEO4J_URI`: Neo4j数据库URI
- `NEO4J_USER`: Neo4j用户名
- `NEO4J_PASSWORD`: Neo4j密码
- `NEO4J_DATABASE`: Neo4j数据库名称
- `LLM_PROVIDER`: LLM提供商
- `LLM_MODEL`: LLM模型名称
- `EMBEDDING_PROVIDER`: 嵌入提供商
- `EMBEDDING_MODEL`: 嵌入模型名称
- `API_HOST`: API服务主机地址
- `API_PORT`: API服务端口

## 服务架构

服务采用分层架构设计：

1. **API层**：处理HTTP请求和响应，输入验证和错误处理
2. **服务层**：封装业务逻辑，调用底层组件
3. **核心层**：实现具体功能，如图检索、text2Cypher转换等
4. **依赖管理**：管理共享资源和依赖注入
5. **工具类**：提供辅助功能和结果格式化
