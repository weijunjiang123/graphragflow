"""
GraphRAG API服务 - 主入口点
"""
import os
import uvicorn

if __name__ == "__main__":
    # 读取环境变量或使用默认值
    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", 8000))
    
    # 启动Uvicorn服务器
    uvicorn.run("src.api.app:app", host=host, port=port, reload=True)
