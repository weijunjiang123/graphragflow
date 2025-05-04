#!/usr/bin/env python
"""
GraphRAG API 服务启动脚本
"""
import os
import sys
import uvicorn
from pathlib import Path

# 添加项目根目录到 Python 路径
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))

# 从环境变量获取配置
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8000"))
RELOAD = os.environ.get("RELOAD", "true").lower() == "true"

if __name__ == "__main__":
    print(f"启动 GraphRAG API 服务 - 监听: {HOST}:{PORT}, 热重载: {RELOAD}")
    uvicorn.run(
        "src.app:app",
        host=HOST,
        port=PORT,
        reload=RELOAD
    )