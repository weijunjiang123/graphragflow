"""
GraphRAG API服务启动脚本
"""
import os
import argparse
import uvicorn
from src.service.utils import LogUtils

def main():
    """启动FastAPI服务"""
    # 设置日志
    LogUtils.setup_logging()
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="启动GraphRAG API服务")
    parser.add_argument("--host", default="0.0.0.0", help="服务主机地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="是否启用热重载")
    parser.add_argument("--llm-provider", default="ollama", help="LLM提供商: ollama 或 openai")
    parser.add_argument("--llm-model", default="qwen2.5", help="LLM模型名称")
    parser.add_argument("--embedding-provider", default="ollama", help="嵌入提供商: ollama 或 openai")
    parser.add_argument("--embedding-model", default="nomic-embed-text", help="嵌入模型名称")
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["LLM_PROVIDER"] = args.llm_provider
    os.environ["LLM_MODEL"] = args.llm_model
    os.environ["EMBEDDING_PROVIDER"] = args.embedding_provider
    os.environ["EMBEDDING_MODEL"] = args.embedding_model
    
    # 输出配置信息
    print(f"启动服务配置:")
    print(f"  主机: {args.host}")
    print(f"  端口: {args.port}")
    print(f"  热重载: {args.reload}")
    print(f"  LLM提供商: {args.llm_provider}")
    print(f"  LLM模型: {args.llm_model}")
    print(f"  嵌入提供商: {args.embedding_provider}")
    print(f"  嵌入模型: {args.embedding_model}")
      # 启动服务
    uvicorn.run(
        "src.api.app:app", 
        host=args.host, 
        port=args.port, 
        reload=args.reload
    )

if __name__ == "__main__":
    main()
