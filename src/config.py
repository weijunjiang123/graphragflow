import os
from dotenv import load_dotenv

# 尝试加载.env文件中的环境变量
load_dotenv("../.env")

# Neo4j连接参数
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "jwj20020124")

# 文档处理参数
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 256))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 24))
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5")
DOCUMENT_PATH = os.getenv("DOCUMENT_PATH", "../data/乡土中国.txt")

# 日志配置
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# 结果输出目录
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")

