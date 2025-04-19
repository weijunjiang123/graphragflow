import os
import logging
from pathlib import Path
from neo4j import GraphDatabase
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator  # 更新为 field_validator
from dotenv import load_dotenv, find_dotenv

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 查找.env文件的多个可能位置
possible_env_paths = [
    Path(__file__).resolve().parent / '.env',                     # 当前目录
    Path(__file__).resolve().parent.parent / '.env',              # 上级目录
    Path(__file__).resolve().parent.parent.parent / '.env',       # 项目根目录
    Path.cwd() / '.env',                                          # 工作目录
]

# 尝试找到.env文件并加载
env_found = False
for env_path in possible_env_paths:
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
        logger.info(f"已加载环境变量文件: {env_path}")
        print(f"已加载环境变量文件: {env_path}")
        env_found = True
        break

# 检查是否可以通过 find_dotenv() 自动查找 .env 文件
if not env_found:
    dotenv_path = find_dotenv()
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"通过自动查找加载环境变量文件: {dotenv_path}")
        print(f"通过自动查找加载环境变量文件: {dotenv_path}")
        env_found = True

if not env_found:
    logger.warning("未找到.env文件, 将使用默认值")
    print("警告: 未找到.env文件, 将使用默认值")

class Config(BaseSettings):
    # Neo4j 连接参数
    NEO4J_URL: str = Field(default='neo4j://localhost:7687')
    NEO4J_USER: str = Field(default='neo4j')
    NEO4J_PASSWORD: str = Field(default='neo4j')  # Neo4j默认密码
    NEO4J_DATABASE: str = Field(default='neo4j')

    # 使用新的 field_validator 替代旧的 validator
    @field_validator("NEO4J_URL")
    @classmethod  # 需要添加 classmethod 装饰器
    def validate_uri(cls, v):
        if not v.startswith(("neo4j://", "bolt://")):
            raise ValueError("Neo4j URI必须以neo4j://或bolt://开头")
        return v

    # 日志配置
    LOG_LEVEL: str = Field(default='INFO')
    LOG_FILE: str = Field(default='graph_import.log')

    # 导入设置
    BATCH_SIZE: int = Field(default=1000)

    model_config = SettingsConfigDict(
        env_file=[str(p) for p in possible_env_paths if p.exists()],
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

# 实例化配置
config = Config()

# 导出正确的变量映射
DATABASE_URI = config.NEO4J_URL
USERNAME = config.NEO4J_USER
PASSWORD = config.NEO4J_PASSWORD
LOG_LEVEL = config.LOG_LEVEL
LOG_FILE = config.LOG_FILE
BATCH_SIZE = config.BATCH_SIZE

# 打印当前配置(隐藏密码)
logger.info(f"Neo4j连接: {DATABASE_URI}")
logger.info(f"用户名: {USERNAME}")
logger.info(f"密码: {'*' * len(PASSWORD)}")  # 安全起见不显示实际密码

# 便于调试的控制台输出
print("\n当前Neo4j连接配置:")
print(f"- URI: {DATABASE_URI}")
print(f"- 用户名: {USERNAME}")
print(f"- 密码: {'*' * len(PASSWORD)}")

# 尝试测试连接
print("\n正在测试Neo4j连接...")
try:
    driver = GraphDatabase.driver(DATABASE_URI, auth=(USERNAME, PASSWORD))
    driver.verify_connectivity()
    print("✅ 连接成功!")
    driver.close()
except Exception as e:
    print(f"❌ 连接失败: {e}")
    print("\n可能的解决方法:")
    print("1. 确认 Neo4j 数据库已启动")
    print("2. 检查用户名和密码是否正确")
    print("3. 在项目根目录中创建 .env 文件并设置以下内容:")
    print("   NEO4J_URL=neo4j://localhost:7687")
    print("   NEO4J_USER=neo4j")
    print("   NEO4J_PASSWORD=你的实际密码")
    print("4. 如果是首次使用 Neo4j，默认密码是 'neo4j'，系统会要求你修改密码")