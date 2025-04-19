import os
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, ClassVar
from dotenv import load_dotenv, find_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator

# 增强环境变量文件查找能力
def find_and_load_env_file():
    """尝试使用dotenv寻找并加载.env文件"""
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path, override=True)
        print(f"成功加载环境变量文件: {dotenv_path}")
        return Path(dotenv_path).parent, Path(dotenv_path)
    else:
        print("警告: 未找到.env文件，将使用默认配置")
        # 默认根目录为当前文件父目录的父目录
        default_root = Path(__file__).resolve().parent.parent
        return default_root, default_root / '.env'

# 查找并加载环境变量
PROJECT_ROOT, env_path = find_and_load_env_file()

# 基础目录结构
CORE_DIRECTORIES = ["data", "logs", "results"]
for directory in CORE_DIRECTORIES:
    (PROJECT_ROOT / directory).mkdir(exist_ok=True, parents=True)


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    URI: str = Field(default="bolt://localhost:7687")
    USERNAME: str = Field(default="neo4j")
    PASSWORD: str = Field(default="123456")
    DATABASE_NAME: str = Field(default="neo4j")

    model_config = SettingsConfigDict(
        env_file=str(env_path),
        case_sensitive=True,
        env_prefix="NEO4J_",
        extra="ignore",
        env_mapping={
            "URI": "NEO4J_URL",
            "USERNAME": "NEO4J_USER",
            "PASSWORD": "NEO4J_PASSWORD",
            "DATABASE_NAME": "NEO4J_DATABASE"
        }
    )


class ModelProviderSettings(BaseSettings):
    """模型提供商配置"""
    MODEL_PROVIDER: str = Field(default="openai")
    
    # OpenAI 配置
    OPENAI_API_KEY: Optional[str] = Field(default=None)
    OPENAI_API_BASE: Optional[str] = Field(default="https://api.openai.com/v1")
    OPENAI_MODEL: str = Field(default="gpt-3.5-turbo")
    OPENAI_EMBEDDING_API_KEY: Optional[str] = Field(default=None)
    OPENAI_EMBEDDING_API_BASE: Optional[str] = Field(default="https://api.openai.com/v1")
    OPENAI_EMBEDDINGS_MODEL: str = Field(default="text-embedding-ada-002")
    
    # Ollama 配置
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434")
    OLLAMA_LLM_MODEL: str = Field(default="qwen2.5")
    VALID_PROVIDERS: ClassVar[List[str]] = ["ollama", "openai"]
    # 添加有效的Ollama模型列表
    VALID_LLM_MODELS: ClassVar[List[str]] = ["qwen2.5", "llama3", "llama2", "mistral", "gemma"]
    
    @field_validator("OLLAMA_LLM_MODEL")
    def validate_llm_model(cls, model):
        # Only validate for Ollama models, OpenAI models can be any string
        if os.environ.get("MODEL_PROVIDER", "ollama").lower() == "ollama" and model not in cls.VALID_LLM_MODELS:
            raise ValueError(f"OLLAMA_LLM_MODEL must be one of {cls.VALID_LLM_MODELS} when using Ollama")
        return model

    @field_validator("MODEL_PROVIDER")  # 修改这里，与实际字段名匹配
    def validate_provider(cls, v):
        if v.lower() not in cls.VALID_PROVIDERS:
            raise ValueError(f"MODEL_PROVIDER must be one of {cls.VALID_PROVIDERS}")
        return v.lower()
    
    model_config = SettingsConfigDict(
        env_file=str(env_path),
        case_sensitive=True,
        env_prefix="MODEL_",
        extra="ignore"
    )

class DocumentSettings(BaseSettings):
    """文档处理参数"""
    CHUNK_SIZE: int = Field(default=1000)
    CHUNK_OVERLAP: int = Field(default=200)
    DOCUMENT_PATH: Path = Field(default=PROJECT_ROOT / "data")
    
    @field_validator("DOCUMENT_PATH", mode='before')
    def parse_path(cls, v):
        return Path(v) if isinstance(v, str) else v
    
    model_config = SettingsConfigDict(
        env_file=str(env_path),
        case_sensitive=True,
        extra="ignore"
    )


class AppSettings(BaseSettings):
    """应用程序设置与路径"""
    # 路径配置
    LOG_DIR: Path = Field(default=PROJECT_ROOT / 'logs')
    DATA_DIR: Path = Field(default=PROJECT_ROOT / "data")
    OUTPUT_DIR: Path = Field(default=PROJECT_ROOT / "results")
    RESULTS_DIR: Path = Field(default=PROJECT_ROOT / "results")

    # 日志配置
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    LOG_DATE_FORMAT: str = Field(default="%Y-%m-%d %H:%M:%S")

    # 应用模式
    DEBUG_MODE: bool = Field(default=False, description="Enable debug mode")
    TEST_MODE: bool = Field(default=False)
    VECTOR_INDEX_NAME: str = Field(default="document_vector")

    @field_validator("DATA_DIR", "OUTPUT_DIR", "LOG_DIR", mode='before')
    def parse_path(cls, v):
        if isinstance(v, str):
            path = Path(v)
            path.mkdir(exist_ok=True, parents=True)
            return path
        return v

    @property
    def LOG_FILE(self) -> Path:
        return self.LOG_DIR / 'graph_import.log'

    @property
    def LOG_LEVEL_ENUM(self) -> int:
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        return level_map.get(self.LOG_LEVEL.upper(), logging.INFO)

    model_config = SettingsConfigDict(
        env_file=str(env_path),
        case_sensitive=True,
        extra="ignore"
    )


# 实例化配置
DATABASE = DatabaseSettings()
DOCUMENT = DocumentSettings()
APP = AppSettings()
MODEL = ModelProviderSettings()


def validate_config() -> Dict[str, str]:
    """验证配置并返回错误信息，提供更详细的错误提示"""
    errors = {}

    # 验证数据库连接
    if not DATABASE.URI:
        errors["DATABASE_URI"] = "数据库URI不能为空. 请设置 NEO4J_URL 环境变量."

    # 检查文档路径是否存在
    document_path = DOCUMENT.DOCUMENT_PATH
    if isinstance(document_path, str):
        document_path = Path(document_path)

    if not document_path.exists() and not document_path.is_dir():
        # 尝试查找备选路径
        alt_paths = [PROJECT_ROOT / "data", Path.cwd() / "data"]
        for alt_path in alt_paths:
            if alt_path.exists() and alt_path.is_dir():
                print(f"注意: 文档路径 {document_path} 不存在，使用备选路径 {alt_path}")
                DOCUMENT.DOCUMENT_PATH = alt_path
                break
        else:
            errors["DOCUMENT_PATH"] = (
                f"文档路径不存在: {document_path}. "
                f"请检查 DOCUMENT_PATH 环境变量是否正确，或者将文档放在以下默认路径之一: {', '.join(map(str, alt_paths))}"
            )

    return errors


def get_config_info() -> Dict[str, Any]:
    """返回配置信息的字典，用于日志记录和调试"""
    return {
        "database": {
            "uri": DATABASE.URI,
            "username": DATABASE.USERNAME,
            "database_name": DATABASE.DATABASE_NAME,
        },
        "document": {
            "path": str(DOCUMENT.DOCUMENT_PATH),
            "chunk_size": DOCUMENT.CHUNK_SIZE,
            "chunk_overlap": DOCUMENT.CHUNK_OVERLAP,
        },
        "model": {
            "provider": MODEL.MODEL_PROVIDER,
            "openai_model": MODEL.OPENAI_MODEL if MODEL.MODEL_PROVIDER == "openai" else "N/A",
            "openai_embeddings_model": MODEL.OPENAI_EMBEDDINGS_MODEL if MODEL.MODEL_PROVIDER == "openai" else "N/A",
            "ollama_base_url": MODEL.OLLAMA_BASE_URL if MODEL.MODEL_PROVIDER == "ollama" else "N/A",
        },
        "paths": {
            "data_dir": str(APP.DATA_DIR),
            "output_dir": str(APP.OUTPUT_DIR),
            "log_dir": str(APP.LOG_DIR),
            "log_file": str(APP.LOG_FILE),
        },
        "app": {
            "debug_mode": APP.DEBUG_MODE,
            "test_mode": APP.TEST_MODE,
            "log_level": APP.LOG_LEVEL,
        }
    }


# 验证配置
config_errors = validate_config()
if config_errors:
    error_message = "配置错误:\n" + "\n".join([f"{k}: {v}" for k, v in config_errors.items()])
    # Use the DEBUG_MODE from AppSettings
    if APP.DEBUG_MODE:
        print(error_message)

# 调试模式下显示配置
if APP.DEBUG_MODE:
    import json
    print(f"应用程序配置:\n{json.dumps(get_config_info(), indent=2, ensure_ascii=False)}")
