import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, validator

class Config(BaseSettings):
    # Neo4j connection parameters
    DATABASE_URI: str = Field(default='neo4j://localhost:7687')
    USERNAME: str = Field(default='neo4j')
    PASSWORD: str = Field(default='neo4j')

    @validator("DATABASE_URI")
    def validate_database_uri(cls, uri):
        if not uri.startswith("neo4j://") and not uri.startswith("bolt://"):
            raise ValueError("DATABASE_URI must start with neo4j:// or bolt://")
        return uri

    # Logging configuration
    LOG_LEVEL: str = Field(default='INFO')
    LOG_FILE: str = Field(default='graph_import.log')

    # Import settings
    BATCH_SIZE: int = Field(default=1000)

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )

config = Config()

DATABASE_URI = config.DATABASE_URI
USERNAME = config.USERNAME
PASSWORD = config.PASSWORD
LOG_LEVEL = config.LOG_LEVEL
LOG_FILE = config.LOG_FILE
BATCH_SIZE = config.BATCH_SIZE
