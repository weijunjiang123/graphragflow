import os
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Neo4j connection parameters
DATABASE_URI = os.getenv('NEO4J_URI', 'neo4j://localhost:7687')
USERNAME = os.getenv('NEO4J_USERNAME', 'neo4j')
PASSWORD = os.getenv('NEO4J_PASSWORD', 'neo4j')

# Logging configuration
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'graph_import.log')

# Import settings
BATCH_SIZE = int(os.getenv('BATCH_SIZE', 1000))