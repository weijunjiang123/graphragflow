import datetime
import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class CustomJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle non-serializable data types"""
    def default(self, obj):
        if isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

def save_graph_documents(graph_docs, output_dir="results"):
    """Save graph documents to a JSON file

    Args:
        graph_docs: List of graph documents to save
        output_dir: Directory to save the output file

    Returns:
        Path to the saved file
    """
    # Create a results directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/graph_documents_{timestamp}.json"

    # Convert graph documents to serializable format
    serializable_docs = []
    for doc in graph_docs:
        doc_dict = {
            "nodes": [
                {
                    "id": node.id,
                    "type": getattr(node, 'type', getattr(node, 'label', None)),
                    "properties": node.properties if hasattr(node, 'properties') else {}
                } for node in doc.nodes
            ],
            "relationships": [
                {
                    "source": getattr(rel.source, 'id', rel.source),
                    "target": getattr(rel.target, 'id', rel.target),
                    "type": getattr(rel, 'type', None),
                    "properties": getattr(rel, 'properties', {})
                } for rel in doc.relationships
            ],
            "source_document": {
                "page_content": getattr(doc.source_document, 'page_content', getattr(doc.source, 'page_content', None)),
                "metadata": getattr(doc.source_document, 'metadata', getattr(doc.source, 'metadata', None))
            } if hasattr(doc, 'source_document') else {
                "page_content": getattr(doc, 'page_content', None),
                "metadata": getattr(doc, 'metadata', None)
            } if hasattr(doc, 'page_content') or hasattr(doc, 'metadata') else None
        }
        serializable_docs.append(doc_dict)

    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)

    logger.info(f"Graph documents saved to {filename}")
    print(f"✓ Graph documents saved to {filename}")
    return filename

def setup_logging(level=logging.INFO, log_file=None):
    """Set up logging configuration
    
    Args:
        level: Logging level
        log_file: Path to log file (if None, logs to console only)
    """
    handlers = []
    
    # Always add console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    handlers.append(console_handler)
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Suppress verbose logging from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)
