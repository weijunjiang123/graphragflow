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

def save_graph_documents(graph_documents):
    """Save graph documents to a JSON file
    
    Args:
        graph_documents: List of graph documents to save
        
    Returns:
        Path to the saved file
    """
    import json
    import os
    from datetime import datetime
    import logging
    
    logger = logging.getLogger(__name__)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"graph_documents_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    try:
        # Debug information about the structure of the graph documents
        logger.debug(f"Graph documents type: {type(graph_documents)}")
        if graph_documents and len(graph_documents) > 0:
            logger.debug(f"First document type: {type(graph_documents[0])}")
            # Log available attributes or keys
            if hasattr(graph_documents[0], "__dict__"):
                logger.debug(f"First document attributes: {graph_documents[0].__dict__.keys()}")
            elif isinstance(graph_documents[0], dict):
                logger.debug(f"First document keys: {graph_documents[0].keys()}")
        
        # Convert to serializable format
        serializable_docs = []
        for doc in graph_documents:
            # Handle different possible document formats
            if hasattr(doc, "nodes") and hasattr(doc, "relationships"):
                # Object with attributes
                doc_data = {
                    "nodes": [
                        {
                            "id": node.id if hasattr(node, "id") else node.get("id", ""),
                            "type": node.type if hasattr(node, "type") else node.get("type", ""),
                            "properties": node.properties if hasattr(node, "properties") else node.get("properties", {})
                        } for node in doc.nodes
                    ],
                    "relationships": [
                        {
                            "source": rel.source if hasattr(rel, "source") else rel.get("source", ""),
                            "target": rel.target if hasattr(rel, "target") else rel.get("target", ""),
                            "type": rel.type if hasattr(rel, "type") else rel.get("type", ""),
                            "properties": rel.properties if hasattr(rel, "properties") else rel.get("properties", {})
                        } for rel in doc.relationships
                    ]
                }
                
                # Add source if available
                if hasattr(doc, "source"):
                    # Handle source regardless of whether it's a Document or string
                    if hasattr(doc.source, "metadata"):
                        doc_data["source"] = {
                            "content": doc.source.page_content if hasattr(doc.source, "page_content") else "",
                            "metadata": doc.source.metadata
                        }
                    else:
                        doc_data["source"] = str(doc.source)
                    
                serializable_docs.append(doc_data)
            elif isinstance(doc, dict) and "nodes" in doc and "relationships" in doc:
                # Dictionary format
                serializable_docs.append(doc)
            else:
                logger.warning(f"Unrecognized document format: {type(doc)}")
                
        # Save to file
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Graph documents saved to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving graph documents: {str(e)}", exc_info=True)
        raise

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
