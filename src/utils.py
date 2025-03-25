import datetime
import json
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

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
            "nodes": [],
            "relationships": [],
            "source_document": {
                "page_content": doc.source_document.page_content if hasattr(doc, 'source_document') else 
                               (doc.source.page_content if hasattr(doc, 'source') else None),
                "metadata": doc.source_document.metadata if hasattr(doc, 'source_document') else 
                           (doc.source.metadata if hasattr(doc, 'source') else None)
            } if (hasattr(doc, 'source_document') or hasattr(doc, 'source')) else None
        }
        
        # Add nodes with proper error handling
        for node in doc.nodes:
            node_dict = {"id": node.id}
            if hasattr(node, 'type'):
                node_dict["type"] = node.type
            elif hasattr(node, 'label'):
                node_dict["type"] = node.label
            if hasattr(node, 'properties'):
                try:
                    props = {}
                    for key, value in node.properties.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            props[key] = value
                        else:
                            props[key] = str(value)
                    node_dict["properties"] = props
                except Exception as e:
                    node_dict["properties"] = {}
            else:
                node_dict["properties"] = {}
            
            doc_dict["nodes"].append(node_dict)
        
        for rel in doc.relationships:
            rel_dict = {}
            if hasattr(rel, 'source'):
                if hasattr(rel.source, 'id'):  
                    rel_dict["source"] = rel.source.id
                else:  
                    rel_dict["source"] = rel.source
                    
            if hasattr(rel, 'target'):
                if hasattr(rel.target, 'id'):  
                    rel_dict["target"] = rel.target.id
                else: 
                    rel_dict["target"] = rel.target
                    
            if hasattr(rel, 'type'):
                rel_dict["type"] = rel.type
            if hasattr(rel, 'properties'):
                try:
                    props = {}
                    for key, value in rel.properties.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            props[key] = value
                        else:
                            props[key] = str(value)
                    rel_dict["properties"] = props
                except Exception as e:
                    rel_dict["properties"] = {}
            else:
                rel_dict["properties"] = {}
            
            doc_dict["relationships"].append(rel_dict)
        
        serializable_docs.append(doc_dict)
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
    
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


