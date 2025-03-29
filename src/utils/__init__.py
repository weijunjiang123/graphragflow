import re
import json
import os
import logging
import datetime
from pathlib import Path
from typing import List, Any

def clean_markdown_json(text: str) -> str:
    """
    Clean JSON from markdown code blocks that may be returned by LLMs.
    
    Args:
        text: Text that may contain markdown-formatted JSON
        
    Returns:
        Cleaned JSON string
    """
    # Pattern to match JSON content within markdown code blocks
    pattern = r'```(?:json)?\s*([\s\S]*?)```'
    
    # Find all matches
    matches = re.findall(pattern, text)
    
    if matches:
        # Return the first match (the JSON content)
        return matches[0].strip()
    
    # If no markdown formatting, return the original text
    return text

def save_graph_documents(graph_documents: List[Any]) -> str:
    """
    Save graph documents to a JSON file in the 'output' directory.
    
    Args:
        graph_documents: List of graph documents to save
        
    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"graph_documents_{timestamp}.json"
    filepath = output_dir / filename
    
    # Convert graph documents to serializable format
    serializable_docs = []
    for doc in graph_documents:
        # Convert each document to a dictionary
        if hasattr(doc, '__dict__'):
            doc_dict = doc.__dict__
            # Handle nested objects
            serializable_doc = {}
            for key, value in doc_dict.items():
                if hasattr(value, '__dict__'):
                    serializable_doc[key] = value.__dict__
                elif isinstance(value, list):
                    # Handle list of objects
                    serializable_list = []
                    for item in value:
                        if hasattr(item, '__dict__'):
                            serializable_list.append(item.__dict__)
                        else:
                            serializable_list.append(item)
                    serializable_doc[key] = serializable_list
                else:
                    serializable_doc[key] = value
            serializable_docs.append(serializable_doc)
        else:
            # If it's already a dict or other serializable type
            serializable_docs.append(doc)
    
    # Save to file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2, default=str)
    
    return str(filepath)

def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Path to log file (optional)
        
    Returns:
        None
    """
    # Create logs directory if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Basic configuration
    logging_config = {
        'level': level,
        'format': '%(levelname)s: %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    # Add file handler if log_file is specified
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    # Apply configuration
    logging.basicConfig(**logging_config)
    
    # Set specific levels for noisy libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)