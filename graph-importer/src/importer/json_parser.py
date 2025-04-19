import json
import logging
from typing import Dict, List, Tuple, Any, Optional
from models.node import Node
from models.relationship import Relationship

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class JsonParser:
    """Parser for the GraphRAG JSON format to extract nodes and relationships."""
    
    def __init__(self, json_data=None, json_path=None):
        """Initialize parser with JSON data or file path.
        
        Args:
            json_data (list, optional): List of document entries with nodes and relationships
            json_path (str, optional): Path to JSON file containing graph documents
        """
        if json_path and not json_data:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    # 移除可能存在的注释行（以 // 开头的内容）
                    content = ""
                    for line in f:
                        if not line.strip().startswith('//'):
                            content += line
                    self.json_data = json.loads(content)
                logger.info(f"Successfully loaded JSON data from {json_path}")
            except (json.JSONDecodeError, FileNotFoundError) as e:
                logger.error(f"Failed to load JSON from {json_path}: {str(e)}")
                self.json_data = []
        else:
            self.json_data = json_data if json_data else []
            
        self.nodes = []
        self.relationships = []
        self.stats = {
            "total_entries": 0,
            "total_nodes": 0,
            "total_relationships": 0,
            "skipped_nodes": 0,
            "skipped_relationships": 0,
            "errors": 0
        }
        
    def _extract_source_info(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Extract source document information from entry.
        
        Args:
            entry: Document entry containing source information
            
        Returns:
            Dictionary with source document metadata
        """
        source_info = {}
        try:
            source_doc = entry.get('source', {})
            if not source_doc and 'source_document' in entry:
                source_doc = entry.get('source_document', {})
                
            # Handle different source document formats
            if source_doc:
                # Extract metadata
                metadata = source_doc.get('metadata', {})
                if metadata:
                    if 'source' in metadata:
                        source_info['source_document'] = metadata['source']
                    if 'id' in metadata:
                        source_info['document_id'] = metadata['id']
                
                # Extract content
                content = source_doc.get('page_content', '')
                if content:
                    # Truncate content if too long
                    source_info['content_snippet'] = content[:100] + '...' if len(content) > 100 else content
        except Exception as e:
            logger.warning(f"Error extracting source info: {str(e)}")
            
        return source_info
    
    def _validate_node(self, node: Dict[str, Any]) -> bool:
        """Validate if a node has all required fields.
        
        Args:
            node: Node dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(node, dict):
            return False
            
        # Check for required fields
        if 'id' not in node or not node['id']:
            return False
            
        # Type is recommended but not strictly required
        if 'type' not in node:
            node['type'] = 'Entity'  # Default type
            
        # Properties should be a dictionary
        if 'properties' in node and not isinstance(node['properties'], dict):
            node['properties'] = {}
            
        return True
    
    def _validate_relationship(self, rel: Dict[str, Any]) -> bool:
        """Validate if a relationship has all required fields.
        
        Args:
            rel: Relationship dictionary to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(rel, dict):
            return False
            
        # Check for required fields
        required_fields = ['source', 'target', 'type']
        for field in required_fields:
            if field not in rel or not rel[field]:
                return False
                
        # Properties should be a dictionary
        if 'properties' in rel and not isinstance(rel['properties'], dict):
            rel['properties'] = {}
            
        return True
    
    def parse(self) -> Tuple[List[Node], List[Relationship]]:
        """Parse the JSON data into nodes and relationships.
        
        Returns:
            tuple: (nodes, relationships) lists
        """
        if not self.json_data or not isinstance(self.json_data, list):
            logger.warning("Invalid JSON data format: expected a list")
            return [], []
        
        self.stats["total_entries"] = len(self.json_data)
        logger.info(f"Parsing {len(self.json_data)} document entries")
        
        # Track existing nodes to avoid duplicates
        node_ids = set()
        
        # Process each document entry
        for entry_idx, entry in enumerate(self.json_data):
            try:
                # Extract source document info
                source_info = self._extract_source_info(entry)
                
                # Process nodes if they exist
                if 'nodes' in entry and isinstance(entry['nodes'], list):
                    for node in entry['nodes']:
                        try:
                            # Skip empty nodes or place fillers
                            if not node or node == {} or not self._validate_node(node):
                                self.stats["skipped_nodes"] += 1
                                continue
                                
                            node_id = node['id']
                            if node_id not in node_ids:
                                node_ids.add(node_id)
                                node_type = node.get('type', 'Entity')
                                properties = node.get('properties', {})
                                
                                # Add source information to properties if available
                                if source_info:
                                    properties.update(source_info)
                                    
                                # Create Node object
                                self.nodes.append(Node(
                                    node_id, 
                                    node_type, 
                                    properties
                                ))
                                self.stats["total_nodes"] += 1
                        except Exception as e:
                            logger.warning(f"Error parsing node in entry {entry_idx}: {str(e)}")
                            self.stats["errors"] += 1
                
                # Process relationships if they exist
                if 'relationships' in entry and isinstance(entry['relationships'], list):
                    for rel in entry['relationships']:
                        try:
                            # Skip empty relationships or place fillers
                            if not rel or rel == {} or not self._validate_relationship(rel):
                                self.stats["skipped_relationships"] += 1
                                continue
                                
                            source = rel['source']
                            target = rel['target']
                            rel_type = rel['type']
                            properties = rel.get('properties', {})
                            
                            # Add source information to properties if available
                            if source_info:
                                properties.update(source_info)
                                
                            # Create Relationship object
                            self.relationships.append(Relationship(
                                source,
                                target,
                                rel_type,
                                properties
                            ))
                            self.stats["total_relationships"] += 1
                        except Exception as e:
                            logger.warning(f"Error parsing relationship in entry {entry_idx}: {str(e)}")
                            self.stats["errors"] += 1
                    
            except Exception as e:
                logger.error(f"Error parsing entry {entry_idx}: {str(e)}")
                self.stats["errors"] += 1
                
        # Log statistics
        logger.info(f"Parsing completed: {self.stats['total_nodes']} nodes and {self.stats['total_relationships']} relationships")
        if self.stats["skipped_nodes"] > 0 or self.stats["skipped_relationships"] > 0:
            logger.info(f"Skipped items: {self.stats['skipped_nodes']} nodes, {self.stats['skipped_relationships']} relationships")
        if self.stats["errors"] > 0:
            logger.warning(f"Encountered {self.stats['errors']} errors during parsing")
            
        return self.nodes, self.relationships
        
    @classmethod
    def from_file(cls, file_path: str) -> 'JsonParser':
        """Create a JsonParser instance from a file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            JsonParser instance
        """
        return cls(json_path=file_path)
        
    def get_stats(self) -> Dict[str, int]:
        """Get parsing statistics.
        
        Returns:
            Dictionary with parsing statistics
        """
        return self.stats