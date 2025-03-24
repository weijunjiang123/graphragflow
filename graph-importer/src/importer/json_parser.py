from models.node import Node
from models.relationship import Relationship

class JsonParser:
    """Parser for the GraphRAG JSON format to extract nodes and relationships."""
    
    def __init__(self, json_data):
        """Initialize parser with JSON data.
        
        Args:
            json_data (list): List of document entries with nodes and relationships
        """
        self.json_data = json_data
        self.nodes = []
        self.relationships = []
        
    def parse(self):
        """Parse the JSON data into nodes and relationships.
        
        Returns:
            tuple: (nodes, relationships) lists
        """
        if not self.json_data or not isinstance(self.json_data, list):
            print("Invalid JSON data format")
            return [], []
        
        print(f"Parsing {len(self.json_data)} document entries")
        
        # Track existing nodes to avoid duplicates
        node_ids = set()
        
        # Process each document entry
        for entry_idx, entry in enumerate(self.json_data):
            try:
                # Extract source document metadata if available
                source_doc = entry.get('source_document', {})
                doc_source = source_doc.get('metadata', {}).get('source', '')
                doc_content = source_doc.get('page_content', '')
                
                # Process nodes
                for node in entry.get('nodes', []):
                    node_id = node.get('id')
                    if not node_id or not isinstance(node_id, str):
                        continue
                        
                    # Skip if node is empty or has no valid ID
                    if node_id not in node_ids:
                        node_ids.add(node_id)
                        node_type = node.get('type', 'Entity')
                        properties = node.get('properties', {})
                        
                        # Add source information to properties
                        if doc_source:
                            properties['source_document'] = doc_source
                            
                        # Create Node object
                        self.nodes.append(Node(
                            node_id, 
                            node_type, 
                            properties
                        ))
                
                # Process relationships
                for rel in entry.get('relationships', []):
                    source = rel.get('source')
                    target = rel.get('target')
                    rel_type = rel.get('type')
                    
                    # Skip if relationship is missing required fields
                    if not all([source, target, rel_type]):
                        continue
                        
                    properties = rel.get('properties', {})
                    
                    # Add source information to properties
                    if doc_source:
                        properties['source_document'] = doc_source
                        
                    # Create Relationship object
                    self.relationships.append(Relationship(
                        source,
                        target,
                        rel_type,
                        properties
                    ))
                    
            except Exception as e:
                print(f"Error parsing entry {entry_idx}: {str(e)}")
                
        print(f"Parsed {len(self.nodes)} nodes and {len(self.relationships)} relationships")
        return self.nodes, self.relationships