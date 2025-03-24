class Node:
    """Represents a node to be imported into Neo4j."""
    
    def __init__(self, id, label, properties=None):
        """Initialize a node.
        
        Args:
            id (str): Unique identifier for the node
            label (str): Node label/type in Neo4j
            properties (dict, optional): Node properties
        """
        self.id = id
        self.label = label
        self.properties = properties or {}
        
        # Ensure id is included in properties
        self.properties['id'] = id
        
    def __str__(self):
        return f"Node(id={self.id}, label={self.label})"