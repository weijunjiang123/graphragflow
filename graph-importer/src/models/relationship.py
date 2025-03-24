class Relationship:
    """Represents a relationship to be imported into Neo4j."""
    
    def __init__(self, source_id, target_id, type, properties=None):
        """Initialize a relationship.
        
        Args:
            source_id (str): ID of the source node
            target_id (str): ID of the target node
            type (str): Relationship type
            properties (dict, optional): Relationship properties
        """
        self.source_id = source_id
        self.target_id = target_id
        self.type = type
        self.properties = properties or {}
        
    def __str__(self):
        return f"Relationship({self.source_id})-[{self.type}]->({self.target_id})"