from neo4j import GraphDatabase

class Neo4jImporter:
    """Handles importing nodes and relationships into Neo4j."""
    
    def __init__(self, uri, username, password):
        """Initialize the Neo4j connection.
        
        Args:
            uri (str): Neo4j connection URI
            username (str): Neo4j username
            password (str): Neo4j password
        """
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        print(f"Connected to Neo4j at {uri}")
        
    def close(self):
        """Close the Neo4j connection."""
        if hasattr(self, 'driver'):
            self.driver.close()
            print("Neo4j connection closed")
        
    def import_nodes(self, nodes):
        """Import nodes into Neo4j.
        
        Args:
            nodes (list): List of Node objects
        """
        print(f"Importing {len(nodes)} nodes")
        
        # Process nodes in batches for better performance
        batch_size = 100
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i+batch_size]
            self._import_node_batch(batch)
            
        print("Node import completed")
            
    def _import_node_batch(self, nodes):
        """Import a batch of nodes.
        
        Args:
            nodes (list): Batch of Node objects
        """
        if not nodes:
            return
            
        with self.driver.session() as session:
            # Process each node in the batch
            for node in nodes:
                # Create node with labels and properties
                label = node.label
                properties = {k: v for k, v in node.properties.items() if v is not None}
                
                # Cypher query to merge node - this prevents duplicates
                query = f"""
                MERGE (n:{label} {{id: $id}})
                SET n += $properties
                """
                
                try:
                    session.run(query, id=node.id, properties=properties)
                except Exception as e:
                    print(f"Error importing node {node.id}: {str(e)}")
    
    def import_relationships(self, relationships):
        """Import relationships into Neo4j.
        
        Args:
            relationships (list): List of Relationship objects
        """
        print(f"Importing {len(relationships)} relationships")
        
        # Process relationships in batches for better performance
        batch_size = 100
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i+batch_size]
            self._import_relationship_batch(batch)
            
        print("Relationship import completed")
            
    def _import_relationship_batch(self, relationships):
        """Import a batch of relationships.
        
        Args:
            relationships (list): Batch of Relationship objects
        """
        if not relationships:
            return
            
        with self.driver.session() as session:
            # Process each relationship in the batch
            for rel in relationships:
                # Properties for the relationship
                properties = {k: v for k, v in rel.properties.items() if v is not None}
                
                # Cypher query to merge relationship - requires existing nodes
                query = f"""
                MATCH (source {{id: $source_id}})
                MATCH (target {{id: $target_id}})
                MERGE (source)-[r:{rel.type}]->(target)
                SET r += $properties
                """
                
                try:
                    session.run(
                        query, 
                        source_id=rel.source_id, 
                        target_id=rel.target_id, 
                        properties=properties
                    )
                except Exception as e:
                    print(f"Error importing relationship {rel}: {str(e)}")