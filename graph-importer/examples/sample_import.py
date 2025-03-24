import json
from src.importer.neo4j_importer import Neo4jImporter
from src.importer.json_parser import JsonParser

def main():
    # Load JSON data from file
    with open('path/to/your/graph_documents.json', 'r') as file:
        data = json.load(file)

    # Initialize the JSON parser
    parser = JsonParser()
    nodes, relationships = parser.parse(data)

    # Initialize the Neo4j importer
    neo4j_importer = Neo4jImporter(uri='neo4j://localhost:7687', user='your_username', password='your_password')

    # Import nodes and relationships into Neo4j
    neo4j_importer.import_nodes(nodes)
    neo4j_importer.import_relationships(relationships)

if __name__ == "__main__":
    main()