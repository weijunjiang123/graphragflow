import sys
import os
import json
from pathlib import Path  # Use pathlib for path handling

# Get the current script's absolute path
script_path = os.path.abspath(__file__)
# Get the current script's directory
script_dir = os.path.dirname(script_path)
# Add the script's directory to the module search path
sys.path.append(script_dir)

from config import DATABASE_URI, USERNAME, PASSWORD
from importer.neo4j_importer import Neo4jImporter
from importer.json_parser import JsonParser

def main():
    # Use pathlib for cross-platform path handling
    data_path = Path(script_dir).parent / "data" / "graph_documents_20250324_215838.json"
    
    # Ensure the file exists
    if not data_path.exists():
        print(f"Error: File not found at {data_path}")
        return
    
    # Load JSON data from a file
    print(f"Loading data from {data_path}")
    with open(data_path, 'r', encoding='utf-8') as file:
        json_data = json.load(file)

    # Initialize the JSON parser
    parser = JsonParser(json_data)

    # Parse nodes and relationships
    nodes, relationships = parser.parse()
    print(f"Parsed {len(nodes)} nodes and {len(relationships)} relationships")

    # Initialize the Neo4j importer
    neo4j_importer = Neo4jImporter(DATABASE_URI, USERNAME, PASSWORD)

    try:
        # Import nodes and relationships into Neo4j
        neo4j_importer.import_nodes(nodes)
        neo4j_importer.import_relationships(relationships)
        print("Import completed successfully")
    except Exception as e:
        print(f"Error during import: {str(e)}")
    finally:
        # Close the Neo4j connection
        neo4j_importer.close()

if __name__ == "__main__":
    main()