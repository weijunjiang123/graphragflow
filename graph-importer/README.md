# Graph Importer

This project provides a tool for importing data from JSON files into a Neo4j graph database. It includes utilities for parsing JSON data, connecting to the Neo4j database, and importing nodes and relationships.

## Features

- **JSON Parsing**: Efficiently parse JSON data to extract nodes and relationships.
- **Neo4j Integration**: Connect to a Neo4j database and import parsed data.
- **Modular Design**: Organized into packages for easy maintenance and scalability.

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd graph-importer
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To run the importer, execute the following command:

```
python src/main.py <path_to_json_file>
```

Replace `<path_to_json_file>` with the path to your JSON file.

## Configuration

Edit the `src/config.py` file to set your Neo4j database connection details:

```python
DATABASE_URI = "bolt://localhost:7687"
USERNAME = "neo4j"
PASSWORD = "your_password"
```

## Testing

To run the tests, use the following command:

```
pytest
```

## Example

An example of how to use the importer can be found in the `examples/sample_import.py` file.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.