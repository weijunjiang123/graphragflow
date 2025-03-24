import unittest
from src.importer.neo4j_importer import Neo4jImporter

class TestNeo4jImporter(unittest.TestCase):

    def setUp(self):
        self.importer = Neo4jImporter(uri="bolt://localhost:7687", user="neo4j", password="password")

    def test_connection(self):
        self.assertTrue(self.importer.connect())

    def test_import_nodes(self):
        data = {
            "nodes": [
                {"id": "Test Node", "type": "TestType", "properties": {"key": "value"}}
            ],
            "relationships": []
        }
        result = self.importer.import_nodes(data)
        self.assertTrue(result)

    def test_import_relationships(self):
        data = {
            "nodes": [
                {"id": "Node1", "type": "Type1", "properties": {}},
                {"id": "Node2", "type": "Type2", "properties": {}}
            ],
            "relationships": [
                {"source": "Node1", "target": "Node2", "type": "CONNECTED_TO", "properties": {}}
            ]
        }
        self.importer.import_nodes(data)  # Import nodes first
        result = self.importer.import_relationships(data)
        self.assertTrue(result)

    def tearDown(self):
        self.importer.close()

if __name__ == '__main__':
    unittest.main()