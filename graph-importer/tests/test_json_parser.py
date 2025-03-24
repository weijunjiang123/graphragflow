import json
import unittest
from src.importer.json_parser import JsonParser

class TestJsonParser(unittest.TestCase):

    def setUp(self):
        self.parser = JsonParser()

    def test_parse_valid_json(self):
        json_data = '''
        [
            {
                "nodes": [
                    {
                        "id": "Test Node",
                        "type": "TestType",
                        "properties": {}
                    }
                ],
                "relationships": [],
                "source_document": {
                    "page_content": "Test content",
                    "metadata": {
                        "source": "test_source.txt"
                    }
                }
            }
        ]
        '''
        parsed_data = self.parser.parse(json.loads(json_data))
        self.assertEqual(len(parsed_data['nodes']), 1)
        self.assertEqual(parsed_data['nodes'][0]['id'], "Test Node")

    def test_parse_empty_json(self):
        json_data = '[]'
        parsed_data = self.parser.parse(json.loads(json_data))
        self.assertEqual(parsed_data, {'nodes': [], 'relationships': []})

    def test_parse_invalid_json(self):
        with self.assertRaises(ValueError):
            self.parser.parse("Invalid JSON")

if __name__ == '__main__':
    unittest.main()