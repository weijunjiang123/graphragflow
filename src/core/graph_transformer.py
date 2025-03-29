import time
import logging
import re
import json
from typing import List, Tuple, Optional, Any, Dict, Union
from tqdm import tqdm
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.base_language import BaseLanguageModel

# Import the ModelResponseAdapter
from src.core.model_adapter import ModelResponseAdapter

logger = logging.getLogger(__name__)

class Node:
    """Class to represent a node in a graph"""
    def __init__(self, node_id: str, node_type: str, properties: Dict[str, Any] = None):
        self.id = node_id
        self.type = node_type
        self.properties = properties or {}
    
    @classmethod
    def from_dict(cls, node_dict: Dict[str, Any]) -> 'Node':
        """Create a Node from a dictionary
        
        Args:
            node_dict: Dictionary with 'id', 'type', and 'properties'
            
        Returns:
            Node instance
        """
        return cls(
            node_id=node_dict.get("id", ""),
            node_type=node_dict.get("type", "Entity"),
            properties=node_dict.get("properties", {})
        )

class Relationship:
    """Class to represent a relationship in a graph"""
    def __init__(self, source: str, target: str, rel_type: str, properties: Dict[str, Any] = None):
        self.source = source
        self.target = target
        self.type = rel_type
        self.properties = properties or {}
    
    @classmethod
    def from_dict(cls, rel_dict: Dict[str, Any]) -> 'Relationship':
        """Create a Relationship from a dictionary
        
        Args:
            rel_dict: Dictionary with 'source', 'target', 'type', and 'properties'
            
        Returns:
            Relationship instance
        """
        return cls(
            source=rel_dict.get("source", ""),
            target=rel_dict.get("target", ""),
            rel_type=rel_dict.get("type", "RELATED_TO"),
            properties=rel_dict.get("properties", {})
        )

@dataclass
class GraphDocument:
    """Class to represent a graph document with the expected structure"""
    nodes: List[Node] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    source: Union[Document, str] = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], source: Union[Document, str] = "") -> 'GraphDocument':
        """Create a GraphDocument from a dictionary
        
        Args:
            data: Dictionary with 'nodes' and 'relationships' keys
            source: Source document or string
            
        Returns:
            GraphDocument instance
        """
        # Convert dictionary nodes to Node objects
        nodes = [Node.from_dict(node) for node in data.get("nodes", [])]
        
        # Convert dictionary relationships to Relationship objects
        relationships = [Relationship.from_dict(rel) for rel in data.get("relationships", [])]
        
        return cls(
            nodes=nodes,
            relationships=relationships,
            source=source
        )

class GraphTransformerWrapper:
    """Wrapper for LLMGraphTransformer to provide additional functionality"""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize graph transformer wrapper
        
        Args:
            llm: Language model to use for graph transformation
        """
        self.llm = llm
        self.transformer = CustomLLMGraphTransformer(llm=llm)
        
    @staticmethod
    def clean_json_from_markdown(text: str) -> str:
        """Strip away markdown formatting from possible JSON response
        
        Args:
            text: Text that may contain JSON wrapped in markdown code blocks
            
        Returns:
            Cleaned JSON string
        """
        logger.debug(f"Cleaning text: {text[:100]}...")
        
        # First try to extract from markdown code blocks
        if "```" in text:
            # Extract content between code blocks (including json, python, or no language specified)
            pattern = r"```(?:json|python)?\s*\n?([\s\S]*?)\n?```"
            matches = re.findall(pattern, text)
            if matches:
                cleaned = matches[0].strip()
                logger.debug(f"Extracted from markdown: {cleaned[:100]}...")
                return cleaned
        
        # Second, try to handle cases where there might be partial code blocks
        if text.strip().startswith("```") and not text.strip().endswith("```"):
            pattern = r"```(?:json|python)?\s*\n?([\s\S]*)"
            matches = re.match(pattern, text.strip())
            if matches:
                cleaned = matches.group(1).strip()
                logger.debug(f"Extracted from incomplete markdown: {cleaned[:100]}...")
                return cleaned
                
        if not text.strip().startswith("```") and text.strip().endswith("```"):
            pattern = r"([\s\S]*?)\n?```$"
            matches = re.match(pattern, text.strip())
            if matches:
                cleaned = matches.group(1).strip()
                logger.debug(f"Extracted from trailing markdown: {cleaned[:100]}...")
                return cleaned
        
        # If no markdown found or extraction failed, try to find JSON object by brackets
        if "{" in text and "}" in text:
            try:
                # Find the first opening brace and last closing brace
                start_idx = text.find("{")
                end_idx = text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    potential_json = text[start_idx:end_idx]
                    # Validate if it's valid JSON
                    json.loads(potential_json)
                    logger.debug(f"Extracted by brackets: {potential_json[:100]}...")
                    return potential_json
            except json.JSONDecodeError:
                logger.debug("Failed to extract JSON by brackets")
                pass
        
        # If all else fails, return the original text
        logger.debug("Using original text as no clean-up method worked")
        return text
        
    def create_graph_from_documents(self, 
                                   documents: List[Document], 
                                   batch_size: int = 5) -> Tuple[List, BaseLanguageModel]:
        """Convert documents to graph documents with progress tracking
        
        Args:
            documents: List of documents to convert
            batch_size: Size of each batch
            
        Returns:
            Tuple of (graph_documents, llm)
        """
        total_docs = len(documents)
        logger.info(f"Initializing graph transformation of {total_docs} documents")
        print(f"Converting {total_docs} documents to graph format (this may take a while)...")
        
        # Process in batches with progress bar
        start_time = time.time()
        graph_documents = []

        with tqdm(total=len(documents), desc="Converting to graph format") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:min(i + batch_size, len(documents))]

                # Process batch
                try:
                    # Process each document individually with better error handling
                    batch_results = []
                    for doc in batch:
                        try:
                            result = self.transformer.convert_to_graph_documents([doc])
                            if result:
                                batch_results.extend(result)
                        except Exception as e:
                            logger.error(f"Error processing document: {str(e)}")
                            print(f"⚠️ Skipping document due to processing error: {str(e)}")
                            continue
                    
                    graph_documents.extend(batch_results)
                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    print(f"❌ Error processing batch: {str(e)}")
                    # Skip the batch and continue
                    continue

                # Update progress
                pbar.update(len(batch))

        print(f"✓ Conversion complete: {len(graph_documents)} graph documents created in {time.time() - start_time:.1f}s")
        logger.info(f"Created {len(graph_documents)} graph documents")

        return graph_documents, self.llm


class CustomLLMGraphTransformer(LLMGraphTransformer):
    """Custom graph transformer that handles markdown-formatted responses"""
    
    def __init__(self, llm: BaseLanguageModel, **kwargs):
        """Initialize with explicit LLM storage"""
        super().__init__(llm=llm, **kwargs)
        self._llm = llm  # Store LLM explicitly as instance variable
    
    def convert_to_graph_documents(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """Override to handle markdown-formatted JSON responses
        
        Args:
            documents: List of documents to convert
            
        Returns:
            List of graph documents
        """
        # Try custom conversion first instead of falling back to it
        try:
            # Get the raw string result from the language model
            raw_result = self._get_raw_response(documents)
            logger.debug(f"Raw LLM response: {raw_result[:200]}...")
            
            # Clean any markdown formatting
            cleaned_json = GraphTransformerWrapper.clean_json_from_markdown(raw_result)
            logger.debug(f"Cleaned JSON: {cleaned_json[:200]}...")
            
            # Try to parse the JSON
            try:
                graph_data = json.loads(cleaned_json)
            except json.JSONDecodeError as json_err:
                logger.warning(f"JSON parsing error: {str(json_err)}")
                # Last resort: try to fix common JSON issues
                cleaned_json = self._fix_json_format(cleaned_json)
                logger.debug(f"Fixed JSON format: {cleaned_json[:200]}...")
                graph_data = json.loads(cleaned_json)
            
            # Validate the graph data structure
            if not isinstance(graph_data, dict):
                raise ValueError(f"Expected a dictionary, got {type(graph_data)}")
            
            if "nodes" not in graph_data or "relationships" not in graph_data:
                logger.warning("Missing 'nodes' or 'relationships' in graph data, trying to restructure")
                graph_data = self._restructure_graph_data(graph_data)
            
            # Process the manually parsed graph data
            return self._process_graph_data(graph_data, documents)
        except Exception as e:
            logger.warning(f"Custom conversion failed: {str(e)}. Trying standard conversion...")
            
            # If the custom handling fails, try the original implementation as fallback
            try:
                return super().convert_to_graph_documents(documents)
            except Exception as e2:
                logger.error(f"Both custom and standard conversion failed: {str(e2)}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                raise
    
    def _get_raw_response(self, documents: List[Document]) -> str:
        """Get the raw string response from the language model
        
        Args:
            documents: List of documents to convert
            
        Returns:
            Raw string response
        """
        # Use the explicitly stored LLM
        prompt = self._build_prompt(documents)
        # Add explicit instructions to return raw JSON
        prompt = ModelResponseAdapter.format_prompt_for_json(prompt)
        
        logger.debug(f"Sending prompt: {prompt[:200]}...")
        response = self._llm.invoke(prompt)
        
        # Use the adapter to clean the response
        return ModelResponseAdapter.clean_llm_response(response)
    
    def _build_prompt(self, documents: List[Document]) -> str:
        """Build the prompt for the LLM
        
        Args:
            documents: List of documents to convert
            
        Returns:
            String prompt
        """
        # Get document content with truncation for very large documents
        text_contents = []
        for doc in documents:
            content = doc.page_content
            # Truncate very long documents to prevent context window issues
            if len(content) > 8000:  # Arbitrary limit, adjust based on your model
                content = content[:8000] + "... [truncated]"
            text_contents.append(content)
            
        text_content = "\n\n".join(text_contents)
        
        prompt = (
            "You are a knowledge graph expert. Extract entities and relationships from the following text.\n\n"
            f"TEXT: {text_content}\n\n"
            "INSTRUCTIONS:\n"
            "1. Identify all entities (people, organizations, concepts, locations, etc.)\n"
            "2. Identify relationships between these entities\n"
            "3. Return the result as a JSON object with this exact structure:\n"
            "{\n"
            '  "nodes": [\n'
            '    {"id": "unique_id", "type": "EntityType", "properties": {"name": "Name of entity"}},\n'
            '    ...\n'
            '  ],\n'
            '  "relationships": [\n'
            '    {"source": "source_id", "target": "target_id", "type": "RELATIONSHIP_TYPE"},\n'
            '    ...\n'
            '  ]\n'
            "}\n\n"
            "Every node MUST have a unique 'id' field, a 'type' field, and a 'properties' object with at least a 'name'.\n"
            "Every relationship MUST have 'source' and 'target' that match node ids, and a 'type' field.\n"
            "IMPORTANT: Return ONLY the JSON object. No explanations, no markdown formatting with ``` tags.\n"
            "DO NOT wrap your response in code blocks, backticks, or markdown formatting.\n"
            "Just return the raw JSON data directly."
        )
        return prompt
    
    def _create_source_document(self, original_doc: Document, doc_id: str) -> Document:
        """Create a source document with proper metadata
        
        Args:
            original_doc: Original document
            doc_id: Document ID
            
        Returns:
            Document with proper metadata
        """
        # Create a new document with the same content but with proper metadata
        metadata = dict(original_doc.metadata) if hasattr(original_doc, "metadata") else {}
        
        # Ensure ID is present in metadata
        if "id" not in metadata:
            metadata["id"] = doc_id
            
        # Create a new Document with the same content but updated metadata
        return Document(
            page_content=original_doc.page_content if hasattr(original_doc, "page_content") else "",
            metadata=metadata
        )
    
    def _process_graph_data(self, graph_data: Dict[str, Any], documents: List[Document]) -> List[Any]:
        """Process the parsed graph data into the expected format
        
        Args:
            graph_data: Parsed JSON data
            documents: Original documents
            
        Returns:
            List of graph documents
        """
        try:
            # Validate required keys
            if "nodes" not in graph_data or "relationships" not in graph_data:
                raise ValueError(f"Missing required keys in graph data: {graph_data.keys()}")
            
            # Validate nodes and relationships
            if not isinstance(graph_data["nodes"], list) or not isinstance(graph_data["relationships"], list):
                raise ValueError(f"'nodes' and 'relationships' must be lists")
            
            # Create the result
            result = []
            
            # Create graph documents with proper source
            for idx, doc in enumerate(documents):
                # Create a Document object with proper metadata for source
                source_doc = self._create_source_document(doc, f"document_{idx}")
                
                # Create a GraphDocument instance with the Document as source
                # This will automatically convert dictionaries to Node and Relationship objects
                graph_doc = GraphDocument.from_dict(graph_data, source=source_doc)
                
                # Log some info about the nodes and relationships
                logger.debug(f"Created graph document with {len(graph_doc.nodes)} nodes and {len(graph_doc.relationships)} relationships")
                if graph_doc.nodes:
                    logger.debug(f"First node type: {type(graph_doc.nodes[0])}, attrs: {dir(graph_doc.nodes[0])[:10]}")
                
                result.append(graph_doc)
                
            logger.info(f"Created {len(result)} graph document objects")
            return result
        except Exception as e:
            logger.error(f"Error processing graph data: {str(e)}")
            logger.debug(f"Graph data keys: {graph_data.keys() if isinstance(graph_data, dict) else 'Not a dict'}")
            raise
    
    def _fix_json_format(self, json_str: str) -> str:
        """Attempt to fix common JSON formatting issues
        
        Args:
            json_str: Potentially malformed JSON string
            
        Returns:
            Fixed JSON string (hopefully)
        """
        # Replace single quotes with double quotes
        fixed = json_str.replace("'", '"')
        
        # Remove trailing commas in arrays and objects
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Ensure property names are quoted
        def quote_keys(match):
            key = match.group(1)
            if key.startswith('"') and key.endswith('"'):
                return match.group(0)
            return f'"{key}":'
            
        fixed = re.sub(r'([a-zA-Z0-9_]+):', quote_keys, fixed)
        
        return fixed
    
    def _restructure_graph_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to restructure data into the expected graph format
        
        Args:
            data: Dictionary that might contain graph data in a different structure
            
        Returns:
            Restructured data with nodes and relationships
        """
        # If we already have the right structure, just return it
        if "nodes" in data and "relationships" in data:
            return data
            
        # Initialize empty result with the expected structure
        result = {"nodes": [], "relationships": []}
        
        # Try to find nodes and relationships in the data
        for key, value in data.items():
            if isinstance(value, list):
                if key.lower() in ["nodes", "entities", "vertices"]:
                    result["nodes"] = value
                elif key.lower() in ["relationships", "relations", "edges"]:
                    result["relationships"] = value
                    
        # If we couldn't find nodes or relationships, create a minimal structure
        if not result["nodes"]:
            logger.warning("Could not find nodes in the data, creating minimal structure")
            # Create at least one node
            result["nodes"] = [{"id": "node1", "type": "Entity", "properties": {"name": "Unknown"}}]
            
        return result
