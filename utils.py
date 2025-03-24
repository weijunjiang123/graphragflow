import datetime
import json
import os

from langchain_ollama import OllamaEmbeddings

def save_graph_documents(graph_docs, output_dir="results"):
    # Create a results directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate a timestamp for the filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/graph_documents_{timestamp}.json"
    
    # Convert graph documents to serializable format
    serializable_docs = []
    for doc in graph_docs:
        # Extract the necessary information from each graph document
        doc_dict = {
            "nodes": [],
            "relationships": [],
            "source_document": {
                "page_content": doc.source_document.page_content if hasattr(doc, 'source_document') else 
                               (doc.source.page_content if hasattr(doc, 'source') else None),
                "metadata": doc.source_document.metadata if hasattr(doc, 'source_document') else 
                           (doc.source.metadata if hasattr(doc, 'source') else None)
            } if (hasattr(doc, 'source_document') or hasattr(doc, 'source')) else None
        }
        
        # Add nodes with proper error handling
        for node in doc.nodes:
            node_dict = {"id": node.id}
            # Add type if available (sometimes called type instead of label)
            if hasattr(node, 'type'):
                node_dict["type"] = node.type
            elif hasattr(node, 'label'):
                node_dict["type"] = node.label
                
            # Add properties if available
            if hasattr(node, 'properties'):
                # Make sure properties are serializable
                try:
                    # Convert properties to a basic dict if needed
                    props = {}
                    for key, value in node.properties.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            props[key] = value
                        else:
                            props[key] = str(value)
                    node_dict["properties"] = props
                except Exception as e:
                    # If accessing properties fails, use empty dict
                    node_dict["properties"] = {}
            else:
                node_dict["properties"] = {}
            
            doc_dict["nodes"].append(node_dict)
        
        # Add relationships with proper error handling
        for rel in doc.relationships:
            rel_dict = {}
            
            # Extract source and target IDs (handle both string IDs and Node objects)
            if hasattr(rel, 'source'):
                if hasattr(rel.source, 'id'):  # If source is a Node object
                    rel_dict["source"] = rel.source.id
                else:  # If source is already an ID string
                    rel_dict["source"] = rel.source
                    
            if hasattr(rel, 'target'):
                if hasattr(rel.target, 'id'):  # If target is a Node object
                    rel_dict["target"] = rel.target.id
                else:  # If target is already an ID string
                    rel_dict["target"] = rel.target
                    
            if hasattr(rel, 'type'):
                rel_dict["type"] = rel.type
                
            # Add properties if available
            if hasattr(rel, 'properties'):
                # Make sure properties are serializable
                try:
                    # Convert properties to a basic dict if needed
                    props = {}
                    for key, value in rel.properties.items():
                        if isinstance(value, (str, int, float, bool, list, dict)) or value is None:
                            props[key] = value
                        else:
                            props[key] = str(value)
                    rel_dict["properties"] = props
                except Exception as e:
                    # If accessing properties fails, use empty dict
                    rel_dict["properties"] = {}
            else:
                rel_dict["properties"] = {}
            
            doc_dict["relationships"].append(rel_dict)
        
        serializable_docs.append(doc_dict)
    
    # Save to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
    
    print(f"Graph documents saved to {filename}")
    return filename

def get_working_embeddings():
    # Try Ollama embeddings with nomic-embed-text
    try:
        print("Trying nomic-embed-text for embeddings...")
        emb = OllamaEmbeddings(base_url="localhost:11434",model="nomic-embed-text")
        # Test if it works
        _ = emb.embed_query("test")
        print("Successfully using nomic-embed-text for embeddings")
        return emb
    except Exception as e:
        print(f"Error with nomic-embed-text: {str(e)}")
    
    
