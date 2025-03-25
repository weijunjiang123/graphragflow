import time
import logging
from typing import List, Tuple, Optional
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.base_language import BaseLanguageModel

logger = logging.getLogger(__name__)

class GraphTransformerWrapper:
    """Wrapper for LLMGraphTransformer to provide additional functionality"""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize graph transformer wrapper
        
        Args:
            llm: Language model to use for graph transformation
        """
        self.llm = llm
        self.transformer = LLMGraphTransformer(llm=llm)
        
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
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:min(i+batch_size, len(documents))]
            
            # Update progress
            progress = min(i+batch_size, len(documents))
            percentage = (progress / total_docs) * 100
            elapsed = time.time() - start_time
            eta = (elapsed / progress) * (total_docs - progress) if progress > 0 else 0
            
            print(f"\rProcessing: {progress}/{total_docs} documents ({percentage:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s - ETA: {eta:.1f}s", end="")
            
            # Process batch
            batch_results = self.transformer.convert_to_graph_documents(batch)
            graph_documents.extend(batch_results)
        
        print(f"\n✓ Conversion complete: {len(graph_documents)} graph documents created in {time.time()-start_time:.1f}s")
        logger.info(f"Created {len(graph_documents)} graph documents")
        
        return graph_documents, self.llm
