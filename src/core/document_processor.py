import os
import logging
from typing import List, Callable, Optional
from tqdm import tqdm

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handler for document loading and processing"""
    
    def __init__(self, chunk_size, chunk_overlap):
        """Initialize document processor
        
        Args:
            chunk_size: Size of document chunks
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Initializing DocumentProcessor with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
        self.text_splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
    def load_and_split(self, file_path: str) -> List[Document]:
        """Load and split documents into chunks with progress indication
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of document chunks
            
        Raises:
            FileNotFoundError: If document file is not found
        """
        logger.info(f"Loading documents from {file_path}")
        
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        file_name = os.path.basename(file_path)
        logger.info(f"Processing document: {file_name}")


        if file_name.endswith(".txt"):
            # Load text document
            loader = TextLoader(file_path=file_path)
            docs = loader.load()
            logger.info(f"Loaded document with {len(docs)} pages")

        elif file_name.endswith(".pdf"):
            # Load PDF document
            loader = PyPDFLoader(file_path=file_path, mode="single")
            docs = loader.load()
            logger.info(f"Loaded document with {len(docs)} pages")
        
        elif file_name.endswith(".docx"):
            # Load Word document
            loader = Docx2txtLoader(file_path=file_path)
            docs = loader.load()
            logger.info(f"Loaded document with {len(docs)} pages")
        
        else:
            raise ValueError(f"Unsupported file type: {file_name}")
            
        # Show progress during splitting
        print(f"Splitting document into chunks (size={self.chunk_size}, overlap={self.chunk_overlap})...")
        documents = self.text_splitter.split_documents(documents=docs)
        
        logger.info(f"Split documents into {len(documents)} chunks")
        print(f"✓ Document processing complete: {len(documents)} chunks created")
        
        return documents
        
    def batch_process(self, 
                      documents: List[Document], 
                      batch_size: int = 5, 
                      process_fn: Optional[Callable] = None) -> List:
        """Process documents in batches with progress bar
        
        Args:
            documents: List of documents to process
            batch_size: Size of each batch
            process_fn: Function to process each batch of documents
            
        Returns:
            List of processed documents
        """
        results = []
        
        with tqdm(total=len(documents), desc="Processing documents") as pbar:
            for i in range(0, len(documents), batch_size):
                batch = documents[i:min(i+batch_size, len(documents))]
                
                # Process the batch
                if process_fn:
                    batch_results = process_fn(batch)
                    results.extend(batch_results)
                else:
                    results.extend(batch)
                    
                # Update progress
                pbar.update(len(batch))
                
        return results
