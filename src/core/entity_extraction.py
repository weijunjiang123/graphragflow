import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain.base_language import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class Entities(BaseModel):
    """Identifying information about entities."""
    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

class EntityExtractor:
    """Handler for entity extraction from text"""
    
    def __init__(self, llm: BaseLanguageModel):
        """Initialize entity extractor
        
        Args:
            llm: Language model to use for entity extraction
        """
        self.llm = llm
        self.chain = self._setup_extraction_chain()
        
    def _setup_extraction_chain(self):
        """Set up the entity extraction chain
        
        Returns:
            Entity extraction chain
        """
        logger.info("Setting up entity extraction chain")
        print("Setting up entity extraction chain...")
        
        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are extracting organization and person entities from the text.",
            ),
            (
                "human",
                "Use the given format to extract information from the following "
                "input: {question}",
            ),
        ])
        
        entity_chain = self.llm.with_structured_output(Entities)
        print("✓ Entity extraction chain ready")
        return entity_chain
        
    def extract(self, text: str) -> Entities:
        """Extract entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Extracted entities
        """
        return self.chain.invoke(text)
