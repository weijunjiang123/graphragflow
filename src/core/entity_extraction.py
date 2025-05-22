import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from langchain.base_language import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

class StructuredEntity(BaseModel):
    """Represents a single extracted entity with its type."""
    name: str = Field(..., description="The text of the extracted entity.")
    type: str = Field(..., description="The type of the entity (e.g., PERSON, ORGANIZATION).")

class Entities(BaseModel):
    """Identifying information about entities."""
    entities: List[StructuredEntity] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text, along with their types.",
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
                (
                    "You are an expert in extracting organization and person entities from text. "
                    "For each entity found, identify its name and its type (e.g., PERSON, ORGANIZATION, LOCATION, etc.)."
                    "Return a list of these entities, where each entity is an object with 'name' and 'type' fields."
                ),
            ),
            (
                "human",
                "Extract all person and organization entities from the following text: {question}"
            ),
        ])
        
        # Ensure the LLM is instructed to output according to the `Entities` Pydantic model,
        # which now expects a list of `StructuredEntity` objects.
        entity_chain = self.llm.with_structured_output(Entities)
        print("✓ Entity extraction chain ready (with structured output)")
        return entity_chain
        
    def extract(self, text: str) -> Entities:
        """Extract entities from text
        
        Args:
            text: Text to extract entities from
            
        Returns:
            An Entities object containing a list of StructuredEntity.
        """
        logger.info(f"Extracting entities from text: {text[:100]}...")
        # The chain is already set up to return an `Entities` object due to `with_structured_output(Entities)`
        extracted_data = self.chain.invoke({"question": text}) # Pass text as 'question' matching prompt
        logger.info(f"Extracted entities: {extracted_data}")
        return extracted_data
