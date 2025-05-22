import logging
from typing import List
from fastapi import APIRouter, Depends, HTTPException

from src.api.models import EntityExtractRequest, EntityExtractResponse, Entity as ApiEntity
from src.api.main import get_entity_extractor
from src.core.entity_extraction import EntityExtractor, StructuredEntity as CoreEntity # Renamed for clarity

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/entities/extract", response_model=EntityExtractResponse)
async def extract_entities_endpoint(
    request: EntityExtractRequest,
    entity_extractor: EntityExtractor = Depends(get_entity_extractor)
):
    """
    Extracts structured entities (name and type) from the provided text.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    logger.info(f"Received request to extract entities from text: {request.text[:100]}...")

    try:
        # The core EntityExtractor now returns an `Entities` object
        # which contains a list of `StructuredEntity` objects.
        core_entities_obj = entity_extractor.extract(request.text)
        
        # Map core entities to API response entities
        api_entities_list: List[ApiEntity] = []
        if core_entities_obj and core_entities_obj.entities:
            for core_entity in core_entities_obj.entities:
                api_entities_list.append(
                    ApiEntity(text=core_entity.name, label_=core_entity.type)
                )
        
        logger.info(f"Successfully extracted {len(api_entities_list)} entities.")
        
        return EntityExtractResponse(
            text=request.text,
            entities=api_entities_list
        )

    except Exception as e:
        logger.error(f"Error during entity extraction: {e}", exc_info=True)
        # More specific error handling can be added here if needed
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during entity extraction: {str(e)}")

# Example of how to include this router in main.py (already done in a previous step):
# from .routers import entities
# app.include_router(entities.router, prefix="/api/v1", tags=["Entities"])
