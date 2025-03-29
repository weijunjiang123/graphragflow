import logging
import re
import json
from typing import Any, Dict, Union, Optional

logger = logging.getLogger(__name__)

class ModelResponseAdapter:
    """Utility class to handle various LLM response formats and standardize them"""
    
    @staticmethod
    def clean_llm_response(response: Any) -> str:
        """Convert various LLM response types to a standardized string
        
        Args:
            response: Response from LLM, could be string, object with content attribute, etc.
            
        Returns:
            Cleaned string response
        """
        # Handle different response types
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif hasattr(response, "message"):
            if hasattr(response.message, "content"):
                return response.message.content
        elif hasattr(response, "text"):
            return response.text
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        
        # If we can't determine the type, convert to string
        return str(response)
    
    @staticmethod
    def format_prompt_for_json(prompt: str) -> str:
        """Enhance a prompt to encourage the model to output valid JSON
        
        Args:
            prompt: Original prompt
            
        Returns:
            Enhanced prompt
        """
        # Add explicit instructions to return valid JSON
        json_instructions = "\n\nIMPORTANT: Your response must be a valid JSON object. Do not include markdown formatting, explanations, or any text outside the JSON object."
        
        # Add instructions about escaping special characters
        json_instructions += "\nEnsure all quotes, backslashes, and other special characters within strings are properly escaped according to JSON standards."
        
        # Add examples of good vs bad responses
        json_instructions += "\n\nGOOD RESPONSE EXAMPLE:\n{\"property\": \"value\"}\n\nBAD RESPONSE EXAMPLE:\n```json\n{\"property\": \"value\"}\n```"
        
        return prompt + json_instructions
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[str]:
        """Extract JSON object from text that may contain other content
        
        Args:
            text: Text that may contain JSON mixed with other content
            
        Returns:
            Extracted JSON string or None if extraction fails
        """
        # Look for JSON object pattern (from first { to last })
        start_idx = text.find('{')
        if start_idx == -1:
            return None
            
        # Count braces to find proper closing brace
        brace_count = 0
        in_quotes = False
        escaped = False
        
        for i in range(start_idx, len(text)):
            char = text[i]
            
            # Handle escape sequences
            if char == '\\' and not escaped:
                escaped = True
                continue
                
            # Handle quotes (considering escaping)
            if char == '"' and not escaped:
                in_quotes = not in_quotes
                
            # Only count braces when not in quotes
            if not in_quotes:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        # We found the closing brace
                        return text[start_idx:i+1]
            
            escaped = False
        
        return None
    
    @staticmethod
    def fix_common_json_errors(json_str: str) -> str:
        """Fix common errors in JSON strings
        
        Args:
            json_str: JSON string that may contain errors
            
        Returns:
            Fixed JSON string
        """
        # Replace single quotes with double quotes
        fixed = re.sub(r"(?<!')('|\')(?!')", '"', json_str)
        
        # Remove trailing commas in lists and objects
        fixed = re.sub(r',\s*}', '}', fixed)
        fixed = re.sub(r',\s*]', ']', fixed)
        
        # Fix unquoted property names (convert words followed by colon to quoted property names)
        fixed = re.sub(r'([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'"\1"\2', fixed)
        
        # Fix missing commas between array elements or object properties
        fixed = re.sub(r'}\s*{', '},{', fixed)
        fixed = re.sub(r']\s*{', '],{', fixed)
        fixed = re.sub(r'}\s*[', '},[', fixed)
        fixed = re.sub(r']\s*[', '],[', fixed)
        
        return fixed
    
    @staticmethod
    def parse_json_safely(text: str) -> Optional[Dict[str, Any]]:
        """Attempt to parse JSON with several fallback strategies
        
        Args:
            text: String that should contain JSON
            
        Returns:
            Parsed JSON dictionary or None if parsing fails
        """
        # Try direct parsing first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.debug("Direct JSON parsing failed, trying to clean the input")
            
        # Remove markdown code blocks if present
        if "```" in text:
            markdown_pattern = r"```(?:json|python)?\s*\n?([\s\S]*?)\n?```"
            matches = re.findall(markdown_pattern, text)
            if matches:
                cleaned_text = matches[0].strip()
                try:
                    return json.loads(cleaned_text)
                except json.JSONDecodeError:
                    logger.debug("JSON parsing from markdown failed")
        
        # Try to extract JSON from mixed content
        extracted_json = ModelResponseAdapter.extract_json_from_text(text)
        if extracted_json:
            try:
                return json.loads(extracted_json)
            except json.JSONDecodeError:
                logger.debug("Extracted JSON parsing failed")
        
        # Try to fix common JSON errors
        fixed_text = ModelResponseAdapter.fix_common_json_errors(text)
        try:
            return json.loads(fixed_text)
        except json.JSONDecodeError:
            logger.debug("Fixed JSON parsing failed")
            
        # All attempts failed
        logger.error(f"Failed to parse JSON from text: {text[:100]}...")
        return None
