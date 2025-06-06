import logging
import os
from typing import Optional, Dict, Any, Union

from langchain.base_language import BaseLanguageModel
from langchain_experimental.llms.ollama_functions import OllamaFunctions
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings

logger = logging.getLogger(__name__)

class ModelProvider:
    """Provider for language models and embeddings"""
    
    @staticmethod
    def get_llm(provider: str = "ollama", **kwargs) -> Optional[BaseLanguageModel]:
        """Get a language model based on the specified provider
        
        Args:
            provider: Model provider ("ollama" or "openai")
            **kwargs: Additional parameters for model initialization
            
        Returns:
            Language model instance if successful, None otherwise
        """
        if provider.lower() == "ollama":
            model_name = kwargs.get("model_name", os.environ.get("OLLAMA_LLM_MODEL", "qwen2.5"))
            base_url = kwargs.get("base_url", "http://localhost:11434")
            temperature = kwargs.get("temperature", 0)
            
            try:
                logger.info(f"Initializing Ollama model: {model_name}")
                print(f"Initializing Ollama model: {model_name}")
                llm = OllamaFunctions(
                    model=model_name,
                    base_url=base_url,
                    temperature=temperature,
                    format="json"
                )
                # Test if the model works
                llm.invoke("test")
                logger.info(f"Successfully initialized Ollama model: {model_name}")
                print(f"✓ Successfully initialized Ollama model: {model_name}")
                return llm
            except Exception as e:
                logger.error(f"Error initializing Ollama model: {str(e)}")
                print(f"❌ Error initializing Ollama model: {str(e)}")
                return None
                
        elif provider.lower() == "openai":
            api_key = kwargs.get("api_key", os.environ.get("OPENAI_API_KEY"))
            api_base = kwargs.get("api_base", os.environ.get("OPENAI_API_BASE"))
            model_name = kwargs.get("model_name", os.environ.get("OPENAI_MODEL", "gpt-3.5-turbo"))
            temperature = kwargs.get("temperature", 0)
            
            if not api_key:
                logger.error("OpenAI API key not provided.")
                print("❌ OpenAI API key not provided. Please set OPENAI_API_KEY in your .env file.")
                return None
                
            try:
                logger.info(f"Initializing OpenAI model: {model_name}")
                print(f"Initializing OpenAI model: {model_name}")
                
                # Initialize with optional api_base
                params = {
                    "model": model_name,
                    "api_key": api_key,
                    "temperature": temperature,
                }
                
                # Only add response_format for compatible models (not for OpenRouter)
                is_openrouter = api_base and "openrouter.ai" in api_base
                if not is_openrouter:
                    params["model_kwargs"] = {"response_format": {"type": "json_object"}}
                
                if api_base:
                    params["base_url"] = api_base
                    
                llm = ChatOpenAI(**params)
                
                # Test if the model works, with better error handling
                try:
                    # Use a simple prompt that should work with any LLM
                    test_prompt = "Generate a one-word response: hello"
                    _ = llm.invoke(test_prompt)
                    logger.info(f"Successfully initialized OpenAI model: {model_name}")
                    print(f"✓ Successfully initialized OpenAI model: {model_name}")
                    return llm
                except Exception as test_error:
                    logger.warning(f"Model test failed: {str(test_error)}. Trying again without test...")
                    # Return the model anyway, assume it might work for actual use
                    return llm
                
            except Exception as e:
                logger.error(f"Error initializing OpenAI model: {str(e)}")
                print(f"❌ Error initializing OpenAI model: {str(e)}")
                return None
        else:
            logger.error(f"Unsupported model provider: {provider}")
            print(f"❌ Unsupported model provider: {provider}")
            return None
            
    @staticmethod
    def get_embeddings(provider: str = "ollama", **kwargs):
        """Get embeddings based on the specified provider
        
        Args:
            provider: Embeddings provider ("ollama" or "openai")
            model_name: Name of the model to use
            base_url: Base URL for the provider (if applicable)
            api_key: API key for the provider (if applicable)
            **kwargs: Additional parameters for embeddings initialization
            
        Returns:
            Embeddings instance if successful, None otherwise
        """
        # 从提供的参数或环境变量中获取嵌入模型提供者，默认使用与LLM相同的提供者
        provider = provider.lower() if provider else os.environ.get("MODEL_PROVIDER", "ollama").lower()
        
        if provider == "ollama":
            model_name = kwargs.get("model_name", os.environ.get("OLLAMA_EMBEDDINGS_MODEL", "nomic-embed-text"))
            base_url = kwargs.get("base_url", os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))
            
            # 有效的Ollama嵌入模型列表
            valid_models = ["nomic-embed-text", "llama3", "all-MiniLM-L6-v2"]
            
            try:
                logger.info(f"Initializing Ollama embeddings: {model_name}")
                print(f"Initializing Ollama embeddings: {model_name}")
                embeddings = OllamaEmbeddings(base_url=base_url, model=model_name)
                
                # Test if embeddings work
                _ = embeddings.embed_query("test")
                logger.info(f"Successfully initialized Ollama embeddings: {model_name}")
                print(f"✓ Successfully initialized Ollama embeddings: {model_name}")
                return embeddings
            except Exception as e:
                logger.error(f"Error initializing Ollama embeddings with {model_name}: {str(e)}")
                print(f"❌ Error initializing Ollama embeddings with {model_name}: {str(e)}")
                
                # 尝试其他Ollama嵌入模型
                for fallback_model in valid_models:
                    if fallback_model == model_name:
                        continue  # 跳过已经尝试过的模型
                    
                    try:
                        logger.info(f"Trying fallback Ollama embeddings model: {fallback_model}")
                        print(f"Trying fallback Ollama embeddings model: {fallback_model}")
                        fallback_embeddings = OllamaEmbeddings(base_url=base_url, model=fallback_model)
                        
                        # Test if embeddings work
                        _ = fallback_embeddings.embed_query("test")
                        logger.info(f"Successfully initialized fallback Ollama embeddings: {fallback_model}")
                        print(f"✓ Successfully initialized fallback Ollama embeddings: {fallback_model}")
                        return fallback_embeddings
                    except Exception as fallback_error:
                        logger.error(f"Error initializing fallback Ollama embeddings with {fallback_model}: {str(fallback_error)}")
                        print(f"❌ Error initializing fallback Ollama embeddings with {fallback_model}: {str(fallback_error)}")
                
                # 所有Ollama模型都失败，返回None
                return None
                
        elif provider == "openai":
            api_key = kwargs.get("api_key", os.environ.get("OPENAI_EMBEDDING_API_KEY") or os.environ.get("OPENAI_API_KEY"))
            api_base = kwargs.get("api_base", os.environ.get("OPENAI_EMBEDDING_API_BASE") or os.environ.get("OPENAI_API_BASE"))
            model_name = kwargs.get("model_name", os.environ.get("OPENAI_EMBEDDINGS_MODEL", "text-embedding-ada-002"))
            
            if not api_key:
                logger.error("OpenAI API key not provided.")
                print("❌ OpenAI API key not provided. Please set OPENAI_API_KEY or OPENAI_EMBEDDING_API_KEY in your .env file.")
                
                # 如果OpenAI失败，尝试回退到Ollama嵌入
                logger.info("Falling back to Ollama embeddings")
                print("Falling back to Ollama embeddings")
                return ModelProvider.get_embeddings(provider="ollama", **kwargs)
                
            try:
                logger.info(f"Initializing OpenAI embeddings: {model_name}")
                print(f"Initializing OpenAI embeddings: {model_name}")
                
                # Initialize with optional api_base
                params = {
                    "model": model_name,
                    "openai_api_key": api_key
                }
                
                if api_base:
                    params["openai_api_base"] = api_base
                    
                embeddings = OpenAIEmbeddings(**params)
                
                # Test if embeddings work
                _ = embeddings.embed_query("test")
                logger.info(f"Successfully initialized OpenAI embeddings: {model_name}")
                print(f"✓ Successfully initialized OpenAI embeddings: {model_name}")
                return embeddings
            except Exception as e:
                logger.error(f"Error initializing OpenAI embeddings: {str(e)}")
                print(f"❌ Error initializing OpenAI embeddings: {str(e)}")
                
                # 如果OpenAI失败，尝试回退到Ollama嵌入
                logger.info("Falling back to Ollama embeddings")
                print("Falling back to Ollama embeddings")
                return ModelProvider.get_embeddings(provider="ollama", **kwargs)
        else:
            logger.error(f"Unsupported embeddings provider: {provider}")
            print(f"❌ Unsupported embeddings provider: {provider}")
            
            # 对于不支持的提供者，尝试回退到Ollama
            logger.info("Falling back to Ollama embeddings")
            print("Falling back to Ollama embeddings")
            return ModelProvider.get_embeddings(provider="ollama", **kwargs)
