# src/models.py

import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

def get_llm(model_name: str, api_key_env: str, base_url: str = None, temperature: float = 0.2):
    """Initializes and returns an OpenAI-compatible LLM."""
    api_key = os.getenv(api_key_env)
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        raise ValueError(f"{api_key_env} environment variable is not set correctly.")
    
    return ChatOpenAI(
        model=model_name, 
        temperature=temperature, 
        openai_api_key=api_key,
        base_url=base_url
    )

def get_embeddings(model_name: str = None, api_key_env: str = "OPENAI_API_KEY_EMBED", base_url: str = None, chunk_size: int = 1000):
    """
    Initializes and returns OpenAI-compatible embeddings.
    - chunk_size: The number of documents to send in a single batch. 
    """
    api_key = os.getenv(api_key_env)
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY1") or os.getenv("OPENAI_API_KEY")
        
    if not api_key or api_key == "YOUR_OPENAI_API_KEY":
        raise ValueError("No valid API key found for Embeddings. Please set OPENAI_API_KEY_EMBED.")
            
    # CRITICAL FIX for DashScope: 
    # 1. chunk_size=1 ensures pure string sending.
    # 2. check_embedding_ctx_length=False prevents local tokenizer errors.
    is_dashscope = base_url and "dashscope" in base_url
    
    # Smart default model names
    if not model_name or model_name == "text-embedding-3-small":
        if is_dashscope:
            model_name = "text-embedding-v3"
        else:
            model_name = "text-embedding-3-small"
    
    final_chunk_size = 1 if is_dashscope else chunk_size
    
    return OpenAIEmbeddings(
        model=model_name, 
        openai_api_key=api_key, 
        base_url=base_url,
        chunk_size=final_chunk_size,
        check_embedding_ctx_length=not is_dashscope, # Disable for DashScope
        max_retries=3
    )
