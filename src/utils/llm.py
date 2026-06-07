from openai import OpenAI
from pydantic import BaseModel
import os
from typing import Optional, Type, List, Dict, Any

def _build_client(api_key: Optional[str] = None) -> OpenAI:
    """Return an OpenAI client, preferring the supplied key then env var."""
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key found. Pass api_key= or set OPENAI_API_KEY."
        )
    return OpenAI(api_key=key)

def parse_chat_completion(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Type[BaseModel],
    api_key: Optional[str] = None
) -> BaseModel:
    """
    Handles the boilerplate of creating a client, sending a structured prompt,
    and extracting the parsed response.
    """
    client = _build_client(api_key)
    
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=response_format,
    )
    
    # Return the fully parsed Pydantic object directly
    return completion.choices[0].message.parsed



def get_embedding(text: str, 
    model: str = "text-embedding-3-small",
    api_key: Optional[str] = None):
    """
    Get embedding for text using OpenAI API.

    Args:
        text: Text to embed
        model: Embedding model to use

    Returns:
        List of floats representing the embedding
    """
    client = _build_client(api_key)
    response = client.embeddings.create(
        input=text,
        model=model
    )

    return response.data[0].embedding