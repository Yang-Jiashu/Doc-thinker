import asyncio
import os
from openai import AsyncOpenAI
import numpy as np

async def test_embedding():
    api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("Set DASHSCOPE_API_KEY or LLM_BINDING_API_KEY for this test.")
    base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = "text-embedding-v4"
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    try:
        print(f"Testing embedding model: {model}")
        response = await client.embeddings.create(
            model=model,
            input="Hello world"
        )
        embedding = response.data[0].embedding
        print(f"Success! Embedding dimension: {len(embedding)}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_embedding())
