import asyncio
import os
from openai import AsyncOpenAI
import numpy as np

async def test_dim():
    api_key = os.getenv("SILICONFLOW_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
    if not api_key:
        raise ValueError("Set SILICONFLOW_API_KEY or LLM_BINDING_API_KEY for this script.")
    base_url = "https://api.siliconflow.cn/v1"
    model = "BAAI/bge-m3" # SiliconFlow model name example
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    try:
        response = await client.embeddings.create(
            input=["hello world"],
            model=model
        )
        dim = len(response.data[0].embedding)
        print(f"Embedding dimension for {model}: {dim}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_dim())
