import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_cognitive_stream():
    print("1. Ingesting Raw Information Stream (Cognitive Pipeline)...")
    
    # Simulating a raw thought or snippet
    raw_content = """
    Project Chronos: A new initiative to map human memory to digital storage.
    Key components: 
    1. Neural Link Interface (NLI) for direct extraction.
    2. Temporal Database for storing time-series memory data.
    3. Emotion Tagging Engine using amygdala resonance.
    Hypothesis: If we can tag memories with emotion, retrieval accuracy increases by 400%.
    """
    
    res = requests.post(f"{BASE_URL}/ingest/stream", json={
        "content": raw_content,
        "source_type": "research_note",
        "session_id": None # Global ingestion
    })
    
    print(f"Ingest Response: {res.json()}")
    
    print("\n2. Waiting for Cognitive Processing (Thinking time)...")
    time.sleep(15)
    
    print("\n3. Querying to verify Understanding & Reasoning...")
    questions = [
        "What is Project Chronos?",
        "How does emotion affect memory retrieval according to the hypothesis?",
        "What are the key components of the memory initiative?"
    ]
    
    for q in questions:
        print(f"\nQ: {q}")
        res = requests.post(f"{BASE_URL}/query", json={
            "question": q,
            "mode": "hybrid"
        })
        answer = res.json().get('answer')
        print(f"A: {answer[:200]}...") # Truncate for display

if __name__ == "__main__":
    try:
        test_cognitive_stream()
    except Exception as e:
        print(f"Test failed: {e}")
