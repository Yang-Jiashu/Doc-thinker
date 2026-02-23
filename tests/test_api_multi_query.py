import requests
import json

def test_multi_doc_query():
    url = "http://127.0.0.1:8000/query/multi-document"
    payload = {
        "question": "What is this file for?",
        "mode": "hybrid",
        "enable_rerank": True
    }
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_multi_doc_query()
