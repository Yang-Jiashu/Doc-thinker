import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_session_flow():
    print("Testing Session Flow...")
    
    # 1. Create Session
    print("1. Creating Session...")
    resp = requests.post(f"{BASE_URL}/sessions", json={"title": "Test Session"})
    if resp.status_code != 200:
        print(f"Failed to create session: {resp.text}")
        return
    
    session_data = resp.json()["session"]
    session_id = session_data["id"]
    print(f"Session Created: {session_id}")
    
    # 2. List Sessions
    print("2. Listing Sessions...")
    resp = requests.get(f"{BASE_URL}/sessions")
    sessions = resp.json()["sessions"]
    found = any(s["id"] == session_id for s in sessions)
    print(f"Session found in list: {found}")
    
    # 3. Upload File (Mocking a file upload)
    # Create a dummy file
    with open("test_doc.txt", "w") as f:
        f.write("Apple produces iPhone. Tim Cook is the CEO of Apple.")
        
    print("3. Uploading File to Session...")
    with open("test_doc.txt", "rb") as f:
        files = {'files': ('test_doc.txt', f)}
        # We need to pass session_id as query param for /ingest if we follow api_multi_document.py
        # api_multi_document.py: session_id: Optional[str] = Form(None)
        # So it should be data/form, not query param? 
        # In api_multi_document.py: ingest_files(files: List[UploadFile], session_id: Optional[str] = None)
        # FastAPI handles non-Body params as query params usually unless explicitly Form.
        # Let's check api_multi_document.py again. 
        # It says: session_id: Optional[str] = None
        # It doesn't say Form(...) explicitly in the signature I wrote?
        # Wait, I wrote: session_id: Optional[str] = None
        # For UploadFile, other params usually need to be Query or Form. 
        # If I didn't specify, FastAPI might infer Query.
        # Let's try Query param first.
        resp = requests.post(f"{BASE_URL}/ingest", files=files, params={"session_id": session_id})
        
    print(f"Upload Status: {resp.status_code}")
    print(resp.json())
    
    # 4. Query Session
    print("4. Querying Session...")
    query_payload = {
        "question": "Who is the CEO of Apple?",
        "session_id": session_id,
        "memory_mode": "session"
    }
    resp = requests.post(f"{BASE_URL}/query", json=query_payload)
    print(f"Query Result: {resp.json()}")
    
    # 5. Check History
    print("5. Checking History...")
    resp = requests.get(f"{BASE_URL}/sessions/{session_id}/history")
    history = resp.json()["history"]
    print(f"History length: {len(history)}")
    print(history)
    
    # Clean up
    os.remove("test_doc.txt")

if __name__ == "__main__":
    test_session_flow()
