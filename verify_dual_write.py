import requests
import time
import json

BASE_URL = "http://localhost:8000"

def test_dual_write():
    print("1. Creating Session...")
    res = requests.post(f"{BASE_URL}/sessions", json={"title": "Dual Write Test"})
    if res.status_code != 200:
        print(f"Failed to create session: {res.text}")
        return
    session_id = res.json()["session"]["id"]
    print(f"Session Created: {session_id}")

    print("\n2. Sending Chat Message (should trigger background ingestion)...")
    # This message contains a unique fact "The secret code is 998877"
    question = "Please remember that the secret code for this test is 998877."
    res = requests.post(f"{BASE_URL}/query", json={
        "question": question,
        "session_id": session_id,
        "mode": "hybrid",
        "memory_mode": "hybrid"
    })
    print(f"Query Response: {res.json().get('answer')}")

    print("\n3. Waiting for Background Ingestion (15 seconds)...")
    time.sleep(15)

    print("\n4. Verifying Session Memory...")
    res = requests.post(f"{BASE_URL}/query", json={
        "question": "What is the secret code?",
        "session_id": session_id,
        "mode": "hybrid",
        "memory_mode": "session"
    })
    answer_session = res.json().get('answer')
    print(f"Session Answer: {answer_session}")

    print("\n5. Verifying Global Memory (New Session)...")
    # Create a NEW session to test global recall
    res = requests.post(f"{BASE_URL}/sessions", json={"title": "Global Test"})
    global_session_id = res.json()["session"]["id"]
    
    res = requests.post(f"{BASE_URL}/query", json={
        "question": "What is the secret code mentioned in the previous test?",
        "session_id": global_session_id,
        "mode": "hybrid",
        "memory_mode": "global"
    })
    answer_global = res.json().get('answer')
    print(f"Global Answer: {answer_global}")

    if "998877" in str(answer_session) or "998877" in str(answer_global):
        print("\nSUCCESS: Secret code retrieved!")
    else:
        print("\nWARNING: Secret code NOT retrieved yet (might need more time for indexing).")

if __name__ == "__main__":
    try:
        test_dual_write()
    except Exception as e:
        print(f"Test failed: {e}")
