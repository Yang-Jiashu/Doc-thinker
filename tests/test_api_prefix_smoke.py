import os
import time
import requests


BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
API_PREFIX = os.getenv("API_PREFIX", "/api/v1")


def main():
    print("Smoke test for API_PREFIX routes...")

    r = requests.post(f"{BASE_URL}{API_PREFIX}/health")
    print("health:", r.status_code, r.text[:200])

    r = requests.post(f"{BASE_URL}{API_PREFIX}/sessions", json={"title": "Smoke"})
    print("sessions create:", r.status_code, r.text[:200])
    sid = r.json().get("session", {}).get("id")
    if not sid:
        raise RuntimeError("failed to create session")

    r = requests.post(
        f"{BASE_URL}{API_PREFIX}/ingest/stream",
        json={"content": "Alpha is related to Beta.", "source_type": "smoke", "session_id": sid},
        timeout=60,
    )
    print("ingest stream:", r.status_code, r.text[:200])
    time.sleep(2)

    r = requests.post(
        f"{BASE_URL}{API_PREFIX}/query",
        json={"question": "What is Alpha related to?", "session_id": sid, "memory_mode": "session"},
        timeout=120,
    )
    print("query:", r.status_code, r.text[:200])


if __name__ == "__main__":
    main()

