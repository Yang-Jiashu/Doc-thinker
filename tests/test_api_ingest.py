import requests
import os

url = "http://127.0.0.1:8000/ingest"
file_path = "test_ingest.txt"

with open(file_path, "rb") as f:
    files = [("files", (os.path.basename(file_path), f, "text/plain"))]
    response = requests.post(url, files=files)

print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
