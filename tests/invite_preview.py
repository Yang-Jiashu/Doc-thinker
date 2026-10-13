"""Loopback-only, temporary invitation UI fixture. Never a real AI service.

Run from the repository: python -m tests.invite_preview
The fixed password is for disposable, artificial preview data only.
"""
from pathlib import Path
import json
import tempfile
import uuid

import httpx
import uvicorn

from docthinker.server.invite_app import create_invite_app
from docthinker.server.invite_site import create_invite_site
from docthinker.server.invite_store import InviteStore


def main():
    origin = "http://127.0.0.1:5186"
    with tempfile.TemporaryDirectory(prefix="docthinker-invite-ui-") as temporary:
        store = InviteStore.initialize(Path(temporary).resolve() / "registry")
        invitation = store.add_user("preview", 18101)
        store.register(invitation["invite_code"], "preview", "OfflinePreviewOnly2026")
        sessions, history, receipts = [], [], {}

        def fixture(request):
            path = request.url.path.rsplit("/", 1)[-1]
            if path == "sessions":
                if request.method == "POST":
                    body = json.loads(request.content)
                    session = {"id": "#00001", "title": body.get("title", "本机模拟会话")}
                    sessions[:] = [session]
                    return httpx.Response(200, json={"session": session})
                return httpx.Response(200, json={"sessions": sessions})
            if path == "history":
                pending = [{"request_id": key, "request_status": "running"}
                           for key, value in receipts.items() if not value["completed"]]
                return httpx.Response(200, json={"history": history, "query_requests": pending[-5:]})
            if path == "status":
                request_id = request.url.params.get("request_id")
                receipt = receipts.get(request_id)
                if receipt is None:
                    return httpx.Response(200, json={"request_id": request_id, "request_status": "not_found"})
                if not receipt["completed"]:
                    receipt["completed"] = True
                    history.extend([{"role": "user", "content": receipt["question"]},
                                    {"role": "assistant", "content": receipt["result"]["answer"],
                                     "answer_id": receipt["result"]["answer_id"]}])
                return httpx.Response(200, json=receipt["result"])
            if path == "query":
                body = json.loads(request.content)
                request_id = body["request_id"]
                if request_id in receipts:
                    receipt = receipts[request_id]
                    if receipt["completed"]:
                        return httpx.Response(200, json=receipt["result"])
                else:
                    answer = "这是本机界面联调的模拟回复，没有调用真实模型。已验证提问回执、手动查询结果和回答复制界面；这不是模型质量测试或真实任务执行。"
                    receipts[request_id] = {"question": body["question"], "completed": False,
                                           "result": {"answer": answer, "sources": [],
                                                      "answer_id": str(uuid.uuid4()),
                                                      "request_id": request_id,
                                                      "request_status": "completed",
                                                      "context_windowed": True}}
                # Deliberately simulate a short HTTP wait expiring. Only an explicit
                # status lookup completes this fixture; no provider is ever invoked.
                return httpx.Response(202, json={"request_id": request_id, "request_status": "running"})
            return httpx.Response(503, json={"detail": "Offline preview only"})

        api = create_invite_app(store=store, origin=origin, transport=httpx.MockTransport(fixture))
        root = Path(__file__).resolve().parents[1]
        app = create_invite_site(api_app=api, web_dir=root / "clients/mobile/dist/build/h5")
        print("Local disposable UI fixture only: " + origin, flush=True)
        print("Test username preview; password OfflinePreviewOnly2026; no real model or private data.", flush=True)
        uvicorn.run(app, host="127.0.0.1", port=5186, access_log=False, log_level="warning", proxy_headers=False)


if __name__ == "__main__":
    main()
