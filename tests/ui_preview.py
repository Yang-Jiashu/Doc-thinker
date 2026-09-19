"""Local UI fixture, no backend/model calls or persistent session writes.

Run: PYTHONPATH=. python tests/ui_preview.py
Only test data is served on http://127.0.0.1:5055/query.
"""

import json

from flask import Response, jsonify, request

from docthinker.ui.app import app

sessions = [{"id": "#preview", "title": "界面验收 · 模拟数据"}]
requests_seen = []


@app.before_request
def preview_api():
    path = request.path
    if path == "/_preview/requests":
        return jsonify(requests_seen)
    if not path.startswith("/api/"):
        return None
    if path == "/api/v1/sessions":
        if request.method == "POST":
            session = {"id": f"#preview-{len(sessions)}", "title": "新对话 · 模拟数据"}
            sessions.append(session)
            return jsonify(session=session)
        return jsonify(sessions=sessions)
    if path.endswith("/history"):
        return jsonify(history=[])
    if path.endswith("/files"):
        return jsonify(files=[])
    if path == "/api/v1/settings" and request.method == "GET":
        return jsonify(
            llm_base_url="https://example.invalid/v1",
            llm_model="preview-model",
            vlm_model="preview-vision",
            keyword_llm_model="preview-model",
            embed_base_url="https://example.invalid/v1",
            embed_model="preview-embedding",
            embed_dim=1024,
            rerank_model="preview-reranker",
            llm_max_async=4,
            embedding_max_async=4,
            max_parallel_insert=2,
            llm_router_max_concurrency=4,
            workdir="模拟环境，不读写实际资料",
        )
    if path == "/api/v1/knowledge-graph/data":
        return jsonify(nodes=[], links=[], metadata={"total_nodes": 0})
    if path == "/api/v1/memory/long-horizon":
        return jsonify(memories=[])
    if path == "/api/v1/query/stream":
        payload = request.get_json()
        requests_seen.append(payload)
        if "模拟失败" in payload.get("question", ""):
            return jsonify(response="模拟服务不可用，请重试。"), 503
        mode = payload.get("evolution_mode", "auto")
        events = [
            {
                "type": "meta",
                "data": {
                    "question_policy": {"mode": mode, "reason": "界面验收模拟结果"},
                    "context_budget": {
                        "retrieval_limits": {
                            "max_total_tokens": 12000,
                            "chunk_top_k": 8,
                            "max_relations": 16,
                        },
                        "history_tokens": 120,
                        "history_limit": 1200,
                        "instruction_tokens": 240,
                        "instruction_limit": 2000,
                        "instruction_truncated": False,
                    },
                    "sources": [
                        {
                            "content": "示例原文：增加国内供给可能降低进口依赖。",
                            "confidence": 1,
                        }
                    ],
                },
            },
            {
                "type": "chunk",
                "content": "这是界面验收的模拟回答，没有调用模型。\n\n**证据与推断分开：**原文支持供给和进口依赖的关系；完整能源安全因果链仍需额外证据。",
            },
        ]
        return Response(
            "".join(
                f"data: {json.dumps(event, ensure_ascii=False)}\n\n" for event in events
            )
            + "data: [DONE]\n\n",
            mimetype="text/event-stream",
        )
    if request.method == "DELETE":
        return jsonify(success=True)
    return jsonify(detail="Not implemented by the preview fixture"), 404


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=False, use_reloader=False)
