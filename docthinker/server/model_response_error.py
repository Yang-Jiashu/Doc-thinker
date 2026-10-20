"""Known, terminal text-response failures; never carry provider body content.

These codes mean a provider response arrived without a complete usable answer.
They say nothing about billed tokens. Transport failures/timeouts remain unknown.
"""
from types import MappingProxyType


MODEL_RESPONSE_FAILURES = MappingProxyType({
    "model_response_truncated": "模型达到本次输出上限，未完成回答。没有自动重试；草稿已保留。",
    "model_response_empty": "模型已返回，但没有提供可用的回答正文。没有自动重试；草稿已保留。",
    "model_response_invalid": "模型返回的结果不符合完整文本回答格式。没有自动重试；草稿已保留。",
})


def model_response_failure(code: str) -> dict:
    if not isinstance(code, str) or code not in MODEL_RESPONSE_FAILURES:
        raise ValueError("Invalid model response failure code")
    return {"code": code, "message": MODEL_RESPONSE_FAILURES[code], "automatic_retry": False}


class ModelResponseError(ValueError):
    def __init__(self, code: str):
        model_response_failure(code)
        self.code = code
        super().__init__("Chat model did not return a complete text answer")
