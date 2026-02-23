"""Async client for the BLTCY GPT-4o-mini multimodal API."""
#对接vlm客户端，负责发送图像和处理响应。
from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import asyncio

import aiohttp


class VLMClient:
    """Lightweight wrapper around https://api.bltcy.ai/v1/chat/completions."""

    def __init__(
        self,
        api_key: str,
        *,
        api_base: str = "https://api.siliconflow.cn/v1/chat/completions",
        model: str = "qwen2.5-3b-instruct",
        timeout: float = 240.0,
    ) -> None:
        self.api_key = api_key
        # Normalise API base: allow passing either root or full chat endpoint.
        api_base = api_base.rstrip("/")
        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"
        self.api_base = api_base
        self.model = model
        self.timeout = timeout

    async def generate(
        self,
        prompt: str,
        *,
        images: Optional[Sequence[str]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 5120,
        temperature: float = 0.2,
        extra_messages: Optional[List[dict]] = None,
        extra_body: Optional[dict] = None,
    ) -> str:
        """Generate a response using the multimodal endpoint."""

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        if extra_messages:
            messages: List[dict] = list(extra_messages)
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
        else:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})

            if images:
                content = [{"type": "text", "text": prompt}]
                for img in images or []:
                    content.append(self._encode_image(img))
                messages.append({"role": "user", "content": content})
            else:
                messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if extra_body:
            try:
                payload.update(dict(extra_body))
            except Exception:
                payload.update({"enable_thinking": extra_body.get("enable_thinking", False)})
        # For non-streaming calls on DashScope-compatible endpoints, explicitly disable thinking
        if "enable_thinking" not in payload:
            payload["enable_thinking"] = False

        last_error = None
        for attempt in range(3):
            try:
                timeout_cfg = aiohttp.ClientTimeout(total=self.timeout)
                async with aiohttp.ClientSession(timeout=timeout_cfg) as session:
                    async with session.post(
                        self.api_base,
                        headers=headers,
                        data=json.dumps(payload),
                        timeout=timeout_cfg,
                    ) as response:
                        if response.status >= 400:
                            try:
                                err_text = await response.text()
                            except Exception:
                                err_text = "<no body>"
                            raise aiohttp.ClientResponseError(
                                response.request_info,
                                response.history,
                                status=response.status,
                                message=f"Bad Request: {err_text}",
                                headers=response.headers,
                            )
                        data = await response.json()
                        choices = data.get("choices") or []
                        if not choices:
                            raise RuntimeError(f"Unexpected response payload: {data}")
                        return choices[0]["message"]["content"]
            except asyncio.TimeoutError as e:
                last_error = e
                if attempt < 2:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                raise
            except Exception as e:
                last_error = e
                raise
        raise last_error

    @staticmethod
    def _encode_image(path_like: str) -> dict:
        """Encode image file as the API expects."""
        path = Path(path_like)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path_like}")

        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        mime, _ = mimetypes.guess_type(path_like)
        if not mime:
            mime = "image/png"
        data_uri = f"data:{mime};base64,{encoded}"
        return {"type": "image_url", "image_url": {"url": data_uri}}
