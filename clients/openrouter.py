import asyncio
import json
import logging
import re
import httpx
import numpy as np

import config

logger = logging.getLogger(__name__)


def _robust_json_loads(s: str) -> dict:
    """Parse JSON from LLM output, handling common model quirks."""
    # Strip markdown code fences
    s = s.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    # Fix common LLM JSON issues upfront:
    # 1. Invalid \' escape (valid in JS but not JSON)
    s = re.sub(r"\\'", "'", s)
    # 2. Invalid \" that's not inside a string boundary (rare but happens)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # Allow unescaped control characters (newlines inside strings)
    try:
        return json.loads(s, strict=False)
    except json.JSONDecodeError:
        pass
    # Last resort: extract first complete JSON object/array with regex
    m = re.search(r"(\{.*\}|\[.*\])", s, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1), strict=False)
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Cannot parse JSON from LLM response: {s[:300]!r}")


_RETRYABLE = (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError, KeyError)

_TIMEOUT = httpx.Timeout(connect=10, read=120, write=30, pool=30)
_POOL_LIMITS = httpx.Limits(
    max_connections=config.GEMINI_CONCURRENCY + 10,
    max_keepalive_connections=config.GEMINI_CONCURRENCY,
)


class OpenRouterClient:
    def __init__(self, api_key: str = config.OPENROUTER_KEY):
        self._key = api_key
        self._headers = {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=_TIMEOUT,
                limits=_POOL_LIMITS,
                headers=self._headers,
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def chat(
        self,
        prompt: str,
        system: str = "",
        model: str = config.MODEL_BULK,
        json_mode: bool = False,
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        body: dict = {"model": model, "messages": messages}
        if json_mode:
            body["response_format"] = {"type": "json_object"}

        client = self._get_client()
        for attempt in range(3):
            try:
                resp = await client.post(
                    f"{config.OPENROUTER_BASE}/chat/completions",
                    json=body,
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except _RETRYABLE as e:
                if attempt == 2:
                    raise
                logger.warning("[chat] attempt %d failed (%s), retrying...", attempt + 1, e)
                await asyncio.sleep(2 ** attempt)
        return ""

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Batch embed texts using qwen3-embedding-8b."""
        all_embeddings: list[list[float]] = []
        batch_size = config.EMBED_BATCH_SIZE
        n_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info("[embed] %d text(s) → %d batch(es) (model=%s)", len(texts), n_batches, config.MODEL_EMBED)

        client = self._get_client()
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_num = i // batch_size + 1
            for attempt in range(3):
                try:
                    resp = await client.post(
                        f"{config.OPENROUTER_BASE}/embeddings",
                        json={"model": config.MODEL_EMBED, "input": batch},
                    )
                    resp.raise_for_status()
                    data = resp.json()["data"]
                    all_embeddings.extend(item["embedding"] for item in data)
                    logger.info("[embed]   batch %d/%d done (%d texts)", batch_num, n_batches, len(batch))
                    break
                except _RETRYABLE as e:
                    if attempt == 2:
                        raise
                    logger.warning("[embed]   batch %d attempt %d failed (%s), retrying...", batch_num, attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)

        arr = np.array(all_embeddings, dtype=np.float32)
        logger.info("[embed] done — shape=%s", arr.shape)
        return arr

    async def chat_json(self, prompt: str, system: str = "", model: str = config.MODEL_BULK) -> dict:
        raw = await self.chat(prompt, system=system, model=model, json_mode=True)
        return _robust_json_loads(raw)
