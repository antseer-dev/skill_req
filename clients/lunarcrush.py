import asyncio
from datetime import datetime
from typing import Optional
import httpx

import config
from models import Post
from storage.cache import Cache


class LunarCrushClient:
    def __init__(self, api_key: str = config.LUNARCRUSH_KEY, cache: Optional[Cache] = None):
        self._key = api_key
        self._cache = cache or Cache()
        self._sem = asyncio.Semaphore(config.LUNARCRUSH_CONCURRENCY)
        self._headers = {"Authorization": f"Bearer {self._key}"}

    async def _get(self, path: str, params: dict | None = None) -> dict:
        url = f"{config.LUNARCRUSH_BASE}{path}"
        async with self._sem:
            async with httpx.AsyncClient(timeout=30) as client:
                for attempt in range(3):
                    try:
                        resp = await client.get(url, headers=self._headers, params=params or {})
                        if resp.status_code == 429:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        resp.raise_for_status()
                        return resp.json()
                    except httpx.HTTPStatusError as e:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 ** attempt)
        return {}

    async def get_topic_posts(self, topic: str) -> list[Post]:
        cached = self._cache.get("posts", topic)
        if cached:
            return [Post(**p) for p in cached]

        data = await self._get(f"/api4/public/topic/{topic}/posts/v1")
        posts = self._parse_posts(data, topic)

        self._cache.set("posts", topic, value=[p.model_dump(mode="json") for p in posts])
        return posts

    async def get_topic_timeseries(self, topic: str) -> list[dict]:
        cached = self._cache.get("timeseries", topic)
        if cached:
            return cached

        data = await self._get(
            f"/api4/public/topic/{topic}/time-series/v1",
            params={"bucket": "days", "interval": "2w"},
        )
        series = data.get("data", [])
        self._cache.set("timeseries", topic, value=series)
        return series

    def _parse_posts(self, data: dict, topic: str) -> list[Post]:
        posts = []
        for item in data.get("data", []):
            try:
                # LunarCrush API v4 uses post_title/post_created/post_sentiment
                text = (
                    item.get("body") or item.get("text") or
                    item.get("post_title") or item.get("title") or ""
                )
                # post_type is e.g. "twitter-post", "instagram-post"
                network = (
                    item.get("network") or
                    item.get("post_type", "unknown").split("-")[0]
                )
                # timestamp field varies by endpoint
                ts_raw = (
                    item.get("created_at") or item.get("post_created") or
                    item.get("time") or "2024-01-01T00:00:00"
                )
                # post_created is a unix timestamp (int), others are ISO strings
                if isinstance(ts_raw, (int, float)):
                    timestamp = datetime.fromtimestamp(ts_raw)
                else:
                    timestamp = datetime.fromisoformat(str(ts_raw))

                sentiment_raw = item.get("post_sentiment") or item.get("sentiment") or 50.0

                posts.append(Post(
                    post_id=str(item.get("id", "")),
                    text=text,
                    creator_id=str(item.get("creator_id") or item.get("user_id", "")),
                    network=network,
                    creator_followers=item.get("creator_followers"),
                    interactions=item.get("interactions_24h") or item.get("interactions") or 0,
                    sentiment_lc=float(sentiment_raw),
                    timestamp=timestamp,
                    topic=topic,
                ))
            except Exception:
                continue
        return [p for p in posts if len(p.text.strip()) > 20 and _is_latin(p.text)]


def _is_latin(text: str) -> bool:
    """Return True if ≥60% of alphabetic chars are ASCII (filters CJK/Arabic/Cyrillic posts)."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    ascii_alpha = sum(1 for c in alpha if ord(c) < 128)
    return ascii_alpha / len(alpha) >= 0.6
