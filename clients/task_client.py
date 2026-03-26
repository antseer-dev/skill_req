"""
Client for the mock task API.
Replaces LunarCrushClient as the primary data source.

API: GET /api/v1/mock/tasks
Returns: { data: { task_id, topics: [{ topic, state, posts }] } }
"""
import logging
import httpx
from datetime import datetime

from models import Post

logger = logging.getLogger(__name__)


TASK_API_URL = "http://54.169.201.50:8575/api/v1/mock/tasks"


class TaskClient:
    def __init__(self, url: str = TASK_API_URL, timeout: int = 30):
        self._url = url
        self._timeout = timeout

    async def get_data(self) -> tuple[list[Post], dict[str, dict]]:
        """
        Fetch all posts and topic states from the task API.

        Returns:
            posts: flat list of Post objects across all topics
            states: dict of topic → state dict (used for trend/timeseries)
        """
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(self._url)
            resp.raise_for_status()

        payload = resp.json()
        topics_data = payload["data"]["topics"]
        logger.info("[task_client] API returned %d topic(s)", len(topics_data))

        all_posts: list[Post] = []
        states: dict[str, dict] = {}

        for topic_entry in topics_data:
            topic = topic_entry["topic"]
            states[topic] = topic_entry.get("state", {})
            raw_posts = topic_entry.get("posts", [])
            parsed = _parse_posts(raw_posts, topic)
            logger.info("[task_client]   topic=%r: %d raw → %d kept (latin+length filter)",
                        topic, len(raw_posts), len(parsed))
            all_posts.extend(parsed)

        logger.info("[task_client] total posts across all topics: %d", len(all_posts))
        return all_posts, states


def _parse_posts(raw_posts: list[dict], topic: str) -> list[Post]:
    posts = []
    for item in raw_posts:
        try:
            # Use post_title + post_description (some topics have both)
            title = item.get("post_title") or ""
            desc  = item.get("post_description") or ""
            text  = (title + (" " + desc if desc else "")).strip()

            # post_type: "tweet" → "twitter", "reddit-post" → "reddit", etc.
            post_type = item.get("post_type", "unknown")
            network = post_type.replace("-post", "").replace("-", "_")

            # post_created is unix timestamp
            ts_raw = item.get("post_created")
            timestamp = datetime.fromtimestamp(ts_raw) if ts_raw else datetime(2024, 1, 1)

            # post_sentiment is 0-5 scale; normalise to 0-100 for consistency
            sentiment_raw = float(item.get("post_sentiment") or 2.5)
            sentiment_lc  = sentiment_raw * 20.0  # 0-5 → 0-100

            posts.append(Post(
                post_id=str(item.get("id", "")),
                text=text,
                creator_id=str(item.get("creator_id", "")),
                network=network,
                creator_followers=item.get("creator_followers"),
                interactions=item.get("interactions_24h") or item.get("interactions_total") or 0,
                sentiment_lc=sentiment_lc,
                timestamp=timestamp,
                topic=topic,
            ))
        except Exception:
            continue

    # Filter: meaningful text + Latin characters
    return [p for p in posts if len(p.text.strip()) > 20 and _is_latin(p.text)]


def _is_latin(text: str) -> bool:
    """Return True if ≥60% of alphabetic chars are ASCII."""
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return False
    return sum(1 for c in alpha if ord(c) < 128) / len(alpha) >= 0.6
