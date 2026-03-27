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


TASK_API_URL = "http://54.169.201.50:8575/api/v1/tasks"
MOCK_API_URL = "http://54.169.201.50:8575/api/v1/mock/tasks"
MOCK_TOPIC_API_URL = "http://54.169.201.50:8575/api/v1/mock/topic/{topic}"


class TaskClient:
    def __init__(self, url: str = MOCK_API_URL, timeout: int = 120, task_id: str | None = None):
        self._url = url
        self._timeout = timeout
        self._task_id = task_id

    async def get_data(self) -> tuple[list[Post], dict[str, dict]]:
        """
        Fetch all posts and topic states from the task API.

        Returns:
            posts: flat list of Post objects across all topics
            states: dict of topic → state dict (used for trend/timeseries)
        """
        data = None

        # Use main API when task_id is provided; use mock API otherwise
        if self._task_id:
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.get(TASK_API_URL, params={"task_id": self._task_id})
                    if resp.status_code != 200:
                        payload = resp.json()
                        raise ValueError(payload.get("message", f"HTTP {resp.status_code}"))
                    data = resp.json()
                    topics_data = data.get("data", {}).get("topics", [])
                    if topics_data:
                        logger.info("[task_client] main API returned %d topic(s)", len(topics_data))
                    else:
                        logger.info("[task_client] main API returned empty topics for task_id=%s", self._task_id)
            except ValueError:
                raise
            except Exception as e:
                logger.error("[task_client] main API failed for task_id=%s (%s)", self._task_id, e)
                raise
        else:
            logger.info("[task_client] fetching from mock API (no task_id)")
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(MOCK_API_URL)
                resp.raise_for_status()
            data = resp.json()

        topics_data = data["data"]["topics"]
        self._task_id = data["data"].get("task_id") or self._task_id
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

    async def get_data_by_topic(self, topic: str) -> tuple[list[Post], dict[str, dict]]:
        """
        Fetch posts for a single topic from the mock topic API.

        Args:
            topic: The topic name to fetch (e.g. "rate_cuts").

        Returns:
            posts: flat list of Post objects for the topic
            states: dict of topic → state dict
        """
        url = MOCK_TOPIC_API_URL.format(topic=topic)
        logger.info("[task_client] fetching topic %r from %s", topic, url)
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
        data = resp.json()

        entry = data["data"]  # { task_id, topic, state, posts }
        self._task_id = entry.get("task_id") or self._task_id

        posts = _parse_posts(entry.get("posts", []), topic)
        logger.info("[task_client] topic=%r: %d raw → %d kept", topic, len(entry.get("posts", [])), len(posts))

        # states dict: single key → single value
        states: dict[str, dict] = {topic: entry.get("state", {})}
        return posts, states

    async def get_data_by_topic_file(self, topic: str) -> tuple[list[Post], dict[str, dict]]:
        """
        Load posts for a single topic from a local JSON file.

        Args:
            topic: The topic name (e.g. "oil"). Will load topic_{topic}.json.

        Returns:
            posts: flat list of Post objects for the topic
            states: dict of topic → state dict
        """
        filename = f"topic_{topic}.json"
        logger.info("[task_client] loading topic %r from local file: %s", topic, filename)

        import json as _json
        with open(filename, encoding="utf-8") as f:
            data = _json.load(f)

        # Support both single-topic file and multi-topic file (same format as API)
        if "topics" in data.get("data", {}):
            # Multi-topic file: { data: { task_id, topics: [...] } }
            topics_data = data["data"]["topics"]
            self._task_id = data["data"].get("task_id") or self._task_id
            def _norm(s: str) -> str:
                return s.lower().replace("_", " ").replace("-", " ").strip()
            entry = next((t for t in topics_data if _norm(t.get("topic", "")) == _norm(topic)), None)
            if not entry:
                raise ValueError(f"Topic {topic!r} not found in {filename}")
        else:
            # Single-topic file: { data: { task_id, topic, state, posts } }
            entry = data["data"]
            self._task_id = entry.get("task_id") or self._task_id

        posts = _parse_posts(entry.get("posts", []), topic)
        logger.info("[task_client] topic=%r: %d raw → %d kept", topic, len(entry.get("posts", [])), len(posts))

        states: dict[str, dict] = {topic: entry.get("state", {})}
        return posts, states


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
