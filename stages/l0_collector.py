import logging
from clients.task_client import TaskClient
from models import Post

logger = logging.getLogger(__name__)


class L0Collector:
    def __init__(self, client: TaskClient | None = None, task_id: str | None = None, topic: str | None = None):
        self._client = client or TaskClient(task_id=task_id)
        self._topic = topic

    async def collect(self) -> tuple[list[Post], dict[str, dict]]:
        """
        Fetch posts and topic states from the task API.

        Returns:
            posts: deduplicated list of Post objects
            states: dict of topic → state dict (contains sentiment, interactions, trend signals)
        """
        posts, states = await self._client.get_data()
        return self._dedup(posts), states

    async def collect_topic(self, topic: str) -> tuple[list[Post], dict[str, dict]]:
        """
        Fetch posts for a single topic from the mock topic API.

        Returns:
            posts: deduplicated list of Post objects
            states: dict of topic → state dict
        """
        posts, states = await self._client.get_data_by_topic(topic)
        return self._dedup(posts), states

    async def collect_topic_file(self, topic: str) -> tuple[list[Post], dict[str, dict]]:
        """
        Load posts for a single topic from a local JSON file (topic_{topic}.json).

        Returns:
            posts: deduplicated list of Post objects
            states: dict of topic → state dict
        """
        posts, states = await self._client.get_data_by_topic_file(topic)
        return self._dedup(posts), states

    def _dedup(self, posts: list[Post]) -> list[Post]:
        # Deduplicate by post_id
        seen: set[str] = set()
        unique_posts: list[Post] = []
        for p in posts:
            if p.post_id not in seen:
                seen.add(p.post_id)
                unique_posts.append(p)

        dupes = len(posts) - len(unique_posts)
        if dupes:
            logger.info("[L0] deduplicated %d duplicate post(s), %d unique", dupes, len(unique_posts))
        return unique_posts
