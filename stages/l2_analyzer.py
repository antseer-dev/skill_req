import asyncio
import logging
from pathlib import Path
from clients.openrouter import OpenRouterClient
from models import Post, NeedRecord
from stages.prompt_utils import render
import config

logger = logging.getLogger(__name__)


_PROMPT_TMPL = Path("prompts/l2_need_extract.txt").read_text()

_VALID_USER_TYPES = {"trader", "developer", "investor", "researcher", "casual"}
_VALID_EXP_LEVELS = {"beginner", "intermediate", "expert"}
_VALID_NEED_TYPES = {"explicit", "implicit"}
_VALID_SENTIMENTS = {"positive", "negative", "neutral"}


def _parse_record(post: Post, raw: dict) -> NeedRecord | None:
    try:
        return NeedRecord(
            post_id=post.post_id,
            entity=raw.get("entity") or "unknown",
            feature=raw.get("feature") or "unknown",
            opinion=raw.get("opinion") or "",
            need_type=raw.get("need_type", "implicit") if raw.get("need_type") in _VALID_NEED_TYPES else "implicit",
            pain_point=raw.get("pain_point"),
            root_cause=raw.get("root_cause"),
            sentiment=raw.get("sentiment", "neutral") if raw.get("sentiment") in _VALID_SENTIMENTS else "neutral",
            emotion=raw.get("emotion") or "neutral",
            intensity=max(0.0, min(1.0, float(raw.get("intensity", 0.5)))),
            evidence=raw.get("evidence") or post.text[:60],
            user_type=raw.get("user_type", "casual") if raw.get("user_type") in _VALID_USER_TYPES else "casual",
            experience_level=raw.get("experience_level", "intermediate") if raw.get("experience_level") in _VALID_EXP_LEVELS else "intermediate",
            source_topic=post.topic,
            topic_label=post.topic_label,
            topic_keywords=post.topic_keywords,
        )
    except Exception:
        return None


class L2Analyzer:
    def __init__(self, client: OpenRouterClient | None = None):
        self._client = client or OpenRouterClient()

    async def analyze(self, posts: list[Post]) -> list[NeedRecord]:
        sem = asyncio.Semaphore(config.GEMINI_CONCURRENCY)
        logger.info("[L2] analyzing %d posts (concurrency=%d)...", len(posts), config.GEMINI_CONCURRENCY)
        done = [0]
        skipped_nonfinancial = [0]
        errors = [0]

        async def analyze_one(post: Post) -> NeedRecord | None:
            prompt = render(_PROMPT_TMPL, text=post.text)
            async with sem:
                try:
                    raw = await self._client.chat_json(prompt)
                    done[0] += 1
                    if done[0] % 20 == 0 or done[0] == len(posts):
                        logger.info("[L2]   %d/%d done (skipped_non_financial=%d, errors=%d)",
                                    done[0], len(posts), skipped_nonfinancial[0], errors[0])
                    if raw.get("is_financial") is False:
                        skipped_nonfinancial[0] += 1
                        return None
                    return _parse_record(post, raw)
                except Exception as e:
                    errors[0] += 1
                    done[0] += 1
                    return None

        results = await asyncio.gather(*[analyze_one(p) for p in posts])
        records = [r for r in results if r is not None]
        logger.info("[L2] done — %d financial records, %d non-financial, %d errors",
                    len(records), skipped_nonfinancial[0], errors[0])
        return records
