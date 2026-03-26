import pytest
from unittest.mock import AsyncMock
from datetime import datetime

from stages.l2_analyzer import L2Analyzer, _parse_record
from models import Post


def make_post(text="gas fees are way too high on ETH", post_id="p1") -> Post:
    return Post(
        post_id=post_id, text=text, creator_id="u1", network="twitter",
        interactions=50, sentiment_lc=30.0,
        timestamp=datetime(2024, 1, 1), topic="ethereum",
    )


VALID_RAW = {
    "is_financial": True,
    "entity": "Ethereum",
    "feature": "gas fees",
    "opinion": "too expensive",
    "need_type": "explicit",
    "pain_point": "gas fees are too high",
    "root_cause": "high_cost",
    "sentiment": "negative",
    "emotion": "frustrated",
    "intensity": 0.85,
    "evidence": "gas fees are way too high",
    "user_type": "trader",
    "experience_level": "intermediate",
}


def test_parse_record_valid():
    post = make_post()
    record = _parse_record(post, VALID_RAW)
    assert record is not None
    assert record.entity == "Ethereum"
    assert record.intensity == 0.85
    assert record.user_type == "trader"


def test_parse_record_invalid_user_type_falls_back():
    raw = {**VALID_RAW, "user_type": "alien"}
    record = _parse_record(make_post(), raw)
    assert record is not None
    assert record.user_type == "casual"


def test_parse_record_invalid_sentiment_falls_back():
    raw = {**VALID_RAW, "sentiment": "mixed"}
    record = _parse_record(make_post(), raw)
    assert record.sentiment == "neutral"


def test_parse_record_intensity_clamped():
    raw = {**VALID_RAW, "intensity": 1.5}
    record = _parse_record(make_post(), raw)
    assert record.intensity == 1.0


@pytest.mark.asyncio
async def test_analyze_returns_records():
    from unittest.mock import patch
    mock_client = AsyncMock()
    mock_client.chat_json.return_value = VALID_RAW

    posts = [make_post(post_id=str(i)) for i in range(5)]
    with patch("stages.l2_analyzer._PROMPT_TMPL", "Analyze: {text}"):
        records = await L2Analyzer(client=mock_client).analyze(posts)
    assert len(records) == 5
    assert all(r.entity == "Ethereum" for r in records)


@pytest.mark.asyncio
async def test_analyze_skips_failed_posts():
    from unittest.mock import patch
    mock_client = AsyncMock()
    mock_client.chat_json.side_effect = [
        VALID_RAW,
        Exception("API error"),
        VALID_RAW,
    ]
    posts = [make_post(post_id=str(i)) for i in range(3)]
    with patch("stages.l2_analyzer._PROMPT_TMPL", "Analyze: {text}"):
        records = await L2Analyzer(client=mock_client).analyze(posts)
    assert len(records) == 2


@pytest.mark.asyncio
async def test_analyze_filters_non_financial_posts():
    """Posts with is_financial=False must be excluded from results."""
    from unittest.mock import patch
    mock_client = AsyncMock()
    mock_client.chat_json.side_effect = [
        VALID_RAW,                                # financial → included
        {**VALID_RAW, "is_financial": False},     # non-financial → excluded
        VALID_RAW,                                # financial → included
    ]
    posts = [make_post(post_id=str(i)) for i in range(3)]
    with patch("stages.l2_analyzer._PROMPT_TMPL", "Analyze: {text}"):
        records = await L2Analyzer(client=mock_client).analyze(posts)
    assert len(records) == 2
