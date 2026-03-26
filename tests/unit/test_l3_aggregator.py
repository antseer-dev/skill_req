import pytest
import numpy as np
from unittest.mock import AsyncMock
from datetime import datetime

from stages.l3_aggregator import L3Aggregator, _calc_trend, _build_personas, _build_emotion_profile
from models import NeedRecord


def make_record(user_type="trader", emotion="frustrated", intensity=0.8, root_cause="high_cost", idx=0) -> NeedRecord:
    return NeedRecord(
        post_id=str(idx), entity="ETH", feature="gas", opinion="too high",
        need_type="explicit", pain_point="gas too high", root_cause=root_cause,
        sentiment="negative", emotion=emotion, intensity=intensity,
        evidence="gas is insane", user_type=user_type, experience_level="intermediate",
    )


def test_calc_trend_rising():
    ts = [{"sentiment": 40}, {"sentiment": 45}, {"sentiment": 50}, {"sentiment": 60}]
    assert _calc_trend(ts) == "rising"


def test_calc_trend_cooling():
    ts = [{"sentiment": 60}, {"sentiment": 55}, {"sentiment": 50}, {"sentiment": 40}]
    assert _calc_trend(ts) == "cooling"


def test_calc_trend_stable():
    ts = [{"sentiment": 50}, {"sentiment": 52}, {"sentiment": 49}, {"sentiment": 51}]
    assert _calc_trend(ts) == "stable"


def test_calc_trend_insufficient_data():
    assert _calc_trend([{"sentiment": 50}]) == "stable"
    assert _calc_trend([]) == "stable"


def test_build_personas_dominant():
    records = (
        [make_record(user_type="trader", idx=i) for i in range(6)] +
        [make_record(user_type="developer", idx=i + 6) for i in range(2)] +
        [make_record(user_type="casual", idx=i + 8) for i in range(2)]
    )
    personas = _build_personas(records)
    names = [p.segment_name for p in personas]
    assert "trader" in names
    # developer and casual each 20% — borderline, depends on rounding
    trader = next(p for p in personas if p.segment_name == "trader")
    assert trader.pct == 60


def test_build_personas_empty():
    assert _build_personas([]) == []


def test_build_emotion_profile():
    records = [
        make_record(emotion="frustrated", intensity=0.9),
        make_record(emotion="frustrated", intensity=0.8),
        make_record(emotion="confused", intensity=0.5),
        make_record(emotion="confused", intensity=0.4),
    ]
    profile = _build_emotion_profile(records, "rising")
    assert profile.dominant_emotion == "frustrated"
    assert profile.trend == "rising"
    assert 0.0 < profile.avg_intensity <= 1.0
    assert "frustrated" in profile.emotion_distribution


@pytest.mark.asyncio
async def test_aggregate_empty_records():
    aggregator = L3Aggregator(client=AsyncMock())
    result = await aggregator.aggregate([], {})
    assert result == []


@pytest.mark.asyncio
async def test_aggregate_too_few_records():
    from unittest.mock import patch
    mock_client = AsyncMock()
    mock_client.embed.return_value = np.random.rand(3, 8).astype(np.float32)
    mock_client.chat_json.return_value = {
        "name": "gas issues", "description": "high gas fees",
        "solutions": [],
    }
    records = [make_record(idx=i) for i in range(3)]
    with patch("pathlib.Path.read_text", return_value="{keywords} {representative_posts}"):
        result = await L3Aggregator(client=mock_client).aggregate(records, {})
    assert isinstance(result, list)
