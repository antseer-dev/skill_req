"""
E2E test: full pipeline with mocked external APIs.
No real HTTP calls — verifies data flows correctly through all stages.
"""
import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from datetime import datetime

from models import Post, SkillSpec


# ── Fixtures ─────────────────────────────────────────────

MOCK_LC_POSTS_RESPONSE = {
    "data": [
        {
            "id": f"post_{i}",
            "body": text,
            "creator_id": f"creator_{i % 5}",
            "network": "twitter",
            "creator_followers": 1000,
            "interactions_24h": 50 + i * 10,
            "sentiment": 40.0,
            "created_at": "2024-01-15T10:00:00",
        }
        for i, text in enumerate([
            "DeFi yields are impossible to track manually, i have 5 tabs open just to compare APY",
            "Why is there no single tool to compare all DeFi protocols? This is 2024",
            "Spent 3 hours trying to find the best stablecoin yield. There has to be a better way",
            "Gas fees on ETH are ridiculous, I can't make small trades anymore",
            "ETH gas is killing me, layer 2 solutions are too confusing to figure out",
            "I have no idea which L2 to use, they all have different bridges and it's a mess",
            "Whale alert: someone moved 50k BTC, what does this mean for the market?",
            "How do I track whale wallets? I want to know when big players are moving",
            "Smart money is buying ETH again, but I can't find a reliable way to track this",
            "Portfolio tracking across multiple wallets and chains is a nightmare",
            "I have wallets on ETH, SOL, and BSC and no single tool shows my total portfolio",
            "Cross-chain portfolio view would save me so much time every morning",
        ] * 4)   # 48 posts total
    ]
}

MOCK_LC_TIMESERIES = {
    "data": [{"sentiment": 45 + i, "time": f"2024-01-{i+1:02d}"} for i in range(14)]
}

MOCK_EMBEDDINGS = np.random.RandomState(42).rand(48, 16).astype(np.float32)

MOCK_L2_RESPONSE = {
    "is_financial": True,
    "entity": "DeFi protocol",
    "feature": "yield comparison",
    "opinion": "too manual",
    "need_type": "explicit",
    "pain_point": "no unified yield comparison tool",
    "root_cause": "missing_feature",
    "sentiment": "negative",
    "emotion": "frustrated",
    "intensity": 0.8,
    "evidence": "5 tabs open just to compare APY",
    "user_type": "trader",
    "experience_level": "intermediate",
}

MOCK_L3_NAME_RESPONSE = {
    "name": "DeFi Yield Aggregation",
    "description": "Users need a unified tool to compare yields across protocols",
}

MOCK_L3_SOLUTION_RESPONSE = {
    "solutions": [
        {"name": "DeFiLlama", "coverage": "partial", "limitation": "no personalized recommendations"},
    ]
}

MOCK_L4_RESPONSE = {
    "skill_name": "defi-yield-scout",
    "trigger_description": "Find and compare DeFi yields...",
    "expected_output_format": "Protocol comparison table with APY and risk",
    "example_prompts": [
        "find best yield for 10k USDC with low risk",
        "compare aave vs compound APY right now",
        "where should I put stablecoins for max yield",
    ],
    "suggested_approach": "Query DeFiLlama → rank by risk-adjusted APY",
    "need_name": "DeFi Yield Comparison",
    "need_type": "workflow_friction",
    "need_description": "Users manually compare yields across many sites",
    "importance": 8.5,
    "satisfaction": 3.0,
    "has_mature_solution": False,
}


# ── Helpers ──────────────────────────────────────────────

def _mock_openrouter():
    client = AsyncMock()

    call_count = [0]

    async def chat_json_side_effect(prompt, **kwargs):
        call_count[0] += 1
        # L4: Claude Code Skill architect prompt (most specific — check first)
        if "skill architect" in prompt.lower() or "trigger_description" in prompt:
            return MOCK_L4_RESPONSE
        # L2: per-post NLP extraction
        if "entity" in prompt.lower() or "pain_point" in prompt.lower() or "Post:" in prompt:
            return MOCK_L2_RESPONSE
        # L3 cluster name
        if "keywords" in prompt.lower() and "representative" in prompt.lower():
            return MOCK_L3_NAME_RESPONSE
        # L3 existing solutions
        return MOCK_L3_SOLUTION_RESPONSE

    client.chat_json.side_effect = chat_json_side_effect
    client.embed.return_value = MOCK_EMBEDDINGS
    return client


def _mock_task_client():
    client = AsyncMock()
    posts = [
        Post(
            post_id=f"p{i}", text=item["body"], creator_id=item["creator_id"],
            network="twitter", interactions=item["interactions_24h"],
            sentiment_lc=item["sentiment"] * 20.0, timestamp=datetime(2024, 1, 15), topic="bitcoin",
        )
        for i, item in enumerate(MOCK_LC_POSTS_RESPONSE["data"])
    ]
    states = {"bitcoin": {"velocity": 0.1, "sentiment": 55, "sub_label": "bullish"}}
    client.get_data.return_value = (posts, states)
    return client


# ── Tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end():
    """Full pipeline produces SkillSpec list with correct structure."""
    import pipeline

    mock_or = _mock_openrouter()

    with (
        patch("stages.l0_collector.TaskClient", return_value=_mock_task_client()),
        patch("stages.l1_clusterer.OpenRouterClient", return_value=mock_or),
        patch("stages.l2_analyzer.OpenRouterClient", return_value=mock_or),
        patch("stages.l3_aggregator.OpenRouterClient", return_value=mock_or),
        patch("stages.l4_spec_generator.OpenRouterClient", return_value=mock_or),
        patch("stages.l1_clusterer.UMAP") as mock_umap,
        patch("pipeline._write_output"),
    ):
        mock_umap.return_value.fit_transform.return_value = np.random.rand(48, 10)

        specs = await pipeline.run()

    assert isinstance(specs, list)
    assert len(specs) > 0
    assert all(isinstance(s, SkillSpec) for s in specs)


@pytest.mark.asyncio
async def test_pipeline_skill_spec_fields():
    """SkillSpec has all required fields populated."""
    import pipeline

    mock_or = _mock_openrouter()

    with (
        patch("stages.l0_collector.TaskClient", return_value=_mock_task_client()),
        patch("stages.l1_clusterer.OpenRouterClient", return_value=mock_or),
        patch("stages.l2_analyzer.OpenRouterClient", return_value=mock_or),
        patch("stages.l3_aggregator.OpenRouterClient", return_value=mock_or),
        patch("stages.l4_spec_generator.OpenRouterClient", return_value=mock_or),
        patch("stages.l1_clusterer.UMAP") as mock_umap,
        patch("pipeline._write_output"),
    ):
        mock_umap.return_value.fit_transform.return_value = np.random.rand(48, 10)
        specs = await pipeline.run()

    if specs:
        spec = specs[0]
        assert spec.skill_name
        assert spec.trigger_description
        assert len(spec.example_prompts) > 0
        assert spec.opportunity.final_score > 0
        assert spec.post_count > 0


@pytest.mark.asyncio
async def test_pipeline_no_posts_returns_empty():
    """Pipeline returns empty list when task API returns no posts."""
    import pipeline

    mock_task = AsyncMock()
    mock_task.get_data.return_value = ([], {})

    with patch("stages.l0_collector.TaskClient", return_value=mock_task):
        specs = await pipeline.run()

    assert specs == []
