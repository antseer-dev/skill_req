import pytest
from unittest.mock import AsyncMock, patch

from stages.l4_spec_generator import L4SpecGenerator
from models import (
    PainCluster, UserPersona, EmotionProfile,
    ExistingSolution, OpportunityScore,
)

PROMPT_STUB = (
    "{cluster_name} {description} {keywords} {top_scenarios} {trend} "
    "{evidence_samples} {personas} {existing_solutions} {dominant_emotion} {avg_intensity}"
)


def make_cluster(cluster_id=1, trend="rising") -> PainCluster:
    return PainCluster(
        cluster_id=cluster_id,
        cluster_name="DeFi yield comparison",
        description="Users need to compare DeFi yields across protocols",
        post_count=42,
        top_scenarios=["missing_api", "high_cost"],
        trend=trend,
        evidence_samples=['"i spend hours comparing yields"', '"no good aggregator exists"'],
        keywords=["yield", "apy", "defi", "protocol"],
        personas=[UserPersona(segment_name="trader", post_count=30, pct=70, experience_level="intermediate")],
        emotion_profile=EmotionProfile(
            dominant_emotion="frustrated",
            emotion_distribution={"frustrated": 0.7, "confused": 0.3},
            avg_intensity=0.8,
            trend=trend,
        ),
        existing_solutions=[
            ExistingSolution(name="DeFiLlama", coverage="partial", limitation="no personalized recommendations")
        ],
        opportunity=OpportunityScore(
            importance=7.5, satisfaction=3.0, raw_score=12.0,
            intensity_weight=1.5, solution_weight=2.0, trend_boost=1.3,
            final_score=46.8,
        ),
    )


MOCK_LLM_RESPONSE = {
    "skill_name": "defi-yield-scout",
    "trigger_description": "Find and compare DeFi yield opportunities...",
    "expected_output_format": "Comparison table with APY, risk, and recommendation",
    "example_prompts": [
        "find best yield for my 10k USDC with low risk",
        "compare aave vs compound APY",
        "where should i put my stable coins for max yield",
    ],
    "suggested_approach": "Query DeFiLlama API → rank by risk-adjusted APY",
    "need_name": "DeFi Yield Aggregation",
    "need_type": "workflow_friction",
    "need_description": "Users spend too much time manually comparing yields across protocols",
    "importance": 8.0,
    "satisfaction": 3.0,
    "has_mature_solution": False,
}


@pytest.mark.asyncio
async def test_generate_returns_skill_specs():
    mock_client = AsyncMock()
    mock_client.chat_json.return_value = MOCK_LLM_RESPONSE

    with patch("pathlib.Path.read_text", return_value=PROMPT_STUB):
        specs = await L4SpecGenerator(client=mock_client).generate([make_cluster()])

    assert len(specs) == 1
    assert specs[0].skill_name == "defi-yield-scout"
    assert len(specs[0].example_prompts) == 3


@pytest.mark.asyncio
async def test_generate_respects_top_n(monkeypatch):
    monkeypatch.setattr("config.TOP_N_SKILLS", 3)
    mock_client = AsyncMock()
    mock_client.chat_json.return_value = MOCK_LLM_RESPONSE

    with patch("pathlib.Path.read_text", return_value=PROMPT_STUB):
        specs = await L4SpecGenerator(client=mock_client).generate(
            [make_cluster(cluster_id=i) for i in range(10)]
        )

    assert len(specs) == 3


@pytest.mark.asyncio
async def test_opportunity_score_refined_by_llm():
    mock_client = AsyncMock()
    mock_client.chat_json.return_value = {
        **MOCK_LLM_RESPONSE,
        "importance": 9.0,
        "satisfaction": 2.0,
        "has_mature_solution": False,
    }

    with patch("pathlib.Path.read_text", return_value=PROMPT_STUB):
        specs = await L4SpecGenerator(client=mock_client).generate([make_cluster()])

    opp = specs[0].opportunity
    assert opp.importance == 9.0
    assert opp.satisfaction == 2.0
    # raw = 9 + max(9-2, 0) = 16; ×1.5 (intensity) ×2.0 (no solution) ×1.3 (rising) = 62.4
    assert opp.final_score == pytest.approx(62.4)
