import pytest
from datetime import datetime
from pydantic import ValidationError
from models import Post, NeedRecord, UserPersona, EmotionProfile, OpportunityScore, PainCluster, SkillSpec


def make_post(**kwargs):
    defaults = dict(
        post_id="p1", text="BTC is mooning!", creator_id="u1",
        network="twitter", interactions=100, sentiment_lc=75.0,
        timestamp=datetime(2024, 1, 1), topic="bitcoin",
    )
    return Post(**{**defaults, **kwargs})


def test_post_valid():
    p = make_post()
    assert p.post_id == "p1"
    assert p.cluster_id is None


def test_post_with_cluster():
    p = make_post(cluster_id=3, topic_label="price action", topic_keywords=["btc", "moon"])
    assert p.cluster_id == 3
    assert p.topic_keywords == ["btc", "moon"]


def test_need_record_intensity_clamped():
    r = NeedRecord(
        post_id="p1", entity="ETH", feature="gas fees", opinion="too high",
        need_type="explicit", sentiment="negative", emotion="frustrated",
        intensity=0.9, evidence="gas is insane", user_type="trader",
        experience_level="intermediate",
    )
    assert 0.0 <= r.intensity <= 1.0


def test_need_record_intensity_validation():
    with pytest.raises(ValidationError):
        NeedRecord(
            post_id="p1", entity="ETH", feature="gas", opinion="bad",
            need_type="explicit", sentiment="negative", emotion="anger",
            intensity=1.5,          # out of range
            evidence="...", user_type="trader", experience_level="beginner",
        )


def test_user_persona():
    p = UserPersona(segment_name="trader", post_count=30, pct=60, experience_level="intermediate")
    assert p.pct == 60


def test_opportunity_score_final():
    o = OpportunityScore(
        importance=8.0, satisfaction=3.0, raw_score=13.0,
        intensity_weight=1.5, solution_weight=2.0, trend_boost=1.3,
        final_score=round(13.0 * 1.5 * 2.0 * 1.3, 2),
    )
    assert o.final_score == pytest.approx(50.7)


def test_skill_spec_has_required_fields():
    spec = SkillSpec(
        skill_name="defi-yield-scout",
        trigger_description="Find best DeFi yields",
        expected_output_format="Table with APY",
        example_prompts=["find best yield for USDC"],
        suggested_approach="query defillama",
        need_name="DeFi Yield Comparison",
        need_type="workflow_friction",
        need_description="Users need a single place to compare yields",
        evidence=["i spend hours comparing yields"],
        personas=[UserPersona(segment_name="trader", post_count=10, pct=80, experience_level="intermediate")],
        emotion_profile=EmotionProfile(
            dominant_emotion="frustrated",
            emotion_distribution={"frustrated": 0.7},
            avg_intensity=0.8,
            trend="rising",
        ),
        existing_solutions=[],
        opportunity=OpportunityScore(
            importance=8.0, satisfaction=3.0, raw_score=13.0,
            intensity_weight=1.5, solution_weight=2.0, trend_boost=1.3,
            final_score=50.7,
        ),
        source_cluster_id=1,
        data_period="2024-01-01~2024-01-14",
        post_count=42,
    )
    assert spec.skill_name == "defi-yield-scout"
    assert len(spec.example_prompts) == 1
