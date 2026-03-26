"""
Real integration tests — hit actual APIs, no mocks.

Run with:
    LUNARCRUSH_KEY=xxx OPENROUTER_KEY=xxx CLAUDE_API_KEY=xxx \
        pytest tests/integration/ -v -s

Or create a .env file and:
    export $(cat .env | xargs) && pytest tests/integration/ -v -s
"""
import os
import pytest
import numpy as np

# Skip entire module if any key is missing
LUNARCRUSH_KEY = os.environ.get("LUNARCRUSH_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")

need_lunarcrush = pytest.mark.skipif(not LUNARCRUSH_KEY, reason="LUNARCRUSH_KEY not set")
need_openrouter = pytest.mark.skipif(not OPENROUTER_KEY, reason="OPENROUTER_KEY not set")
need_all        = pytest.mark.skipif(
    not (LUNARCRUSH_KEY and OPENROUTER_KEY),
    reason="LUNARCRUSH_KEY and OPENROUTER_KEY required",
)


# ── LunarCrush client ────────────────────────────────────────────

@need_lunarcrush
@pytest.mark.asyncio
async def test_lunarcrush_get_topic_posts():
    from clients.lunarcrush import LunarCrushClient
    client = LunarCrushClient(api_key=LUNARCRUSH_KEY)
    posts = await client.get_topic_posts("bitcoin")

    assert isinstance(posts, list), "Should return a list"
    assert len(posts) > 0, "Should have at least one post"

    post = posts[0]
    assert post.post_id, "post_id must be non-empty"
    assert len(post.text) > 20, "text must be > 20 chars (filter applied)"
    assert post.network, "network must be set"
    assert post.topic == "bitcoin"
    print(f"\n  fetched {len(posts)} posts for 'bitcoin'")
    print(f"  sample: {posts[0].text[:100]!r}")


@need_lunarcrush
@pytest.mark.asyncio
async def test_lunarcrush_get_topic_timeseries():
    from clients.lunarcrush import LunarCrushClient
    client = LunarCrushClient(api_key=LUNARCRUSH_KEY)
    series = await client.get_topic_timeseries("bitcoin")

    assert isinstance(series, list)
    assert len(series) > 0, "Should return time-series data points"
    print(f"\n  {len(series)} time-series points returned")
    print(f"  sample keys: {list(series[0].keys())}")


@need_lunarcrush
@pytest.mark.asyncio
async def test_lunarcrush_cache_hit():
    """Second call for same topic should hit cache (no HTTP request)."""
    from clients.lunarcrush import LunarCrushClient
    import time

    client = LunarCrushClient(api_key=LUNARCRUSH_KEY)
    # Prime the cache
    posts1 = await client.get_topic_posts("ethereum")

    t0 = time.monotonic()
    posts2 = await client.get_topic_posts("ethereum")
    elapsed = time.monotonic() - t0

    assert len(posts1) == len(posts2), "Cache must return same data"
    assert elapsed < 1.5, f"Cache hit should be fast, took {elapsed:.2f}s"
    print(f"\n  cache hit in {elapsed*1000:.1f}ms")


# ── OpenRouter embed ─────────────────────────────────────────────

@need_openrouter
@pytest.mark.asyncio
async def test_openrouter_embed_returns_ndarray():
    from clients.openrouter import OpenRouterClient
    client = OpenRouterClient(api_key=OPENROUTER_KEY)

    texts = [
        "gas fees on ethereum are too high",
        "DeFi yield comparison is painful",
        "whale wallet tracking is hard",
    ]
    embeddings = await client.embed(texts)

    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts), "One embedding per text"
    assert embeddings.shape[1] > 100, "Embedding dim should be large (qwen3-8b = 4096)"
    assert embeddings.dtype == np.float32
    print(f"\n  embedding shape: {embeddings.shape}")


@need_openrouter
@pytest.mark.asyncio
async def test_openrouter_embed_semantic_similarity():
    """Similar texts should have higher cosine similarity than unrelated ones."""
    from clients.openrouter import OpenRouterClient
    from sklearn.metrics.pairwise import cosine_similarity

    client = OpenRouterClient(api_key=OPENROUTER_KEY)
    texts = [
        "gas fees are too expensive on ethereum",
        "ethereum transaction costs are too high",   # similar to [0]
        "defi yield farming strategy tutorial",      # unrelated
    ]
    emb = await client.embed(texts)

    sim_related   = cosine_similarity([emb[0]], [emb[1]])[0][0]
    sim_unrelated = cosine_similarity([emb[0]], [emb[2]])[0][0]

    print(f"\n  related sim: {sim_related:.4f}, unrelated sim: {sim_unrelated:.4f}")
    assert sim_related > sim_unrelated, "Similar texts must score higher than unrelated"


# ── OpenRouter chat (Gemini) ─────────────────────────────────────

@need_openrouter
@pytest.mark.asyncio
async def test_openrouter_chat_json_returns_dict():
    from clients.openrouter import OpenRouterClient
    client = OpenRouterClient(api_key=OPENROUTER_KEY)

    prompt = (
        "You are a JSON API. Respond ONLY with valid JSON.\n"
        "Analyze this crypto user post and return JSON with keys: "
        "entity (str), sentiment (positive/negative/neutral), emotion (str), intensity (float 0-1).\n\n"
        "Post: 'ETH gas fees are absolutely insane, I cannot make any small trades anymore'"
    )
    result = await client.chat_json(prompt)

    assert isinstance(result, dict), "Must return a dict"
    assert "entity" in result or "sentiment" in result, "Must have expected keys"
    print(f"\n  response: {result}")


@need_openrouter
@pytest.mark.asyncio
async def test_openrouter_chat_json_intensity_range():
    from clients.openrouter import OpenRouterClient
    client = OpenRouterClient(api_key=OPENROUTER_KEY)

    prompt = (
        "You are a JSON API. Respond ONLY with valid JSON.\n"
        "Return JSON with exactly one key: intensity (float between 0.0 and 1.0) "
        "representing emotional intensity of this post.\n\n"
        "Post: 'I am so frustrated with these gas fees, this is absolutely ridiculous'"
    )
    result = await client.chat_json(prompt)

    assert "intensity" in result
    assert 0.0 <= float(result["intensity"]) <= 1.0
    print(f"\n  intensity: {result['intensity']}")


# ── L4 spec generator (OpenRouter) ──────────────────────────────

@need_openrouter
@pytest.mark.asyncio
async def test_l4_spec_generator_single_cluster():
    from stages.l4_spec_generator import L4SpecGenerator
    from models import (
        PainCluster, UserPersona, EmotionProfile,
        ExistingSolution, OpportunityScore,
    )

    cluster = PainCluster(
        cluster_id=0,
        cluster_name="DeFi yield comparison",
        description="Users struggle to compare yields across DeFi protocols manually",
        post_count=30,
        top_scenarios=["missing_api", "high_cost"],
        trend="rising",
        evidence_samples=[
            "i spend 3 hours every day comparing yields across 5 sites",
            "there is no single tool that shows all DeFi APYs in one place",
        ],
        keywords=["yield", "apy", "defi", "comparison", "protocol"],
        personas=[UserPersona(segment_name="trader", post_count=20, pct=67, experience_level="intermediate")],
        emotion_profile=EmotionProfile(
            dominant_emotion="frustrated",
            emotion_distribution={"frustrated": 0.7, "confused": 0.3},
            avg_intensity=0.8,
            trend="rising",
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

    generator = L4SpecGenerator()
    specs = await generator.generate([cluster])

    assert len(specs) == 1
    spec = specs[0]
    assert spec.skill_name, "skill_name must be set"
    assert spec.trigger_description, "trigger_description must be set"
    assert len(spec.example_prompts) >= 1, "must have example prompts"
    assert spec.opportunity.final_score > 0
    print(f"\n  L4 spec: {spec.skill_name!r}")
    print(f"  trigger: {spec.trigger_description[:80]!r}")
    print(f"  prompts: {spec.example_prompts[:2]}")
    print(f"  score: {spec.opportunity.final_score}")


# ── Stage integration: L0 collector ──────────────────────────────

@need_lunarcrush
@pytest.mark.asyncio
async def test_l0_collector_single_topic():
    from stages.l0_collector import L0Collector
    collector = L0Collector()

    posts, timeseries = await collector.collect(topics=["bitcoin"])

    assert len(posts) > 0, "Must collect posts"
    assert "bitcoin" in timeseries, "Must have timeseries for bitcoin"
    assert len(timeseries["bitcoin"]) > 0

    # All posts should have valid text
    assert all(len(p.text.strip()) > 20 for p in posts)
    print(f"\n  L0: {len(posts)} posts, {len(timeseries['bitcoin'])} timeseries points")


# ── Stage integration: L2 analyzer (single post) ─────────────────

@need_openrouter
@pytest.mark.asyncio
async def test_l2_analyzer_single_post():
    from stages.l2_analyzer import L2Analyzer
    from models import Post
    from datetime import datetime

    post = Post(
        post_id="test_1",
        text="I am so frustrated. Gas fees on ETH are insane. I can't make any small DeFi trades anymore.",
        creator_id="user1",
        network="twitter",
        interactions=100,
        sentiment_lc=30.0,
        timestamp=datetime(2024, 1, 1),
        topic="ethereum",
    )

    analyzer = L2Analyzer()
    records = await analyzer.analyze([post])

    assert len(records) == 1, "Should produce 1 NeedRecord"
    r = records[0]
    assert r.entity, "entity must be set"
    assert r.sentiment in ("positive", "negative", "neutral")
    assert r.emotion, "emotion must be set"
    assert 0.0 <= r.intensity <= 1.0
    assert r.user_type in ("trader", "developer", "investor", "researcher", "casual")
    print(f"\n  L2 record: entity={r.entity!r} sentiment={r.sentiment} emotion={r.emotion} intensity={r.intensity}")


@need_openrouter
@pytest.mark.asyncio
async def test_l2_analyzer_batch_skips_errors():
    """If one post fails NLP, others should still succeed."""
    from stages.l2_analyzer import L2Analyzer
    from models import Post
    from datetime import datetime

    posts = [
        Post(post_id=str(i), text=f"Gas fees on ETH are too high, hurting small traders #{i}",
             creator_id="u1", network="twitter", interactions=50, sentiment_lc=35.0,
             timestamp=datetime(2024, 1, 1), topic="ethereum")
        for i in range(3)
    ]

    analyzer = L2Analyzer()
    records = await analyzer.analyze(posts)

    assert len(records) >= 2, "At least 2 of 3 should succeed"
    assert all(0.0 <= r.intensity <= 1.0 for r in records)
    print(f"\n  L2 batch: {len(records)}/3 records produced")


# ── Full pipeline integration ────────────────────────────────────

@need_all
@pytest.mark.asyncio
async def test_full_pipeline_single_topic():
    """
    End-to-end real pipeline run using the mock task API.
    This hits all 4 stages with real API calls.
    Expect ~1-3 minutes to complete.
    """
    import pipeline

    specs = await pipeline.run()

    assert isinstance(specs, list)
    # With real data we should get at least some specs
    # (empty is allowed if clustering finds no valid clusters)
    print(f"\n  Full pipeline: {len(specs)} SkillSpec(s) produced")

    for spec in specs:
        assert spec.skill_name, "skill_name must be set"
        assert spec.trigger_description, "trigger_description must be set"
        assert len(spec.example_prompts) > 0, "must have example prompts"
        assert spec.opportunity.final_score >= 0
        assert spec.post_count > 0
        print(f"  - {spec.skill_name}: score={spec.opportunity.final_score:.1f} posts={spec.post_count}")
