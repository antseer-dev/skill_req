import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from datetime import datetime

from stages.l1_clusterer import L1Clusterer, _ctf_idf_keywords, _get_representatives
from models import Post


def make_posts(n: int) -> list[Post]:
    return [
        Post(
            post_id=str(i), text=f"Post about crypto topic number {i} with details",
            creator_id=f"u{i}", network="twitter", interactions=10,
            sentiment_lc=50.0, timestamp=datetime(2024, 1, 1), topic="bitcoin",
        )
        for i in range(n)
    ]


def test_ctf_idf_keywords_returns_per_cluster():
    texts = {
        0: ["defi yield farming protocol apy", "yield high apy stable"],
        1: ["whale wallet moved bitcoin large", "btc whale transfer"],
    }
    result = _ctf_idf_keywords(texts)
    assert 0 in result and 1 in result
    assert len(result[0]) > 0
    # "yield" should be more associated with cluster 0
    assert "yield" in result[0] or "apy" in result[0]


def test_ctf_idf_single_cluster():
    texts = {0: ["bitcoin moon pump bull run"]}
    result = _ctf_idf_keywords(texts)
    assert 0 in result


def test_get_representatives():
    np.random.seed(42)
    embeddings = np.random.rand(10, 8).astype(np.float32)
    labels = np.array([0, 0, 0, 1, 1, 1, -1, 0, 1, -1])
    texts = [f"text_{i}" for i in range(10)]

    reps = _get_representatives(embeddings, labels, texts, top_n=2)
    assert 0 in reps and 1 in reps
    assert -1 not in reps
    assert len(reps[0]) <= 2


@pytest.mark.asyncio
async def test_cluster_too_few_posts():
    mock_client = AsyncMock()
    clusterer = L1Clusterer(client=mock_client)
    posts = make_posts(3)   # below min threshold
    result = await clusterer.cluster(posts)
    # Should return posts unchanged, no API calls made
    assert result == posts
    mock_client.embed.assert_not_called()


@pytest.mark.asyncio
async def test_cluster_assigns_labels():
    mock_client = AsyncMock()
    n = 20
    # Return distinct embeddings so HDBSCAN finds clusters
    mock_client.embed.return_value = np.vstack([
        np.tile([1.0, 0.0], (10, 1)),
        np.tile([0.0, 1.0], (10, 1)),
    ]).astype(np.float32)
    mock_client.chat_json.return_value = {"label": "DeFi yields", "description": "yield needs"}

    posts = make_posts(n)

    with patch("stages.l1_clusterer.UMAP") as MockUMAP:
        mock_umap = MockUMAP.return_value
        mock_umap.fit_transform.return_value = np.vstack([
            np.tile([1.0, 0.0], (10, 1)),
            np.tile([0.0, 1.0], (10, 1)),
        ])
        result = await L1Clusterer(client=mock_client).cluster(posts)

    assert len(result) == n
