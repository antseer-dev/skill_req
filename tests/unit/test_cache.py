import pytest
import tempfile
import os
from storage.cache import Cache


@pytest.fixture
def cache(tmp_path):
    db = str(tmp_path / "test.db")
    return Cache(db_path=db)


def test_set_and_get(cache):
    cache.set("posts", "bitcoin", value={"items": [1, 2, 3]})
    result = cache.get("posts", "bitcoin")
    assert result == {"items": [1, 2, 3]}


def test_miss_returns_none(cache):
    assert cache.get("posts", "nonexistent") is None


def test_expired_returns_none(cache):
    cache.set("posts", "bitcoin", value={"x": 1}, ttl_hours=0)
    # TTL=0 expires immediately
    result = cache.get("posts", "bitcoin")
    assert result is None


def test_overwrite(cache):
    cache.set("posts", "eth", value={"v": 1})
    cache.set("posts", "eth", value={"v": 2})
    assert cache.get("posts", "eth") == {"v": 2}


def test_delete(cache):
    cache.set("posts", "sol", value={"x": 1})
    cache.delete("posts", "sol")
    assert cache.get("posts", "sol") is None


def test_clear_expired(cache):
    cache.set("posts", "a", value=1, ttl_hours=24)
    cache.set("posts", "b", value=2, ttl_hours=0)
    cache.clear_expired()
    assert cache.get("posts", "a") == 1
    assert cache.get("posts", "b") is None


def test_different_namespaces(cache):
    cache.set("posts", "btc", value=1)
    cache.set("timeseries", "btc", value=2)
    assert cache.get("posts", "btc") == 1
    assert cache.get("timeseries", "btc") == 2
