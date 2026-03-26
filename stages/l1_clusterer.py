import asyncio
import logging
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from umap import UMAP

from pathlib import Path
from clients.openrouter import OpenRouterClient
from models import Post
from stages.prompt_utils import render
import config

logger = logging.getLogger(__name__)


def _ctf_idf_keywords(texts_by_cluster: dict[int, list[str]], top_n: int = 10) -> dict[int, list[str]]:
    ids  = list(texts_by_cluster.keys())
    docs = [" ".join(texts_by_cluster[i]) for i in ids]
    if len(docs) < 2:
        return {ids[0]: [] for _ in ids}

    vectorizer = CountVectorizer(stop_words="english", max_features=1000, min_df=1)
    tf    = vectorizer.fit_transform(docs).toarray().astype(float)
    vocab = vectorizer.get_feature_names_out()

    tf_norm  = tf / (tf.sum(axis=1, keepdims=True) + 1e-9)
    idf      = np.log(len(ids) / ((tf > 0).sum(axis=0) + 1))
    ctfidf   = tf_norm * idf

    return {
        ids[i]: list(vocab[ctfidf[i].argsort()[-top_n:][::-1]])
        for i in range(len(ids))
    }


def _get_representatives(
    embeddings: np.ndarray,
    labels: np.ndarray,
    texts: list[str],
    top_n: int = 5,
) -> dict[int, list[str]]:
    result: dict[int, list[str]] = {}
    for cid in set(labels):
        if cid == -1:
            continue
        mask    = labels == cid
        idx     = np.where(mask)[0]
        centroid = embeddings[mask].mean(axis=0)
        sims    = cosine_similarity([centroid], embeddings[mask])[0]
        top_idx = sims.argsort()[-top_n:][::-1]
        result[cid] = [texts[idx[i]] for i in top_idx]
    return result


class L1Clusterer:
    def __init__(self, client: OpenRouterClient | None = None):
        self._client = client or OpenRouterClient()

    async def cluster(self, posts: list[Post]) -> list[Post]:
        if len(posts) < config.HDBSCAN_MIN_CLUSTER * 2:
            logger.info("[L1] too few posts (%d), skipping clustering", len(posts))
            return posts

        texts = [p.text for p in posts]
        logger.info("[L1] embedding %d posts...", len(texts))
        embeddings = await self._client.embed(texts)

        n_neighbors = min(config.UMAP_N_NEIGHBORS, len(posts) - 1)
        n_components = min(config.UMAP_COMPONENTS, len(posts) - 2)
        logger.info("[L1] UMAP: n_components=%d, n_neighbors=%d", n_components, n_neighbors)
        reduced = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=0.0,
            random_state=42,
        ).fit_transform(embeddings)
        logger.info("[L1] UMAP done — reduced shape=%s", reduced.shape)

        logger.info("[L1] HDBSCAN: min_cluster_size=%d, min_samples=%d",
                    config.HDBSCAN_MIN_CLUSTER, config.HDBSCAN_MIN_SAMPLES)
        labels = hdbscan.HDBSCAN(
            min_cluster_size=config.HDBSCAN_MIN_CLUSTER,
            min_samples=config.HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
        ).fit_predict(reduced)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise = int((labels == -1).sum())
        logger.info("[L1] HDBSCAN done — %d cluster(s), %d noise point(s)", n_clusters, noise)

        texts_by_cluster: dict[int, list[str]] = defaultdict(list)
        for text, label in zip(texts, labels):
            if label != -1:
                texts_by_cluster[int(label)].append(text)

        if not texts_by_cluster:
            logger.warning("[L1] no valid clusters formed, returning posts as-is")
            return posts

        keywords    = _ctf_idf_keywords(dict(texts_by_cluster))
        reps        = _get_representatives(embeddings, labels, texts)
        logger.info("[L1] labeling %d cluster(s) via LLM...", len(keywords))
        labels_dict = await self._label_clusters(keywords, reps)

        for post, label in zip(posts, labels):
            if label != -1:
                post.cluster_id     = int(label)
                post.topic_label    = labels_dict.get(label, {}).get("name")
                post.topic_keywords = keywords.get(label, [])

        for cid, name_dict in labels_dict.items():
            kws = keywords.get(cid, [])[:5]
            logger.info("[L1]   cluster %d: %r  keywords=%s", cid, name_dict.get("name"), kws)

        return posts

    async def _label_clusters(
        self,
        keywords: dict[int, list[str]],
        reps: dict[int, list[str]],
    ) -> dict[int, dict]:
        prompt_tmpl = Path("prompts/l3_cluster_name.txt").read_text()
        sem = asyncio.Semaphore(config.GEMINI_CONCURRENCY)

        async def label_one(cid: int) -> tuple[int, dict]:
            prompt = render(
                prompt_tmpl,
                keywords=", ".join(keywords.get(cid, [])),
                representative_posts="\n".join(f"- {t}" for t in reps.get(cid, [])[:3]),
            )
            async with sem:
                result = await self._client.chat_json(prompt)
            return cid, result

        tasks   = [label_one(cid) for cid in keywords]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            cid: res
            for item in results
            if not isinstance(item, Exception)
            for cid, res in [item]
        }
