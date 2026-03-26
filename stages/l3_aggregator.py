import asyncio
import logging
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import hdbscan
from umap import UMAP

from clients.openrouter import OpenRouterClient
from stages.prompt_utils import render
from models import (
    NeedRecord, PainCluster, UserPersona,
    EmotionProfile, ExistingSolution, OpportunityScore,
)
import config

logger = logging.getLogger(__name__)


def _calc_trend(timeseries: list[dict] | dict) -> str:
    # New path: state dict with velocity field
    if isinstance(timeseries, dict):
        velocity = float(timeseries.get("velocity") or 0)
        if velocity > 0.05:
            return "rising"
        if velocity < -0.05:
            return "cooling"
        return "stable"
    # Legacy path: list of timeseries data points
    sentiments = [float(d.get("sentiment", 50)) for d in timeseries if d.get("sentiment")]
    if len(sentiments) < 4:
        return "stable"
    mid  = len(sentiments) // 2
    diff = np.mean(sentiments[mid:]) - np.mean(sentiments[:mid])
    if diff > 3:
        return "rising"
    if diff < -3:
        return "cooling"
    return "stable"


def _build_personas(records: list[NeedRecord]) -> list[UserPersona]:
    total = len(records)
    if total == 0:
        return []
    type_counter  = Counter(r.user_type for r in records)
    level_counter = Counter(r.experience_level for r in records)
    dominant_level = level_counter.most_common(1)[0][0]
    return [
        UserPersona(
            segment_name=ut,
            post_count=count,
            pct=round(count / total * 100),
            experience_level=dominant_level,
        )
        for ut, count in type_counter.most_common()
        if count / total > 0.20
    ]


def _build_emotion_profile(records: list[NeedRecord], trend: str) -> EmotionProfile:
    emotions = [r.emotion for r in records]
    intensities = [r.intensity for r in records]
    counter = Counter(emotions)
    total = len(records)
    return EmotionProfile(
        dominant_emotion=counter.most_common(1)[0][0] if counter else "neutral",
        emotion_distribution={e: round(c / total, 3) for e, c in counter.most_common(5)},
        avg_intensity=round(float(np.mean(intensities)), 3) if intensities else 0.0,
        trend=trend,
    )


def _calc_opportunity(
    importance: float,
    satisfaction: float,
    avg_intensity: float,
    has_mature_solution: bool,
    trend: str,
) -> OpportunityScore:
    raw   = importance + max(importance - satisfaction, 0)
    i_wt  = 1.5 if avg_intensity > 0.7 else 1.0
    s_wt  = 2.0 if not has_mature_solution else 1.0
    t_wt  = 1.3 if trend == "rising" else 1.0
    return OpportunityScore(
        importance=importance,
        satisfaction=satisfaction,
        raw_score=round(raw, 2),
        intensity_weight=i_wt,
        solution_weight=s_wt,
        trend_boost=t_wt,
        final_score=round(raw * i_wt * s_wt * t_wt, 2),
    )


class L3Aggregator:
    def __init__(self, client: OpenRouterClient | None = None):
        self._client = client or OpenRouterClient()

    async def aggregate(
        self,
        records: list[NeedRecord],
        timeseries: dict[str, list[dict]] | dict[str, dict],
    ) -> list[PainCluster]:
        if not records:
            return []

        logger.info("[L3] embedding %d need records...", len(records))
        pain_texts = [r.pain_point or r.opinion for r in records]
        embeddings = await self._client.embed(pain_texts)

        n = len(records)
        n_components = min(config.UMAP_NEED_COMPONENTS, n - 2)
        n_neighbors  = min(10, n - 1)

        if n < config.HDBSCAN_NEED_MIN_CLUSTER * 2:
            logger.info("[L3] too few records for clustering (%d), putting all in one cluster", n)
            labels = np.zeros(n, dtype=int)
        else:
            logger.info("[L3] UMAP: n_components=%d, n_neighbors=%d", n_components, n_neighbors)
            reduced = UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                min_dist=0.0,
                random_state=42,
            ).fit_transform(embeddings)
            logger.info("[L3] UMAP done — running HDBSCAN (min_cluster_size=%d)", config.HDBSCAN_NEED_MIN_CLUSTER)
            labels = hdbscan.HDBSCAN(
                min_cluster_size=config.HDBSCAN_NEED_MIN_CLUSTER,
                min_samples=config.HDBSCAN_NEED_MIN_SAMPLES,
                metric="euclidean",
            ).fit_predict(reduced)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            noise = int((labels == -1).sum())
            logger.info("[L3] HDBSCAN done — %d cluster(s), %d noise record(s)", n_clusters, noise)

        for record, label in zip(records, labels):
            record.need_cluster_id = int(label) if label != -1 else None

        valid_records = [r for r in records if r.need_cluster_id is not None]
        if not valid_records:
            logger.warning("[L3] no valid clusters formed")
            return []

        clusters_map: dict[int, list[NeedRecord]] = defaultdict(list)
        for r in valid_records:
            clusters_map[r.need_cluster_id].append(r)  # type: ignore

        logger.info("[L3] %d cluster(s) with %d records total", len(clusters_map), len(valid_records))
        for cid, recs in sorted(clusters_map.items()):
            logger.info("[L3]   cluster %d: %d records", cid, len(recs))

        # Get overall trend: state dict (velocity) or list of timeseries points
        first_val = next(iter(timeseries.values()), {})
        if isinstance(first_val, dict):
            velocities = [float(v.get("velocity") or 0) for v in timeseries.values()]
            avg_velocity = sum(velocities) / len(velocities) if velocities else 0
            global_trend = _calc_trend({"velocity": avg_velocity})
        else:
            all_ts: list[dict] = [item for ts in timeseries.values() for item in ts]
            global_trend = _calc_trend(all_ts)
        logger.info("[L3] global trend: %s", global_trend)

        # c-TF-IDF keywords per cluster
        texts_by_cluster = {cid: [r.pain_point or r.opinion for r in recs]
                             for cid, recs in clusters_map.items()}
        keywords = self._ctf_idf(texts_by_cluster)

        # Representative evidence per cluster
        reps = self._get_representatives(
            embeddings, labels,
            [r.evidence for r in records],
        )

        # Parallel: name clusters + identify solutions
        logger.info("[L3] building %d cluster(s) via LLM (parallel)...", len(clusters_map))
        results = await asyncio.gather(*[
            self._build_cluster(cid, recs, keywords.get(cid, []), reps.get(cid, []), global_trend)
            for cid, recs in clusters_map.items()
        ], return_exceptions=True)
        clusters = [r for r in results if not isinstance(r, Exception)]
        failed = len(results) - len(clusters)
        if failed:
            logger.warning("[L3] %d cluster(s) failed to build", failed)

        sorted_clusters = sorted(clusters, key=lambda c: c.opportunity.final_score, reverse=True)
        for i, c in enumerate(sorted_clusters, 1):
            logger.info("[L3]   #%d %r — %d posts, score=%.1f, trend=%s",
                        i, c.cluster_name, c.post_count, c.opportunity.final_score, c.trend)
        return sorted_clusters

    async def _build_cluster(
        self,
        cluster_id: int,
        records: list[NeedRecord],
        keywords: list[str],
        evidence_samples: list[str],
        trend: str,
    ) -> PainCluster:
        name_tmpl = Path("prompts/l3_cluster_name.txt").read_text()
        sol_tmpl  = Path("prompts/l3_existing_solutions.txt").read_text()

        name_prompt = render(
            name_tmpl,
            keywords=", ".join(keywords),
            representative_posts="\n".join(f"- {e}" for e in evidence_samples[:3]),
        )
        sol_prompt = render(
            sol_tmpl,
            cluster_name=str(keywords[:3]),
            description=" ".join(keywords[:5]),
            keywords=", ".join(keywords),
        )

        logger.info("[L3]   cluster %d: calling LLM for name + solutions...", cluster_id)
        results = await asyncio.gather(
            self._client.chat_json(name_prompt),
            self._client.chat_json(sol_prompt),
            return_exceptions=True,
        )
        name_res = results[0] if not isinstance(results[0], Exception) else {}
        sol_res  = results[1] if not isinstance(results[1], Exception) else {}
        if isinstance(results[0], Exception):
            logger.warning("[L3]   cluster %d: name LLM failed — %s", cluster_id, results[0])
        if isinstance(results[1], Exception):
            logger.warning("[L3]   cluster %d: solutions LLM failed — %s", cluster_id, results[1])

        cluster_name = name_res.get("name", f"Cluster {cluster_id}")
        description  = name_res.get("description", "")

        solutions = [
            ExistingSolution(
                name=s.get("name", ""),
                coverage=s.get("coverage", "partial"),
                limitation=s.get("limitation", ""),
            )
            for s in sol_res.get("solutions", [])
        ]

        personas        = _build_personas(records)
        avg_intensity   = float(np.mean([r.intensity for r in records]))
        emotion_profile = _build_emotion_profile(records, trend)
        has_mature      = any(s.coverage == "full" for s in solutions)

        # Heuristic importance/satisfaction before Claude refinement in L4
        importance   = min(10.0, 3.0 + len(records) * 0.3 + avg_intensity * 3)
        satisfaction = 7.0 if has_mature else 3.5

        opportunity = _calc_opportunity(importance, satisfaction, avg_intensity, has_mature, trend)

        scenarios = list({r.root_cause for r in records if r.root_cause})[:5]

        source_topics = list({r.source_topic for r in records if r.source_topic})
        source_topic_labels = list({r.topic_label for r in records if r.topic_label})

        return PainCluster(
            cluster_id=cluster_id,
            cluster_name=cluster_name,
            description=description,
            post_count=len(records),
            top_scenarios=scenarios,
            trend=trend,
            evidence_samples=evidence_samples[:5],
            keywords=keywords,
            personas=personas,
            emotion_profile=emotion_profile,
            existing_solutions=solutions,
            opportunity=opportunity,
            source_topics=source_topics,
            source_topic_labels=source_topic_labels,
        )

    @staticmethod
    def _ctf_idf(texts_by_cluster: dict[int, list[str]], top_n: int = 10) -> dict[int, list[str]]:
        ids  = list(texts_by_cluster.keys())
        docs = [" ".join(texts_by_cluster[i]) for i in ids]
        if len(docs) < 2:
            return {ids[0]: []}

        vectorizer = CountVectorizer(stop_words="english", max_features=1000, min_df=1)
        tf    = vectorizer.fit_transform(docs).toarray().astype(float)
        vocab = vectorizer.get_feature_names_out()
        tf_n  = tf / (tf.sum(axis=1, keepdims=True) + 1e-9)
        idf   = np.log(len(ids) / ((tf > 0).sum(axis=0) + 1))
        score = tf_n * idf
        return {ids[i]: list(vocab[score[i].argsort()[-top_n:][::-1]]) for i in range(len(ids))}

    @staticmethod
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
            mask     = labels == cid
            idx      = np.where(mask)[0]
            centroid = embeddings[mask].mean(axis=0)
            sims     = cosine_similarity([centroid], embeddings[mask])[0]
            top      = sims.argsort()[-top_n:][::-1]
            result[cid] = [texts[idx[i]] for i in top]
        return result
