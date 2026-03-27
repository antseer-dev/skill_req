import asyncio
import logging
from pathlib import Path
from clients.openrouter import OpenRouterClient
from models import PainCluster, SkillSpec, OpportunityScore
from stages.prompt_utils import render
import config

logger = logging.getLogger(__name__)


class L4SpecGenerator:
    def __init__(self, client: OpenRouterClient | None = None):
        self._client = client or OpenRouterClient()

    async def generate(self, clusters: list[PainCluster]) -> list[SkillSpec]:
        top = clusters[: config.TOP_N_SKILLS]
        logger.info("[L4] generating specs for top %d/%d cluster(s) (model=%s, parallel)...",
                    len(top), len(clusters), config.MODEL_REPORT)

        async def _build_one(i: int, c: PainCluster) -> SkillSpec:
            logger.info("[L4]   [%d/%d] building spec for cluster %d: %r", i, len(top), c.cluster_id, c.cluster_name)
            spec = await self._build_spec(c)
            logger.info("[L4]   [%d/%d] done → skill_name=%r, score=%.1f",
                        i, len(top), spec.skill_name, spec.opportunity.final_score)
            return spec

        results = await asyncio.gather(*[_build_one(i, c) for i, c in enumerate(top, 1)])
        return list(results)

    async def _build_spec(self, cluster: PainCluster) -> SkillSpec:
        personas_str = "\n".join(
            f"- {p.segment_name} ({p.pct}%, {p.experience_level})"
            for p in cluster.personas
        )
        solutions_str = "\n".join(
            f"- {s.name}: {s.limitation} (coverage: {s.coverage})"
            for s in cluster.existing_solutions
        ) or "None identified"

        topic_labels_str = ", ".join(cluster.source_topic_labels) or "general financial discussion"

        prompt = render(
            Path("prompts/l4_skill_spec.txt").read_text(),
            cluster_name=cluster.cluster_name,
            description=cluster.description,
            keywords=", ".join(cluster.keywords),
            top_scenarios=", ".join(cluster.top_scenarios) or "general",
            trend=cluster.trend,
            source_topic_labels=topic_labels_str,
            evidence_samples="\n".join(f'"{e}"' for e in cluster.evidence_samples),
            personas=personas_str or "Not determined",
            existing_solutions=solutions_str,
            dominant_emotion=cluster.emotion_profile.dominant_emotion,
            avg_intensity=cluster.emotion_profile.avg_intensity,
        )

        raw = await self._client.chat_json(prompt, model=config.MODEL_REPORT)

        # Refine opportunity score with LLM's importance/satisfaction
        importance   = float(raw.get("importance", cluster.opportunity.importance))
        satisfaction = float(raw.get("satisfaction", cluster.opportunity.satisfaction))
        has_mature   = bool(raw.get("has_mature_solution", False))
        i_wt  = 1.5 if cluster.emotion_profile.avg_intensity > 0.7 else 1.0
        s_wt  = 2.0 if not has_mature else 1.0
        t_wt  = cluster.opportunity.trend_boost
        raw_score = importance + max(importance - satisfaction, 0)

        opportunity = OpportunityScore(
            importance=importance,
            satisfaction=satisfaction,
            raw_score=round(raw_score, 2),
            intensity_weight=i_wt,
            solution_weight=s_wt,
            trend_boost=t_wt,
            final_score=round(raw_score * i_wt * s_wt * t_wt, 2),
        )

        return SkillSpec(
            skill_name=raw.get("skill_name", "unknown-skill"),
            trigger_description=raw.get("trigger_description", ""),
            expected_output_format=raw.get("expected_output_format", ""),
            example_prompts=raw.get("example_prompts", []),
            suggested_approach=raw.get("suggested_approach", ""),
            need_name=raw.get("need_name", cluster.cluster_name),
            need_type=raw.get("need_type", "missing_feature"),
            need_description=raw.get("need_description", cluster.description),
            evidence=cluster.evidence_samples,
            personas=cluster.personas,
            emotion_profile=cluster.emotion_profile,
            existing_solutions=cluster.existing_solutions,
            opportunity=opportunity,
            source_cluster_id=cluster.cluster_id,
            data_period="",
            post_count=cluster.post_count,
            source_topics=cluster.source_topics,
            source_topic_labels=cluster.source_topic_labels,
        )
