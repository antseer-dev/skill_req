import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path

import config
from stages.l0_collector import L0Collector
from stages.l1_clusterer import L1Clusterer
from stages.l2_analyzer import L2Analyzer
from stages.l3_aggregator import L3Aggregator
from stages.l4_spec_generator import L4SpecGenerator
from models import SkillSpec

logger = logging.getLogger(__name__)


async def run(run_id: str | None = None, topic: str | None = None) -> list[SkillSpec]:
    logger.info("pipeline starting (run_id=%s, topic=%s)", run_id or "auto", topic or "none")

    # L0: collect from task API
    #   - topic mode: load single topic from local file (topic_{topic}.json)
    #   - task_id mode: fetch by task_id from main API
    #   - otherwise: fetch all from mock API
    logger.info("[L0] collecting posts...")
    if topic:
        posts, states = await L0Collector(topic=topic).collect_topic_file(topic)
        logger.info("[L0] %d posts collected for topic %s", len(posts), topic)
    else:
        posts, states = await L0Collector(task_id=run_id).collect()
        logger.info("[L0] %d posts collected from %d topics", len(posts), len(states))
    if not posts:
        logger.warning("[L0] no posts — aborting")
        return []

    # L1: embed + cluster posts
    logger.info("[L1] clustering posts...")
    posts = await L1Clusterer().cluster(posts)
    clustered = sum(1 for p in posts if p.cluster_id is not None)
    logger.info("[L1] %d/%d posts clustered", clustered, len(posts))

    # L2: structured NLP per post
    logger.info("[L2] analyzing posts...")
    records = await L2Analyzer().analyze(posts)
    logger.info("[L2] %d need records extracted", len(records))
    if not records:
        logger.warning("[L2] no records — aborting")
        return []

    # L3: aggregate into pain clusters
    logger.info("[L3] aggregating pain clusters...")
    clusters = await L3Aggregator().aggregate(records, states)
    logger.info("[L3] %d pain clusters found", len(clusters))
    if not clusters:
        logger.warning("[L3] no clusters — aborting")
        return []

    # L4: generate skill specs
    logger.info("[L4] generating skill specs...")
    specs = await L4SpecGenerator().generate(clusters)
    logger.info("[L4] %d skill specs generated", len(specs))

    _write_output(specs, posts, clusters, run_id=run_id or topic)
    return specs


def _write_output(specs: list[SkillSpec], posts, clusters, run_id: str | None = None) -> None:
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    suffix = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    specs_path = f"{config.OUTPUT_DIR}/skill_specs_{suffix}.json"
    with open(specs_path, "w") as f:
        json.dump([s.model_dump(mode="json") for s in specs], f, indent=2, ensure_ascii=False)
    logger.info("[output] %s", specs_path)

    report_path = f"{config.OUTPUT_DIR}/skill_req_report_{suffix}.md"
    with open(report_path, "w") as f:
        f.write(_build_report(specs, posts, clusters, suffix))
    logger.info("[output] %s", report_path)


def _build_report(specs, posts, clusters, date_str: str) -> str:
    lines = [
        f"# Skill 需求报告 {date_str}\n",
        f"- 帖子总量：{len(posts)} 条",
        f"- 需求簇：{len(clusters)} 个",
        f"- 输出 Skill：{len(specs)} 个\n",
        "## Top Skill 需求\n",
        "| 排名 | Skill 名称 | 核心需求 | 机会分 | 趋势 | 原始话题 | L1 聚类标签 |",
        "|-----|-----------|---------|--------|------|--------|-----------|",
    ]
    trend_icon = {"rising": "📈", "stable": "➡️", "cooling": "📉"}
    for i, s in enumerate(specs, 1):
        icon = trend_icon.get(s.emotion_profile.trend, "➡️")
        raw_topics = ", ".join(s.source_topics) if s.source_topics else "—"
        topic_labels = ", ".join(s.source_topic_labels) if s.source_topic_labels else "—"
        lines.append(
            f"| {i} | `{s.skill_name}` | {s.need_name} | {s.opportunity.final_score} | {icon} | {raw_topics} | {topic_labels} |"
        )

    lines.append("\n## Skill 详情\n")
    for s in specs:
        lines += [
            f"### {s.skill_name}",
            f"**需求**：{s.need_description}",
            f"**原始话题**：{', '.join(s.source_topics) if s.source_topics else '—'}",
            f"**L1 聚类标签**：{', '.join(s.source_topic_labels) if s.source_topic_labels else '—'}",
            f"**画像**：{', '.join(p.segment_name for p in s.personas)}",
            f"**机会分**：{s.opportunity.final_score} (importance={s.opportunity.importance}, satisfaction={s.opportunity.satisfaction})",
            f"**情绪**：{s.emotion_profile.dominant_emotion} (intensity={s.emotion_profile.avg_intensity})",
            "**证据**：",
            *[f"> {e}" for e in s.evidence[:3]],
            "",
        ]

    lines += [
        "## 方法论",
        "- 数据：Mock Task API (inflation, recession 等金融话题)",
        "- 聚类：Qwen3-embedding-8b → UMAP → HDBSCAN + c-TF-IDF",
        "- NLP：Gemini Flash (L1-L3) + Claude Sonnet via OpenRouter (L4)",
        "- 评分：ODI Opportunity Score (Ulwick)",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
    log_path = f"{config.OUTPUT_DIR}/pipeline.log"

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s — %(message)s", datefmt="%H:%M:%S")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)

    logging.basicConfig(level=logging.INFO, handlers=[file_handler, console_handler])

    asyncio.run(run())
