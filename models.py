from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


# ── L0 ───────────────────────────────────────────────────
class Post(BaseModel):
    post_id: str
    text: str
    creator_id: str
    network: str                        # twitter / reddit / youtube...
    creator_followers: Optional[int] = None
    interactions: int
    sentiment_lc: float                 # LunarCrush 情绪值 0-100
    timestamp: datetime
    topic: str                          # 来源话题
    cluster_id: Optional[int] = None
    topic_label: Optional[str] = None
    topic_keywords: Optional[list[str]] = None


# ── L2 ───────────────────────────────────────────────────
class NeedRecord(BaseModel):
    post_id: str
    entity: str                         # 涉及产品/项目
    feature: str                        # 具体功能/方面
    opinion: str                        # 用户观点
    need_type: str                      # explicit | implicit
    pain_point: Optional[str] = None
    root_cause: Optional[str] = None
    sentiment: str                      # positive | negative | neutral
    emotion: str                        # 细粒度情绪
    intensity: float = Field(ge=0.0, le=1.0)
    evidence: str                       # 原文引用
    user_type: str                      # trader | developer | investor | researcher | casual
    experience_level: str               # beginner | intermediate | expert
    need_cluster_id: Optional[int] = None
    # original API topic name
    source_topic: Optional[str] = None
    # L1 cluster context carried forward
    topic_label: Optional[str] = None
    topic_keywords: Optional[list[str]] = None


# ── L3 ───────────────────────────────────────────────────
class UserPersona(BaseModel):
    segment_name: str                   # trader / developer / ...
    post_count: int
    pct: int                            # 占需求簇百分比
    experience_level: str               # beginner / intermediate / expert


class EmotionProfile(BaseModel):
    dominant_emotion: str
    emotion_distribution: dict[str, float]
    avg_intensity: float
    trend: str                          # rising | stable | cooling


class ExistingSolution(BaseModel):
    name: str
    coverage: str                       # partial | full | none
    limitation: str


class OpportunityScore(BaseModel):
    importance: float                   # 1-10
    satisfaction: float                 # 1-10
    raw_score: float
    intensity_weight: float
    solution_weight: float
    trend_boost: float
    final_score: float


class PainCluster(BaseModel):
    cluster_id: int
    cluster_name: str
    description: str
    post_count: int
    top_scenarios: list[str]
    trend: str                          # rising | stable | cooling
    evidence_samples: list[str]
    keywords: list[str]
    personas: list[UserPersona]
    emotion_profile: EmotionProfile
    existing_solutions: list[ExistingSolution]
    opportunity: OpportunityScore
    source_topics: list[str] = []        # original API topic names (e.g. "inflation")
    source_topic_labels: list[str] = []  # L1 cluster labels (Gemini-generated)


# ── L4 ───────────────────────────────────────────────────
class SkillSpec(BaseModel):
    """交付给 skill-creator 的完整规格"""

    # Skill 基本信息
    skill_name: str
    trigger_description: str
    expected_output_format: str
    example_prompts: list[str]
    suggested_approach: str

    # 需求表
    need_name: str
    need_type: str                      # missing_feature | workflow_friction | integration_gap
    need_description: str

    # 证据
    evidence: list[str]

    # 用户画像
    personas: list[UserPersona]

    # 情绪分析
    emotion_profile: EmotionProfile

    # 现有方案
    existing_solutions: list[ExistingSolution]

    # 产品机会
    opportunity: OpportunityScore

    # 元数据
    source_cluster_id: int
    data_period: str
    post_count: int
    source_topics: list[str] = []        # original API topic names
    source_topic_labels: list[str] = []  # L1 cluster labels (Gemini-generated)
