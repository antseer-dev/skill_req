# 社交媒体 → Skill 需求挖掘 Pipeline — 系统设计

## 1. 目标

从 LunarCrush 加密/金融话题的社交数据中挖掘用户真实需求，
输出包含完整研究证据的 **SkillSpec**，交付给 `skill-creator` 生成 Claude Code Skill。

```
LunarCrush 加密/金融话题
    ↓ 用户需求分析
    ↓ 需求表 + 证据 + 画像 + 情绪 + 现有方案 + 产品机会
    ↓
SkillSpec[] → skill-creator → SKILL.md
```

---

## 2. 技术栈

| 层次 | 工具 | 用途 |
|------|------|------|
| 数据源 | LunarCrush API | 帖子、话题、创作者、时间序列 |
| 向量化 | qwen/qwen3-embedding-8b (OpenRouter) | 文本嵌入 |
| 聚类 | UMAP + HDBSCAN | 降维 + 密度聚类（替代 KMeans） |
| 批量 NLP | google/gemini-2.5-flash-preview (OpenRouter) | L1/L2/L3 批量分析 |
| 深度推理 | anthropic/claude-sonnet-4-5 (Claude API) | L4 SkillSpec 生成 |
| 存储 | SQLite | 缓存 + 中间结果 |
| 运行 | Python 3.11+ | — |

---

## 3. 架构总览

```
┌───────────────────────────────────────────────────┐
│              Pipeline Orchestrator                 │
│                  pipeline.py                       │
└──────┬────────────────────────────────────────────┘
       │
 ┌─────▼──────┐
 │  L0 采集   │  LunarCrush topic posts + time-series
 │            │  加密/金融话题，每话题 top 50 帖
 └─────┬──────┘
       │  List[Post]
 ┌─────▼──────┐
 │  L1 聚类   │  Qwen3 embedding → UMAP → HDBSCAN
 │            │  代表性文本 + c-TF-IDF 关键词 + Gemini 标签
 └─────┬──────┘
       │  List[Post + cluster_id + topic_label]
 ┌─────▼──────┐
 │  L2 分析   │  Gemini 并发结构化 NLP
 │            │  输出 NeedRecord（需求 + 情绪 + user_type 画像）
 └─────┬──────┘
       │  List[NeedRecord]
 ┌─────▼──────┐
 │  L3 聚合   │  需求 embedding → UMAP → HDBSCAN → PainCluster
 │            │  代表性文本 + c-TF-IDF + Gemini 命名
 │            │  user_type 分布统计（定性画像）
 │            │  关联 LunarCrush 趋势 → 机会评分
 └─────┬──────┘
       │  List[PainCluster] ranked by opportunity_score
 ┌─────▼──────┐
 │  L4 生成   │  Claude Sonnet
 │            │  每个 Top PainCluster → SkillSpec
 │            │  包含：需求表+证据+画像+情绪+现有方案+机会
 └─────┬──────┘
       │
 ┌─────▼──────┐
 │  输出      │  skill_specs.json  → skill-creator
 │            │  skill_req_report.md → 人工审阅
 └────────────┘
```

---

## 4. 数据模型（models.py）

```python
from pydantic import BaseModel
from typing import Optional
from datetime import datetime


# ── L0 ──────────────────────────────────────────────────
class Post(BaseModel):
    post_id: str
    text: str
    creator_id: str
    network: str                   # twitter / reddit / youtube...
    creator_followers: Optional[int] = None
    interactions: int
    sentiment_lc: float            # LunarCrush 情绪值 0-100
    timestamp: datetime
    topic: str                     # 来源话题
    cluster_id: Optional[int] = None
    topic_label: Optional[str] = None


# ── L2 ──────────────────────────────────────────────────
class NeedRecord(BaseModel):
    post_id: str
    entity: str                    # 涉及的产品/项目/概念
    feature: str                   # 具体功能/方面
    opinion: str                   # 用户观点
    need_type: str                 # explicit | implicit
    pain_point: Optional[str]      # 痛点描述
    root_cause: Optional[str]      # 根因标签
    sentiment: str                 # positive | negative | neutral
    emotion: str                   # 细粒度情绪（27类）
    intensity: float               # 情绪强度 0.0~1.0
    evidence: str                  # 原文引用
    # 画像线索
    user_type_hint: str            # trader / developer / investor / researcher...
    experience_hint: str           # beginner / intermediate / expert


# ── L3 ──────────────────────────────────────────────────
class UserPersona(BaseModel):
    segment_name: str              # trader / developer / investor / researcher / casual
    post_count: int                # 该类型贡献的帖子数
    pct: int                       # 占该需求簇的百分比
    experience_level: str          # beginner / intermediate / expert
    # 用途：trigger_description 措辞风格 + SkillSpec 输出深度参考

class EmotionProfile(BaseModel):
    dominant_emotion: str          # 主情绪
    emotion_distribution: dict[str, float]  # {"frustrated": 0.4, "confused": 0.3, ...}
    avg_intensity: float
    trend: str                     # rising / stable / cooling

class ExistingSolution(BaseModel):
    name: str                      # 现有方案名称
    coverage: str                  # partial / full / none
    limitation: str                # 限制/不足之处

class OpportunityScore(BaseModel):
    importance: float              # 1-10（Claude 评估）
    satisfaction: float            # 1-10（现有方案满足度）
    raw_score: float               # importance + max(imp-sat, 0)
    intensity_weight: float        # 情绪强度权重（强×1.5）
    solution_weight: float         # 无成熟方案权重（无×2.0）
    final_score: float             # 最终排序分
    trend_boost: float             # 趋势加成（rising×1.3）

class PainCluster(BaseModel):
    cluster_id: int
    cluster_name: str              # Gemini 生成的需求名
    description: str
    post_count: int
    top_scenarios: list[str]
    trend: str                     # rising / stable / cooling
    evidence_samples: list[str]    # 3-5条代表性原文
    personas: list[UserPersona]
    emotion_profile: EmotionProfile
    existing_solutions: list[ExistingSolution]
    opportunity: OpportunityScore


# ── L4 ──────────────────────────────────────────────────
class SkillSpec(BaseModel):
    """交付给 skill-creator 的完整规格"""

    # —— Skill 基本信息 ——
    skill_name: str                # kebab-case，如 "defi-yield-scout"
    trigger_description: str       # SKILL.md frontmatter description
                                   # 含：做什么 + 何时触发 + 真实措辞覆盖
    expected_output_format: str    # 输出是什么样的
    example_prompts: list[str]     # 3条真实测试 prompt → evals/evals.json
    suggested_approach: str        # 实现思路提示

    # —— 需求表 ——
    need_name: str                 # 需求名称
    need_type: str                 # missing_feature / workflow_friction / integration_gap
    need_description: str          # 需求详细描述

    # —— 证据 ——
    evidence: list[str]            # 3-5条原文引用

    # —— 用户画像 ——
    personas: list[UserPersona]    # 1-3个主要用户画像

    # —— 情绪分析 ——
    emotion_profile: EmotionProfile

    # —— 现有方案 ——
    existing_solutions: list[ExistingSolution]

    # —— 产品机会 ——
    opportunity: OpportunityScore

    # —— 元数据 ——
    source_cluster_id: int
    data_period: str               # 数据覆盖时间段
    post_count: int                # 支撑该需求的帖子数
```

---

## 5. 各层设计

### L0 — 数据采集

**话题不变，与原始设计一致（加密/金融/AI 等）**

```python
GET /api4/public/topic/{topic}/posts/v1
GET /api4/public/topic/{topic}/time-series/v1?bucket=days&interval=2w
GET /api4/public/creator/{network}/{id}/v1   # 补充创作者画像
```

限速：`asyncio.Semaphore(5)`，缓存：SQLite TTL=24h

---

### L1 — 话题聚类

**Step 1：向量化 + 降维 + HDBSCAN**

```python
import hdbscan
from umap import UMAP

embeddings = qwen3_embed([p.text for p in posts])   # 批次50，高维向量
reduced    = UMAP(n_components=10, n_neighbors=15, min_dist=0.0).fit_transform(embeddings)

clusterer  = hdbscan.HDBSCAN(
    min_cluster_size=5,    # 至少5条帖子构成一个簇
    min_samples=3,         # 控制噪音敏感度
    metric='euclidean'
)
labels = clusterer.fit_predict(reduced)
# labels == -1 为噪音点，直接丢弃（约占 10-20%）
```

**Step 2：提取每个簇的语义信息（三层叠加）**

```python
# 层1：代表性文本（离质心最近的帖子）
from sklearn.metrics.pairwise import cosine_similarity

def get_representatives(embeddings, labels, texts, top_n=5):
    result = {}
    for cid in set(labels):
        if cid == -1: continue
        mask = (labels == cid)
        centroid = embeddings[mask].mean(axis=0)
        sims = cosine_similarity([centroid], embeddings[mask])[0]
        top_idx = sims.argsort()[-top_n:][::-1]
        result[cid] = [texts[i] for i in np.where(mask)[0][top_idx]]
    return result

# 层2：c-TF-IDF 关键词（统计区分度最高的词）
from sklearn.feature_extraction.text import CountVectorizer

def ctf_idf_keywords(texts, labels, top_n=10):
    cluster_docs = {}
    for text, label in zip(texts, labels):
        if label != -1:
            cluster_docs.setdefault(label, []).append(text)
    ids  = list(cluster_docs.keys())
    docs = [" ".join(cluster_docs[i]) for i in ids]
    vectorizer = CountVectorizer(stop_words="english", max_features=1000)
    tf   = vectorizer.fit_transform(docs).toarray().astype(float)
    vocab = vectorizer.get_feature_names_out()
    tf_norm = tf / tf.sum(axis=1, keepdims=True)
    idf = np.log(len(ids) / ((tf > 0).sum(axis=0) + 1))
    ctfidf = tf_norm * idf
    return {ids[i]: list(vocab[ctfidf[i].argsort()[-top_n:][::-1]])
            for i in range(len(ids))}

# 层3：Gemini 生成可读主题标签
# 输入：关键词 + 3条代表帖子 → 输出：10字以内主题名
prompt = """
关键词：{keywords}
代表帖子：{rep_texts}
用一句话（10字以内）描述这组帖子的核心话题。
"""
```

**每个簇最终得到：**
```python
{
    "cluster_id": 2,
    "keywords": ["yield", "apy", "defi", "protocol", "farm"],
    "representative_posts": ["I need to compare APY across...", ...],
    "label": "DeFi 收益比较需求"   # Gemini 生成
}
```

---

### L2 — 结构化 NLP（Gemini 并发 20）

每条帖子输出完整 NeedRecord，包含：
- 需求三元组 `(entity, feature, opinion)`
- `need_type`：显性/隐性
- `sentiment` + `emotion`（27类细粒度）+ `intensity`
- `evidence`（原文引用）
- `user_type`：Gemini 从帖子文本直接推断用户类型
  - 可选值：`trader` / `developer` / `investor` / `researcher` / `casual`
  - 依据：措辞习惯、关注点、技术深度
- `experience_level`：`beginner` / `intermediate` / `expert`

> **设计说明**：原设计通过 twikit 拉取完整推文历史，再用 KMeans 做向量分群。
> 由于数据源限制（仅有 LunarCrush top posts），改为 Gemini 逐帖推断 `user_type`，
> 准确度略低但在稀疏数据下更可靠。

---

### L3 — 聚合

**需求聚类（同 L1 三层流程）：**
```python
# Step1：pain_point 文本向量化 + UMAP + HDBSCAN
need_embeddings = qwen3_embed([r.pain_point for r in records])
reduced  = UMAP(n_components=8, min_dist=0.0).fit_transform(need_embeddings)
labels   = hdbscan.HDBSCAN(min_cluster_size=4, min_samples=2).fit_predict(reduced)

# Step2：每个需求簇的语义信息
# - 代表性 pain_point（离质心最近）→ 最典型痛点描述
# - c-TF-IDF 关键词 → 痛点的高区分度词汇
# - Gemini 命名 → 需求簇名称（如 "DeFi 收益聚合需求"）
```

**画像构建（定性）：**
```python
# 统计每个需求簇的 user_type 分布
# 目的：定性描述"这个需求是哪类人的"，服务于 trigger_description 措辞
# 不做矩阵交叉分析（400条数据每格太稀疏，统计意义有限）
from collections import Counter

def build_personas(need_records, cluster_id):
    cluster_records = [r for r in need_records if r.need_cluster_id == cluster_id]
    type_dist  = Counter(r.user_type for r in cluster_records)
    level_dist = Counter(r.experience_level for r in cluster_records)
    # 取占比 > 20% 的 user_type 作为主要画像
    total = len(cluster_records)
    return [
        UserPersona(
            segment_name=ut,
            post_count=count,
            pct=round(count/total*100),
            experience_level=level_dist.most_common(1)[0][0]
        )
        for ut, count in type_dist.most_common()
        if count/total > 0.2
    ]
```

**画像在 SkillSpec 中的作用：**
- `personas` → 告诉 skill-creator 目标用户是谁
- `dominant user_type` → 指导 `trigger_description` 使用对应措辞风格
- `experience_level` → 指导输出格式深度（beginner 要解释，expert 要简洁）

**情绪画像：** 聚合各簇情绪分布 + 强度均值

**现有方案：** Gemini 根据需求簇关键词 + 代表性文本识别市场现有工具

**机会评分（ODI）：**
```python
# Claude 从证据评估 importance / satisfaction
opp  = importance + max(importance - satisfaction, 0)
if avg_intensity > 0.7:  opp *= 1.5
if no_mature_solution:   opp *= 2.0
if trend == "rising":    opp *= 1.3
```

---

### L4 — SkillSpec 生成（Claude Sonnet）

输入：Top 10 PainCluster（按 final_score 排序）

Claude 任务：
1. 评估 `importance` / `satisfaction`（从证据推断）
2. 识别 `existing_solutions`（现有方案及其局限）
3. 设计 `trigger_description`（参考 skill-creator 规范：pushy 风格）
4. 生成 3 条 `example_prompts`（具体、有细节、像真人说话）
5. 填写完整 SkillSpec JSON

---

## 6. 输出格式

### skill_specs.json（→ skill-creator）

```json
[
  {
    "skill_name": "defi-yield-scout",
    "trigger_description": "Find and compare DeFi yield opportunities, calculate APY/APR across protocols, and surface the best risk-adjusted returns. Use this skill whenever the user asks about yield farming, liquidity mining, staking rewards, or says things like 'where should I put my USDC' or 'find me the best APY' or 'compare these two protocols'.",
    "expected_output_format": "协议对比表 + 风险评级 + 推荐配置",
    "example_prompts": [
      "i have 10k USDC sitting idle, what's the best yield right now with low risk",
      "compare aave vs compound vs morpho for ETH lending APY",
      "where can i get the best stablecoin yield without getting rugged"
    ],
    "suggested_approach": "调用 DeFi 数据 MCP → 聚合各协议 APY → 风险评级 → 排序推荐",

    "need_name": "DeFi 收益聚合与智能推荐",
    "need_type": "workflow_friction",
    "need_description": "用户需要在多个 DeFi 协议间比较收益率，目前需要手动打开多个网站，缺乏统一的风险调整后收益对比工具",

    "evidence": [
      "\"i spend 2 hours every week checking yields on 5 different sites, there has to be a better way\"",
      "\"lost money chasing high APY on a protocol i didnt research properly\"",
      "\"wish there was something like a defi aggregator that also tells me the risk\""
    ],

    "personas": [
      {
        "segment_name": "DeFi 散户投资者",
        "description": "持有稳定币或主流资产，寻求被动收益，非技术背景",
        "typical_behaviors": ["频繁切换协议", "关注 APY 变化", "风险敏感"],
        "experience_level": "intermediate",
        "post_count": 47
      }
    ],

    "emotion_profile": {
      "dominant_emotion": "frustrated",
      "emotion_distribution": {"frustrated": 0.45, "confused": 0.28, "requesting": 0.27},
      "avg_intensity": 0.73,
      "trend": "rising"
    },

    "existing_solutions": [
      {
        "name": "DeFiLlama",
        "coverage": "partial",
        "limitation": "只看数据，不提供个性化推荐和风险分析"
      },
      {
        "name": "Zapper / DeBank",
        "coverage": "partial",
        "limitation": "持仓追踪为主，收益对比和推荐不足"
      }
    ],

    "opportunity": {
      "importance": 8.2,
      "satisfaction": 3.5,
      "raw_score": 13.0,
      "intensity_weight": 1.5,
      "solution_weight": 1.0,
      "final_score": 19.5,
      "trend_boost": 1.3
    },

    "source_cluster_id": 2,
    "data_period": "2026-03-11 ~ 2026-03-25",
    "post_count": 63
  }
]
```

### skill_req_report.md（→ 人工审阅）

```markdown
# Skill 需求报告 2026-03-25

## Top 10 Skill 需求

| 排名 | Skill 名称 | 核心需求 | 机会分 | 趋势 | 画像 |
|-----|-----------|---------|--------|------|------|
| 1   | defi-yield-scout | DeFi 收益聚合推荐 | 19.5 | 📈 | DeFi 散户 |
...

## 方法论
- 数据来源：LunarCrush 加密/金融话题
- 帖子数量：XXX 条
- 分析模型：Gemini Flash (L1-L3) + Claude Sonnet (L4)
- 评分框架：ODI Opportunity Score (Ulwick)
```

---

## 7. 目录结构

```
skill_req/
├── pipeline.py
├── config.py
├── models.py
│
├── clients/
│   ├── lunarcrush.py
│   ├── openrouter.py        # Gemini + Qwen3 embedding
│   └── claude_client.py
│
├── stages/
│   ├── l0_collector.py
│   ├── l1_clusterer.py
│   ├── l2_analyzer.py
│   ├── l3_aggregator.py     # 聚合 + 画像 + 情绪 + 现有方案 + ODI
│   └── l4_spec_generator.py # SkillSpec 生成
│
├── prompts/
│   ├── l2_need_extract.txt
│   ├── l3_cluster_name.txt
│   ├── l3_existing_solutions.txt
│   └── l4_skill_spec.txt
│
├── storage/
│   └── cache.py
│
└── output/
    ├── skill_specs.json      # → skill-creator
    └── skill_req_report.md   # → 人工审阅
```

---

## 8. 成本估算（400 条帖子）

| 阶段 | 模型 | 估算 |
|------|------|------|
| L2 批量分析 | Gemini Flash | ~$0.04 |
| L3 聚类命名 + 现有方案 | Gemini Flash | ~$0.02 |
| L4 生成 10 个 SkillSpec | Claude Sonnet | ~$0.20 |
| Embedding（两次） | Qwen3 | ~$0.02 |
| **合计** | | **≈ $0.28** |

---

## 9. 运行

```bash
pip install httpx pydantic umap-learn hdbscan scikit-learn numpy anthropic

export LUNARCRUSH_KEY=xxx
export OPENROUTER_KEY=sk-or-v1-xxx
export CLAUDE_API_KEY=sk-ant-xxx

python pipeline.py

# 产出：
# output/skill_specs.json     → 交 skill-creator
# output/skill_req_report.md  → 人工审阅后决定提交哪些
```

---

## 10. 与 skill-creator 对接

```
每条 SkillSpec（人工筛选后）
    ↓
skill-creator 读取：
  skill_name              → 新建目录 + SKILL.md
  trigger_description     → frontmatter description
  example_prompts         → evals/evals.json 测试用例
  suggested_approach      → SKILL.md 正文实现指导
  need_description        → SKILL.md 背景/问题陈述
  evidence                → SKILL.md 用户原声引用
  personas                → SKILL.md 目标用户说明
  existing_solutions      → SKILL.md 竞品对比
  opportunity.final_score → 优先级参考
```
