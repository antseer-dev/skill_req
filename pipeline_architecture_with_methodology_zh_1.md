# 社交媒体用户需求洞察分析 Pipeline 架构图（含方法论）

---

## L0 — 数据采集层

**输入：** 推文 + 回复 + 用户 Bio

| 组件 | 说明 |
|------|------|
| twikit ⭐3.8k | 实时关键词 / 话题采集 |
| twscrape ⭐1.6k | 历史数据批量回填 |
| MongoDB | 非结构化原始存储 |

**方法论：**
- ![GitHub](green) `d60/twikit`
- ![GitHub](green) `vladkens/twscrape`
- ![论文](blue) Pak & Paroubek 2010 — Twitter 舆情采集方法论

**核心输出：** `tweet_id`, `text`, `user_id`, `user_bio`, `reply_to`, `likes`, `retweets`, `timestamp`, `lang`

---

↓

---

## L1 — 内容理解 + 用户分群

**步骤：** Step 1 + Step 2

| 组件 | 说明 |
|------|------|
| BERTopic ⭐7.3k | 主题建模 + 种子引导聚类 |
| spaCy ⭐31k | NER 实体识别 + 预处理 |
| KMeans | 用户 topic 分布聚合分群 |

**方法论：**
- ![GitHub](green) `MaartenGr/BERTopic`
- ![论文](blue) Grootendorst 2022 — BERTopic: Neural topic modeling with c-TF-IDF
- ![工程](yellow) Guided Topic Modeling — `seed_topic_list` 引导
- ![工程](yellow) Dynamic Topic Modeling — `topics_over_time()`

**方法细节：** SBERT embedding → UMAP 降维 → HDBSCAN 聚类 → c-TF-IDF 主题词 → LLM 生成可读标签。用户分群：per-user topic distribution 均值 → KMeans 5-cluster

---

↓

---

## L2 — 深度分析（三条并行管道）

**步骤：** Step 3 + Step 4 + Step 5

### 管道 A：需求抽取

- Claude API 结构化 prompt
- `(entity, feature, opinion)` 三元组
- 区分显性 / 隐性需求
- ![论文](blue) OOMB, Heo et al. EMNLP'25
- ![GitHub](green) `ryang1119/OOMB`（仅取方法论）

### 管道 B：痛点归因

- PyABSA ⭐1.1k — 方面级情感
- Aspect 提取 + 极性 + 置信度
- 根因标签（root cause tagging）
- ![GitHub](green) `yangheng95/PyABSA`
- ![论文](blue) Yang et al. CIKM'23
- ![论文](blue) InstructABSA, Scaria NAACL'24

### 管道 C：情绪分析

- VADER — 粗粒度正/负/中性
- GoEmotions — 27 类细粒度情绪
- 综合情绪强度评分
- ![GitHub](green) HuggingFace transformers ⭐142k
- ![论文](blue) GoEmotions, Demszky ACL'20
- ![工程](yellow) VADER + GoEmotions 双层融合

**核心输出：** 每条推文的 `(entity, feature, opinion, need_type, sentiment, emotion, intensity, evidence)` 结构化记录

---

↓

---

## L3 — 聚合综合 + 聚类

**步骤：** Step 6

| 组件 | 说明 |
|------|------|
| 需求层级聚类 | `reduce_topics()` 合并 → Top N |
| 画像 x 需求矩阵 | 用户群 x 需求交叉分析 |
| 时间序列趋势 | `topics_over_time()` 热度追踪 |

**方法论：**
- ![GitHub](green) BERTopic — `visualize_hierarchy()`
- ![论文](blue) Grootendorst 2022 — Hierarchical Topic Reduction
- ![论文](blue) Grootendorst 2022 — Dynamic Topic Modeling

**核心输出：** Top 10 需求聚类 + 每个聚类的用户画像分布 + 热度趋势方向（升温/稳定/降温）

---

↓

---

## L4 — 机会映射 + 产品建议

**步骤：** Step 7

| 组件 | 说明 |
|------|------|
| Opportunity Score | `Imp + max(Imp - Sat, 0) x W` |
| Top 10 需求排序 | 得分 + 证据链 + 画像 |
| 功能级产品建议 | MCP tool / UGC skill 方案 |

**方法论：**
- ![论文](blue) Joung & Kim 2017 — Product Opportunity Mining (Int. J. Info. Mgmt.)
- ![论文](blue) Ulwick ODI — Outcome-Driven Innovation 评分框架
- ![工程](yellow) 自研 50 行 Python scoring pipeline

**公式：**
```
Opportunity = Importance + max(Importance - Satisfaction, 0)
```
叠加情绪强度权重（强 ×1.5）和未满足权重（无方案 ×2.0）

---

↓

---

## 结构化输出报告

**内容：** 需求表 + 证据 + 画像 + 情绪 + 现有方案 + 产品机会

**落地方向：**
- MCP 工具优先级
- GEO 内容选题
- UGC 技能模板

---

## 方法论图例

| 标记 | 含义 |
|------|------|
| 蓝底 | 论文方法论 |
| 绿底 | GitHub 开源项目 |
| 黄底 | 工程实现 / 自研 |

**核心依赖：**
```
pip install bertopic pyabsa transformers twikit
```
其余用 Claude API + 50 行自研代码覆盖
