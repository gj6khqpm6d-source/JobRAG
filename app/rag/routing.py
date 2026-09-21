from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


QueryIntent = Literal["evidence", "aggregate", "global_aggregate", "refusal"]


@dataclass(frozen=True)
class QueryRoute:
    intent: QueryIntent
    reason: str


_AGGREGATE_PATTERNS = (
    re.compile(r"\bhow many\b|\bmost common\b|\btop\b|\bpercentage\b|\bproportion\b", re.I),
    re.compile(r"\baverage\b|\bdistribution\b|\bmore common\b", re.I),
    re.compile(r"多少|数量|占比|比例|最常见|出现最多|统计|分布|平均|均值"),
)

_GLOBAL_AGGREGATE_PATTERNS = (
    re.compile(r"全库|整个知识库|全部岗位|所有岗位"),
    re.compile(r"\bentire (?:corpus|knowledge base)\b|\ball jobs\b|\bwhole database\b", re.I),
)

_REFUSAL_PATTERNS = (
    re.compile(r"天气|股价|股票|电影|菜谱|食谱|旅游攻略|写诗|数学题|汇率"),
    re.compile(r"\bweather\b|\bstock price\b|\brecipe\b|\bmovie review\b", re.I),
)


def classify_query(query: str) -> QueryRoute:
    """Choose a cheap deterministic route before embedding or LLM calls."""
    cleaned = query.strip()
    if any(pattern.search(cleaned) for pattern in _REFUSAL_PATTERNS):
        return QueryRoute("refusal", "问题超出岗位知识库范围")
    if any(pattern.search(cleaned) for pattern in _GLOBAL_AGGREGATE_PATTERNS):
        return QueryRoute("global_aggregate", "用户明确要求统计整个知识库")
    if any(pattern.search(cleaned) for pattern in _AGGREGATE_PATTERNS):
        return QueryRoute("aggregate", "问题要求分析最相关岗位中的统计或分布")
    return QueryRoute("evidence", "问题适合基于具体岗位证据回答")
