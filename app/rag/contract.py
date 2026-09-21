"""Machine-readable business contract for the JobRAG product behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT_PATH = PROJECT_ROOT / "data" / "jobrag" / "business_contract.json"


class BusinessContractError(ValueError):
    """Raised when the JobRAG business contract is missing or invalid."""


def _require(mapping: dict[str, Any], key: str, expected_type: type) -> Any:
    value = mapping.get(key)
    if not isinstance(value, expected_type):
        raise BusinessContractError(f"业务契约字段 {key!r} 必须是 {expected_type.__name__}")
    return value


def validate_business_contract(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        raise BusinessContractError("业务契约必须是 JSON 对象")

    version = _require(contract, "contract_version", str)
    if not version.strip():
        raise BusinessContractError("业务契约版本不能为空")
    default_mode = _require(contract, "default_mode", str)
    modes = _require(contract, "modes", dict)
    if default_mode not in modes:
        raise BusinessContractError("default_mode 必须出现在 modes 中")
    for mode_id, mode in modes.items():
        if not isinstance(mode_id, str) or not isinstance(mode, dict):
            raise BusinessContractError("modes 必须是 mode_id 到对象的映射")
        status = _require(mode, "status", str)
        if status not in {"active", "planned", "deprecated"}:
            raise BusinessContractError(f"未知的业务模式状态：{status}")

    chunking = _require(contract, "chunking_contract", dict)
    if chunking.get("strategy") != "section_paragraph_list":
        raise BusinessContractError("chunking_contract.strategy 必须为 section_paragraph_list")
    for key in ("relevant_sections", "ignored_sections", "context_fields"):
        values = _require(chunking, key, list)
        if not values or any(not isinstance(value, str) or not value.strip() for value in values):
            raise BusinessContractError(f"chunking_contract.{key} 不能为空")
    for key in ("max_chars", "overlap_chars"):
        value = chunking.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise BusinessContractError(f"chunking_contract.{key} 必须为正整数")
    if chunking["overlap_chars"] >= chunking["max_chars"]:
        raise BusinessContractError("chunking overlap_chars 必须小于 max_chars")

    retrieval = _require(contract, "retrieval_contract", dict)
    for key in ("candidate_jobs", "top_relevant_jobs"):
        limits = _require(retrieval, key, dict)
        values = [limits.get(name) for name in ("minimum", "target", "maximum")]
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise BusinessContractError(f"{key} 必须包含整数 minimum、target、maximum")
        minimum, target, maximum = values
        if not 1 <= minimum <= target <= maximum:
            raise BusinessContractError(f"{key} 的数量范围无效")
    if retrieval.get("unit") != "job":
        raise BusinessContractError("retrieval_contract.unit 必须为 job")
    if retrieval.get("deduplicate_by") != "job_id":
        raise BusinessContractError("检索结果必须按 job_id 去重")

    answer = _require(contract, "answer_contract", dict)
    sections = _require(answer, "required_sections", list)
    if not sections or any(not isinstance(section, str) or not section.strip() for section in sections):
        raise BusinessContractError("answer_contract.required_sections 不能为空")
    citation = _require(answer, "citation", dict)
    if citation.get("required") is not True:
        raise BusinessContractError("岗位分析回答必须保留引用")
    refusal = _require(answer, "refusal", dict)
    triggers = _require(refusal, "triggers", list)
    if not triggers:
        raise BusinessContractError("拒答触发条件不能为空")

    quality = _require(contract, "quality_contract", dict)
    for key in ("retrieval_metrics", "answer_metrics", "release_requirements"):
        values = _require(quality, key, list)
        if not values:
            raise BusinessContractError(f"quality_contract.{key} 不能为空")

    workflow = _require(contract, "delivery_workflow", dict)
    if workflow.get("phase_confirmation_required") is not True:
        raise BusinessContractError("每个开发阶段必须经过用户确认")
    if not isinstance(workflow.get("sequence"), list) or not workflow["sequence"]:
        raise BusinessContractError("delivery_workflow.sequence 不能为空")
    return contract


def load_business_contract(path: str | Path = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    contract_path = Path(path)
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise BusinessContractError(f"找不到业务契约文件：{contract_path}") from exc
    except json.JSONDecodeError as exc:
        raise BusinessContractError(f"业务契约不是有效 JSON：{contract_path}") from exc
    return validate_business_contract(payload)
