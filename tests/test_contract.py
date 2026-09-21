from pathlib import Path

import pytest

from app.rag.contract import DEFAULT_CONTRACT_PATH, BusinessContractError, load_business_contract, validate_business_contract


def test_business_contract_has_role_analysis_as_default():
    contract = load_business_contract()

    assert contract["default_mode"] == "role_analysis"
    assert contract["modes"]["role_analysis"]["status"] == "active"
    assert contract["modes"]["job_detail"]["status"] == "planned"
    assert contract["chunking_contract"]["strategy"] == "section_paragraph_list"
    assert "company" in contract["chunking_contract"]["ignored_sections"]
    assert contract["retrieval_contract"]["deduplicate_by"] == "job_id"
    assert contract["answer_contract"]["citation"]["required"] is True


def test_business_contract_requires_job_level_retrieval_and_confirmed_phases():
    contract = load_business_contract()

    assert contract["retrieval_contract"]["unit"] == "job"
    assert contract["retrieval_contract"]["candidate_jobs"]["target"] == 30
    assert contract["retrieval_contract"]["top_relevant_jobs"]["maximum"] == 12
    assert contract["retrieval_contract"]["skill_frequency_denominator"] == "top_relevant_jobs"
    assert contract["delivery_workflow"]["phase_confirmation_required"] is True


def test_business_contract_rejects_invalid_limits():
    contract = load_business_contract()
    contract["retrieval_contract"]["candidate_jobs"]["target"] = 0

    with pytest.raises(BusinessContractError):
        validate_business_contract(contract)


def test_business_contract_path_is_inside_project_data_directory():
    path = Path(DEFAULT_CONTRACT_PATH)

    assert path.name == "business_contract.json"
    assert path.parent.name == "jobrag"
