import pandas as pd
from fastapi.testclient import TestClient

from app import server


client = TestClient(server.app)


def test_health_and_meta():
    assert client.get("/api/health").json() == {"status": "ok"}
    meta = client.get("/api/meta").json()
    assert "indeed" in meta["sites"]
    assert "singapore" in meta["countries"]


def test_business_contract_endpoint_exposes_active_contract():
    response = client.get("/api/kb/contract")

    assert response.status_code == 200
    contract = response.json()
    assert contract["default_mode"] == "role_analysis"
    assert contract["retrieval_contract"]["unit"] == "job"


def test_validation_requires_search_input():
    response = client.post("/api/search", json={"sites": ["indeed"]})
    assert response.status_code == 422


def test_linkedin_full_description_is_enabled_by_default():
    request = server.SearchRequest(sites=["linkedin"], search_term="AI Engineer")
    assert request.linkedin_fetch_description is True


def test_dataframe_records_normalizes_values():
    frame = pd.DataFrame([{"title": "Engineer", "salary": float("nan"), "date": pd.Timestamp("2026-01-02")}])
    assert server.dataframe_records(frame) == [{"title": "Engineer", "salary": None, "date": "2026-01-02T00:00:00"}]
