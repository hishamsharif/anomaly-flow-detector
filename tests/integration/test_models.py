from httpx import AsyncClient


async def test_available_types_has_expected_entries(client: AsyncClient) -> None:
    r = await client.get("/api/models/available-types")
    assert r.status_code == 200
    data = r.json()
    clf_ids = [c["id"] for c in data["classifiers"]]
    det_ids = [d["id"] for d in data["detectors"]]
    assert "xgboost" in clf_ids
    assert "mlp_ae" in det_ids


async def test_active_model_is_null_initially(client: AsyncClient) -> None:
    r = await client.get("/api/models/active")
    assert r.status_code == 200
    assert r.json() is None


async def test_model_list_is_empty_initially(client: AsyncClient) -> None:
    r = await client.get("/api/models")
    assert r.status_code == 200
    data = r.json()
    assert data["items"] == []
    assert data["total"] == 0
