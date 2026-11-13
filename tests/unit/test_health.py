from httpx import AsyncClient


async def test_health_returns_200(client: AsyncClient) -> None:
    response = await client.get("/health")
    assert response.status_code == 200


async def test_health_shape(client: AsyncClient) -> None:
    data = (await client.get("/health")).json()
    assert data["status"] == "ok"
    assert isinstance(data["model_loaded"], bool)
    assert isinstance(data["db_ok"], bool)


async def test_health_db_ok_true(client: AsyncClient) -> None:
    data = (await client.get("/health")).json()
    assert data["db_ok"] is True
