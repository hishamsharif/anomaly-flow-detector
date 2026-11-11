from httpx import AsyncClient

VALID_LEVELS = {"ok", "warn", "blocker"}


async def test_preflight_returns_all_checks(client: AsyncClient) -> None:
    response = await client.get("/api/preflight")
    assert response.status_code == 200
    data = response.json()
    assert "cuda" in data
    assert "docker" in data
    assert "images" in data


async def test_preflight_check_shape(client: AsyncClient) -> None:
    data = (await client.get("/api/preflight")).json()
    for key, check in data.items():
        assert "level" in check, f"{key}.level missing"
        assert "message" in check, f"{key}.message missing"
        assert check["level"] in VALID_LEVELS, f"{key}.level={check['level']!r} not in {VALID_LEVELS}"
