from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from api.db.models import Alert


pytestmark = pytest.mark.asyncio


def _alert_dict(**kwargs) -> dict:
    defaults = dict(
        flow_id="0",
        consensus="KNOWN_ATTACK",
        severity="high",
        ae_score=0.5,
        ae_threshold=0.9,
        top_features_json=json.dumps([]),
        analyst_verdict=None,
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return defaults


async def _insert(client: AsyncClient, **kwargs) -> Alert:
    from api.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        a = Alert(**_alert_dict(**kwargs))
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


# ── Tests ─────────────────────────────────────────────────────────────────────

async def test_dashboard_summary_empty(client: AsyncClient):
    res = await client.get("/api/dashboard/summary")
    assert res.status_code == 200
    body = res.json()
    assert body["total_24h"] == 0
    assert body["safe_24h"] == 0
    assert body["known_attack_24h"] == 0
    assert body["novel_attack_24h"] == 0
    assert body["pending_verdict_24h"] == 0
    assert body["by_severity"]["low"] == 0
    assert body["retrain_unlocked"] is False
    assert "verdict_threshold" in body


async def test_dashboard_summary_counts(client: AsyncClient):
    await _insert(client, consensus="SAFE", severity="low")
    await _insert(client, consensus="SAFE", severity="low")
    await _insert(client, consensus="KNOWN_ATTACK", severity="high", analyst_verdict="confirmed")
    await _insert(client, consensus="NOVEL_ATTACK", severity="critical", analyst_verdict="override")

    res = await client.get("/api/dashboard/summary")
    assert res.status_code == 200
    body = res.json()

    assert body["total_24h"] == 4
    assert body["safe_24h"] == 2
    assert body["known_attack_24h"] == 1
    assert body["novel_attack_24h"] == 1
    # 2 of 4 have no analyst_verdict → pending
    assert body["pending_verdict_24h"] == 2
    assert body["by_severity"]["low"] == 2
    assert body["by_severity"]["high"] == 1
    assert body["by_severity"]["critical"] == 1
    assert body["verdict_eligible"] == 2


async def test_dashboard_retrain_unlocked(client: AsyncClient):
    # Get the configured threshold first
    res0 = await client.get("/api/dashboard/summary")
    threshold = res0.json()["verdict_threshold"]

    # Insert exactly `threshold` confirmed alerts
    for _ in range(threshold):
        await _insert(client, analyst_verdict="confirmed")

    res = await client.get("/api/dashboard/summary")
    assert res.status_code == 200
    assert res.json()["retrain_unlocked"] is True
