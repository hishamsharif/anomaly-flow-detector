"""Slice 3 tests — alerts endpoints (pagination, verdict submission, verdict-count)."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest
from httpx import AsyncClient

from api.db.models import Alert


def _make_alert(**kwargs) -> dict:
    defaults = dict(
        flow_id="0",
        consensus="KNOWN_ATTACK",
        severity="high",
        predicted_label=1,
        classifier_confidence=0.9,
        ae_score=0.1,
        ae_threshold=0.5,
        top_features_json=json.dumps([{"name": "feat_a", "contribution": 0.4}]),
        analyst_verdict=None,
        feedback_at=None,
        timestamp=datetime.now(timezone.utc),
    )
    defaults.update(kwargs)
    return defaults


async def _insert_alert(client: AsyncClient, **kwargs) -> Alert:
    """Helper: directly add an Alert row via the DB session in the test client."""
    from api.database import AsyncSessionLocal
    async with AsyncSessionLocal() as db:
        a = Alert(**_make_alert(**kwargs))
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


# ── Pagination ────────────────────────────────────────────────────────────────

async def test_alerts_empty(client: AsyncClient):
    resp = await client.get("/api/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"] == []
    assert data["total"] == 0
    assert data["page"] == 1


async def test_alerts_pagination(client: AsyncClient):
    # Insert 5 alerts
    for i in range(5):
        await _insert_alert(client, flow_id=str(i))

    resp = await client.get("/api/alerts?page=1&size=3")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["items"]) == 3
    assert data["total"] == 5
    assert data["page"] == 1

    resp2 = await client.get("/api/alerts?page=2&size=3")
    data2 = resp2.json()
    assert len(data2["items"]) == 2


async def test_alerts_filter_by_consensus(client: AsyncClient):
    await _insert_alert(client, consensus="SAFE", severity="low", ae_score=0.01)
    await _insert_alert(client, consensus="NOVEL_ATTACK", severity="critical", ae_score=0.9)
    await _insert_alert(client, consensus="KNOWN_ATTACK", severity="high", ae_score=0.1)

    resp = await client.get("/api/alerts?consensus=NOVEL_ATTACK")
    data = resp.json()
    assert data["total"] == 1
    assert data["items"][0]["consensus"] == "NOVEL_ATTACK"


async def test_alerts_filter_by_verdict_null(client: AsyncClient):
    await _insert_alert(client, flow_id="1", analyst_verdict=None)
    await _insert_alert(client, flow_id="2", analyst_verdict="confirmed")

    resp = await client.get("/api/alerts?verdict=null")
    data = resp.json()
    assert all(a["analyst_verdict"] is None for a in data["items"])


# ── Alert schema ──────────────────────────────────────────────────────────────

async def test_alert_response_shape(client: AsyncClient):
    await _insert_alert(client)
    resp = await client.get("/api/alerts")
    item = resp.json()["items"][0]
    required = {"id", "flow_id", "consensus", "severity", "ae_score", "ae_threshold",
                "top_features", "analyst_verdict", "timestamp"}
    assert required.issubset(item.keys())
    assert isinstance(item["top_features"], list)


# ── Verdict submission ────────────────────────────────────────────────────────

async def test_submit_verdict_200(client: AsyncClient):
    a = await _insert_alert(client)
    resp = await client.post(f"/api/alerts/{a.id}/verdict", json={"verdict": "confirmed"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["analyst_verdict"] == "confirmed"
    assert data["feedback_at"] is not None


async def test_submit_verdict_409_duplicate(client: AsyncClient):
    a = await _insert_alert(client)
    await client.post(f"/api/alerts/{a.id}/verdict", json={"verdict": "confirmed"})
    resp = await client.post(f"/api/alerts/{a.id}/verdict", json={"verdict": "uncertain"})
    assert resp.status_code == 409


async def test_submit_verdict_404_unknown(client: AsyncClient):
    resp = await client.post("/api/alerts/99999/verdict", json={"verdict": "confirmed"})
    assert resp.status_code == 404


async def test_submit_verdict_override_requires_label(client: AsyncClient):
    a = await _insert_alert(client)
    resp = await client.post(f"/api/alerts/{a.id}/verdict", json={"verdict": "override"})
    assert resp.status_code == 422


async def test_submit_verdict_override_with_label(client: AsyncClient):
    a = await _insert_alert(client)
    resp = await client.post(
        f"/api/alerts/{a.id}/verdict",
        json={"verdict": "override", "label": "BENIGN"},
    )
    assert resp.status_code == 200
    assert resp.json()["analyst_verdict"] == "override"


# ── Verdict count ─────────────────────────────────────────────────────────────

async def test_verdict_count_zero(client: AsyncClient):
    resp = await client.get("/api/alerts/verdict-count")
    assert resp.status_code == 200
    data = resp.json()
    assert data["eligible"] == 0
    assert data["unlocked"] is False


async def test_verdict_count_increments(client: AsyncClient):
    a = await _insert_alert(client)
    await client.post(f"/api/alerts/{a.id}/verdict", json={"verdict": "confirmed"})

    resp = await client.get("/api/alerts/verdict-count")
    data = resp.json()
    assert data["eligible"] == 1


async def test_verdict_count_only_eligible(client: AsyncClient):
    # Only confirmed + override count
    a1 = await _insert_alert(client, flow_id="1")
    a2 = await _insert_alert(client, flow_id="2")
    a3 = await _insert_alert(client, flow_id="3")

    await client.post(f"/api/alerts/{a1.id}/verdict", json={"verdict": "confirmed"})
    await client.post(f"/api/alerts/{a2.id}/verdict", json={"verdict": "uncertain"})
    await client.post(
        f"/api/alerts/{a3.id}/verdict",
        json={"verdict": "override", "label": "BENIGN"},
    )

    resp = await client.get("/api/alerts/verdict-count")
    # confirmed + override = 2 eligible; uncertain does not count
    assert resp.json()["eligible"] == 2
