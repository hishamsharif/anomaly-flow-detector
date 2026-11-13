from pathlib import Path

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

import api.http.dao.db_conn as _session_module
from db.base import Base
from api.http.dao.db_conn import get_db
from api.http.main import app

_TEST_DB_URL = "sqlite+aiosqlite:///:memory:"


@pytest_asyncio.fixture
async def client() -> AsyncClient:
    test_engine = create_async_engine(_TEST_DB_URL)
    test_sessions = async_sessionmaker(test_engine, expire_on_commit=False)

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Patch module-level singletons so lifespan (AsyncSessionLocal) and
    # routers (get_db override) both hit the in-memory test DB.
    orig_engine = _session_module.engine
    orig_sessions = _session_module.AsyncSessionLocal
    _session_module.engine = test_engine
    _session_module.AsyncSessionLocal = test_sessions

    async def _override_get_db():
        async with test_sessions() as session:
            yield session

    app.dependency_overrides[get_db] = _override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()
    _session_module.engine = orig_engine
    _session_module.AsyncSessionLocal = orig_sessions
    await test_engine.dispose()


@pytest.fixture
def tmp_model_dir(tmp_path: Path) -> Path:
    d = tmp_path / "models"
    d.mkdir()
    return d
