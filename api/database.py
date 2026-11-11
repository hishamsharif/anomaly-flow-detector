from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

from api.config import get_settings


def _make_engine():
    settings = get_settings()
    url = settings.database_url
    if url.startswith("sqlite"):
        db_path = url.split("///", 1)[-1]
        if db_path and db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(url, echo=False)


engine = _make_engine()
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
