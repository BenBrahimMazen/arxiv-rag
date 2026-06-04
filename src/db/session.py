"""Database engine and session management."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.db.models import Base
from src.logging_conf import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    return create_engine(settings.postgres_url, pool_pre_ping=True, future=True)


@lru_cache(maxsize=1)
def _session_factory() -> sessionmaker[Session]:
    return sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)


def init_db() -> None:
    """Create all tables if they do not exist."""
    Base.metadata.create_all(get_engine())
    logger.info("Database tables ensured")


def get_session() -> Session:
    """Return a new session (caller is responsible for closing it)."""
    return _session_factory()()


@contextmanager
def session_scope() -> Iterator[Session]:
    """Transactional session context manager."""
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
