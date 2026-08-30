"""
LEGALMET AI — Database Engine & Session
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from backend.config import settings
from backend.database.models import Base

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},  # Required for SQLite with FastAPI
    echo=False,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Create all tables. Called on application startup."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency that provides a DB session per request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
