# app/db.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Use a relative path so it creates support.db alongside your project root
DATABASE_URL = "sqlite:///support.db"  # or "sqlite:///./support.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},  # required for SQLite on uvicorn
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Optional: no-op init for compatibility; expand if you create tables later
#def init_db() -> None:
#    # from app import models  # if you later define ORM models
#    # Base.metadata.create_all(bind=engine)
#    pass


def init_db() -> None:
    from app import models  # ensure models are imported before create_all
    Base.metadata.create_all(bind=engine)