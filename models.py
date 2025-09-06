from sqlalchemy import create_engine, Column, String, Integer, DateTime, Boolean, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

DATABASE_URL = "sqlite:///./test.db"

engine = create_engine(DATABASE_URL, echo=False)
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password = Column(String, nullable=False)
    status = Column(String, default='pending')  # pending, active, paused
    payment_active = Column(Integer, default=0)  # 0 = not paid, 1 = paid
    payment_start = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    payment_tier = Column(Integer, default=0)

Base.metadata.create_all(engine)

def ensure_schema():
    with engine.connect() as conn:
        cols = conn.execute(text("PRAGMA table_info(users)")).fetchall()
        col_names = {c[1] for c in cols}
        if "payment_tier" not in col_names:
            conn.execute(text("ALTER TABLE users ADD COLUMN payment_tier INTEGER DEFAULT 0"))
        conn.commit()

ensure_schema()