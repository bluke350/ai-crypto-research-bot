from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json
import os
import uuid

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Float,
    Integer,
    Text,
    DateTime,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"
    run_id = Column(String, primary_key=True)
    cfg = Column(Text)
    created_at = Column(DateTime, server_default=func.now())


class Metric(Base):
    __tablename__ = "metrics"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    name = Column(String)
    value = Column(Float)


class Artifact(Base):
    __tablename__ = "artifacts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, index=True)
    path = Column(String)
    kind = Column(String, default="report")


@dataclass
class RunLogger:
    run_id: str
    cfg: Optional[Dict[str, Any]] = None
    db_url: Optional[str] = None

    def __enter__(self):
        self.db_url = self.db_url or os.getenv("EXPERIMENT_DB_URL") or "sqlite:///experiments/registry.db"
        # create engine and session
        self.engine = create_engine(self.db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()

        # upsert run (use Session.get which is the modern API)
        existing = self.session.get(Run, self.run_id)
        if existing is None:
            new_run = Run(run_id=self.run_id, cfg=json.dumps(self.cfg or {}))
            self.session.add(new_run)
            self.session.commit()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.session.close()
        if hasattr(self, "engine"):
            self.engine.dispose()

    def log_metrics(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            m = Metric(run_id=self.run_id, name=k, value=float(v))
            self.session.add(m)
        self.session.commit()

    def log_artifact(self, path: str, kind: str = "report"):
        a = Artifact(run_id=self.run_id, path=path, kind=kind)
        self.session.add(a)
        self.session.commit()
