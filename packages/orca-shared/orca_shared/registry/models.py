from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncGenerator, Optional

from sqlalchemy import BigInteger, Boolean, Float, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import DateTime


class Base(DeclarativeBase):
    pass


class Embedding(Base):
    __tablename__ = "embeddings"

    embedding_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tasks.task_id"), nullable=True
    )
    embedding_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    embedding_vector: Mapped[Optional[list[float]]] = mapped_column(ARRAY(Float), nullable=True)
    dimension: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    task: Mapped[Optional["Task"]] = relationship(
        "Task", foreign_keys=[task_id], back_populates="embeddings"
    )


class Task(Base):
    __tablename__ = "tasks"

    task_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    domain: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    task_type: Mapped[str] = mapped_column(String(50), nullable=False)
    n_samples: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_features: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_classes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    dataset_uri: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    task_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    embedding_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("embeddings.embedding_id", use_alter=True, name="fk_tasks_embedding_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )

    current_embedding: Mapped[Optional[Embedding]] = relationship(
        "Embedding", foreign_keys=[embedding_id]
    )
    embeddings: Mapped[list[Embedding]] = relationship(
        "Embedding", foreign_keys=[Embedding.task_id], back_populates="task"
    )
    experiments: Mapped[list["Experiment"]] = relationship(
        "Experiment", back_populates="task"
    )
    source_mappings: Mapped[list["TransferMapping"]] = relationship(
        "TransferMapping",
        foreign_keys="[TransferMapping.source_task_id]",
        back_populates="source_task",
    )
    target_mappings: Mapped[list["TransferMapping"]] = relationship(
        "TransferMapping",
        foreign_keys="[TransferMapping.target_task_id]",
        back_populates="target_task",
    )


class Model(Base):
    __tablename__ = "models"

    model_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    architecture: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    config: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    parameter_count: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    flops: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    experiments: Mapped[list["Experiment"]] = relationship("Experiment", back_populates="model")


class Experiment(Base):
    __tablename__ = "experiments"

    experiment_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tasks.task_id"), nullable=True
    )
    model_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("models.model_id"), nullable=True
    )
    training_config: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    mlflow_run_id: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    created_by: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metrics: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONB, nullable=True)

    task: Mapped[Optional[Task]] = relationship("Task", back_populates="experiments")
    model: Mapped[Optional[Model]] = relationship("Model", back_populates="experiments")
    performances: Mapped[list["Performance"]] = relationship(
        "Performance", back_populates="experiment"
    )


class Performance(Base):
    __tablename__ = "performances"

    performance_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    experiment_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("experiments.experiment_id"), nullable=True
    )
    metric_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    metric_value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    epoch: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    is_final: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    perf_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    experiment: Mapped[Optional[Experiment]] = relationship(
        "Experiment", back_populates="performances"
    )


class TransferMapping(Base):
    __tablename__ = "transfer_mappings"

    mapping_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    source_task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tasks.task_id"), nullable=True
    )
    target_task_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tasks.task_id"), nullable=True
    )
    transfer_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    transfer_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    mapping_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(
        "metadata", JSONB, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    source_task: Mapped[Optional[Task]] = relationship(
        "Task", foreign_keys=[source_task_id], back_populates="source_mappings"
    )
    target_task: Mapped[Optional[Task]] = relationship(
        "Task", foreign_keys=[target_task_id], back_populates="target_mappings"
    )


class SearchSpace(Base):
    __tablename__ = "search_spaces"

    search_space_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    definition: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    parent_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("search_spaces.search_space_id"),
        nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    parent: Mapped[Optional["SearchSpace"]] = relationship(
        "SearchSpace", remote_side=[search_space_id], back_populates="children"
    )
    children: Mapped[list["SearchSpace"]] = relationship(
        "SearchSpace", back_populates="parent"
    )


def get_engine(database_url: str) -> AsyncEngine:
    """Return an AsyncEngine for the given asyncpg database URL."""
    if "+" not in database_url.split("://")[0]:
        if database_url.startswith("postgres://"):
            database_url = "postgresql+asyncpg://" + database_url[len("postgres://"):]
        elif database_url.startswith("postgresql://"):
            database_url = "postgresql+asyncpg://" + database_url[len("postgresql://"):]
    return create_async_engine(database_url, echo=False, pool_pre_ping=True)


@asynccontextmanager
async def get_session(engine) -> AsyncGenerator[AsyncSession, None]:
    """Async context manager that yields a transactional AsyncSession."""
    factory = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)
    async with factory() as session:
        async with session.begin():
            yield session
