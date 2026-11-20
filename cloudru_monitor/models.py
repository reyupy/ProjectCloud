"""Shared, validated contracts. Raw events stay in OpenSearch."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from .version import ALGORITHM_VERSION


class TimeWindow(BaseModel):
    start: datetime
    end: datetime

    @field_validator("start", "end")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must include a timezone")
        return value

    @model_validator(mode="after")
    def ordered(self) -> "TimeWindow":
        if self.start >= self.end:
            raise ValueError("start must be earlier than end")
        return self


class LogEvent(BaseModel):
    evidence_id: str
    timestamp: datetime
    level: str
    message: str
    service: str = "unknown"
    host: str | None = None
    request_id: str | None = None
    request_id_kind: Literal["global", "local"] | None = None
    global_request_id: str | None = None
    local_request_id: str | None = None
    http_method: str | None = None
    http_uri: str | None = None
    http_status: int | None = None
    duration_ms: float | None = None

    @field_validator("timestamp")
    @classmethod
    def aware(cls, value: datetime) -> datetime:
        return TimeWindow.aware(value)


class FetchResult(BaseModel):
    hits: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    truncated: bool = False
    warnings: list[str] = Field(default_factory=list)


class ErrorGroup(BaseModel):
    fingerprint: str
    service: str
    template: str
    count: int
    first_seen: datetime
    last_seen: datetime
    representative: LogEvent
    sample_evidence_ids: list[str] = Field(default_factory=list)
    request_count: int = 0
    missing_request_id_count: int = 0


class ChainReport(BaseModel):
    request_id: str | None
    status: Literal["retrieved", "missing_request_id", "missing_local_service", "empty", "truncated", "unavailable"]
    events: list[LogEvent] = Field(default_factory=list)
    total: int = 0
    fetched_candidates: int = 0
    total_candidates: int = 0
    warnings: list[str] = Field(default_factory=list)


class GroupReport(BaseModel):
    group: ErrorGroup
    chain: ChainReport
    analysis_status: Literal["not_requested"] = "not_requested"
    llm_context: dict[str, Any] = Field(default_factory=dict)


class AnalysisReport(BaseModel):
    schema_version: str = "1.2"
    algorithm_version: str = ALGORITHM_VERSION
    run_id: str
    generated_at: datetime
    source: str
    window: TimeWindow
    latest_source_timestamp: datetime | None = None
    freshness: Literal["fresh", "stale", "empty", "unknown"] = "unknown"
    latest_window_timestamp: datetime | None = None
    window_coverage: Literal["present", "empty", "unknown"] = "unknown"
    watch_paused: bool = False
    fetched_candidates: int = 0
    total_candidates: int = 0
    fetched_errors: int = 0
    new_errors: int = 0
    skipped_errors: int = 0
    invalid_errors: int = 0
    total_groups: int = 0
    groups: list[GroupReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    truncated: bool = False
    error_scan_limited: bool = False
    watch_partial_chains: bool = False

    @property
    def complete(self) -> bool:
        return not (self.watch_paused or self.truncated or self.invalid_errors) and all(
            entry.chain.status in {"retrieved", "missing_request_id", "missing_local_service"} for entry in self.groups
        )
