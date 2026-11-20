"""Transactional, source-scoped progress and optional short-lived report cache."""
import json
import sqlite3
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .models import AnalysisReport


class StateStore:
    def __init__(self, path: str | Path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path), timeout=30)
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS processed (
                source TEXT NOT NULL, evidence_id TEXT NOT NULL,
                PRIMARY KEY(source, evidence_id));
            CREATE TABLE IF NOT EXISTS checkpoints (
                source TEXT PRIMARY KEY, timestamp TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS pending_scans (
                source TEXT PRIMARY KEY, start TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY, source TEXT NOT NULL,
                generated_at TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY, created_at REAL NOT NULL, report TEXT NOT NULL);
        """)

    @contextmanager
    def serialize_run(self):
        """Hold the writer reservation from the seen check through delivery/commit."""
        if self.connection.in_transaction:
            yield
        else:
            with self.connection:
                self.connection.execute("BEGIN IMMEDIATE")
                yield

    def _write_transaction(self):
        return nullcontext() if self.connection.in_transaction else self.connection

    def seen(self, source: str, ids: list[str], *, watch: bool = False) -> set[str]:
        if watch:
            source = "watch:" + source
        result = set()
        for offset in range(0, len(ids), 400):
            chunk = ids[offset:offset + 400]
            marks = ",".join("?" for _ in chunk)
            result.update(row[0] for row in self.connection.execute(
                f"SELECT evidence_id FROM processed WHERE source=? AND evidence_id IN ({marks})",
                [source, *chunk]))
        return result

    def get_checkpoint(self, source: str) -> datetime | None:
        row = self.connection.execute("SELECT timestamp FROM checkpoints WHERE source=?", (source,)).fetchone()
        return datetime.fromisoformat(row[0]) if row else None

    def reserve_start(self, source: str, start: datetime) -> datetime:
        """Keep the first cold-start lower bound until a complete scan commits."""
        if start.tzinfo is None or start.utcoffset() is None:
            raise ValueError("pending scan start must include timezone")
        with self._write_transaction():
            self.connection.execute("INSERT OR IGNORE INTO pending_scans VALUES (?,?)",
                                    (source, start.isoformat()))
            row = self.connection.execute("SELECT start FROM pending_scans WHERE source=?", (source,)).fetchone()
        return datetime.fromisoformat(row[0])

    def commit_run(self, source: str, report: AnalysisReport, evidence_ids: list[str], checkpoint: datetime | None) -> None:
        if checkpoint is not None and (checkpoint.tzinfo is None or checkpoint.utcoffset() is None):
            raise ValueError("checkpoint must include timezone")
        metadata = report.model_dump(mode="json", exclude={"groups"})
        metadata["group_count"] = len(report.groups)
        with self._write_transaction():
            self.connection.execute("INSERT INTO runs VALUES (?,?,?,?)", (
                report.run_id, source, report.generated_at.isoformat(), json.dumps(metadata)))
            if report.complete or (checkpoint is not None and report.watch_partial_chains):
                if report.watch_partial_chains:
                    scopes = ("watch:" + source,)
                else:
                    scopes = (source, "watch:" + source) if checkpoint is not None else (source,)
                self.connection.executemany("INSERT OR IGNORE INTO processed VALUES (?,?)",
                                            ((scope, item) for scope in scopes for item in evidence_ids))
                previous = self.get_checkpoint(source)
                if checkpoint is not None and (previous is None or checkpoint > previous):
                    self.connection.execute("INSERT OR REPLACE INTO checkpoints VALUES (?,?)",
                                            (source, checkpoint.isoformat()))
                    self.connection.execute("DELETE FROM pending_scans WHERE source=?", (source,))

    def cached_report(self, cache_key: str, ttl_seconds: int) -> AnalysisReport | None:
        if ttl_seconds <= 0:
            return None
        row = self.connection.execute("SELECT created_at,report FROM cache WHERE cache_key=?", (cache_key,)).fetchone()
        now = datetime.now(timezone.utc).timestamp()
        if not row or not 0 <= now - row[0] < ttl_seconds:
            return None
        return AnalysisReport.model_validate_json(row[1])

    def cache_report(self, cache_key: str, report: AnalysisReport) -> None:
        with self._write_transaction():
            self.connection.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)", (
                cache_key, datetime.now(timezone.utc).timestamp(), report.model_dump_json()))
            self.connection.execute("DELETE FROM cache WHERE created_at < ?",
                                    (datetime.now(timezone.utc).timestamp() - 86400,))

    def close(self) -> None:
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
