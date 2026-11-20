from datetime import datetime, timedelta, timezone

import pytest

from cloudru_monitor.models import AnalysisReport, TimeWindow
from cloudru_monitor.storage import StateStore


def report(run_id="one", **kwargs):
    now = datetime.now(timezone.utc)
    return AnalysisReport(run_id=run_id, generated_at=now, source="source",
                          window=TimeWindow(start=now - timedelta(minutes=5), end=now), **kwargs)


def test_source_scope_and_checkpoint_monotonic():
    with StateStore(":memory:") as store:
        first = report()
        store.commit_run("a", first, ["id"], first.window.end)
        assert store.seen("a", ["id"]) == {"id"}
        assert not store.seen("b", ["id"])
        store.commit_run("a", report("two"), [], first.window.start)
        assert store.get_checkpoint("a") == first.window.end
        assert store.get_checkpoint("b") is None


def test_partial_run_is_audit_only_and_transaction_atomic():
    with StateStore(":memory:") as store:
        partial = report(truncated=True)
        store.commit_run("a", partial, ["id"], partial.window.end)
        assert not store.seen("a", ["id"])
        assert store.get_checkpoint("a") is None
        with pytest.raises(Exception):
            store.commit_run("a", report(), ["second"], partial.window.end)
        assert not store.seen("a", ["second"])


def test_cache_ttl_and_empty_seen():
    with StateStore(":memory:") as store:
        value = report()
        store.cache_report("key", value)
        assert store.cached_report("key", 30) == value
        assert store.cached_report("key", 0) is None
        assert store.cached_report("other", 30) is None
        assert store.seen("a", []) == set()
        store.connection.execute("UPDATE cache SET created_at=0")
        assert store.cached_report("key", 30) is None


def test_pending_scan_survives_partial_and_restart_then_clears_on_commit(tmp_path):
    path = tmp_path / "state.sqlite3"
    first = report(truncated=True)
    start = first.window.start
    with StateStore(path) as store:
        assert store.reserve_start("a", start) == start
        store.commit_run("a", first, ["id"], first.window.end)
    with StateStore(path) as store:
        assert store.reserve_start("a", start + timedelta(hours=1)) == start
        assert store.get_checkpoint("a") is None
        assert not store.seen("a", ["id"])
        # Sources remain independent; reservation itself never consumes events.
        assert store.reserve_start("b", start + timedelta(hours=1)) == start + timedelta(hours=1)
        store.commit_run("a", report("complete"), ["id"], first.window.end)
        assert store.connection.execute("SELECT start FROM pending_scans WHERE source='a'").fetchone() is None
        assert store.connection.execute("SELECT start FROM pending_scans WHERE source='b'").fetchone()
