from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from cloudru_monitor.models import FetchResult, TimeWindow
from cloudru_monitor.pipeline import Monitor
from cloudru_monitor.storage import StateStore


STAMP = datetime(2026, 11, 15, 6, tzinfo=timezone.utc)
WINDOW = TimeWindow(start=STAMP, end=STAMP + timedelta(hours=1))
REQUEST = "req-11111111-1111-4111-8111-111111111111"


def hit(identity="error", level="ERROR", service="nova", request=REQUEST, seconds=10, **extra):
    source = {
        "@timestamp": (STAMP + timedelta(seconds=seconds)).isoformat(),
        "log_level": level, "Logger": service,
        "Payload": "Compute operation failed with HTTP 500",
    }
    if request is not None:
        source["global_request_id"] = request
    source.update(extra)
    return {"_index": "logs", "_id": identity, "_source": source}


class Source:
    source_key = "test-source-algorithm-1"

    def __init__(self, errors=None, chain=None):
        self.errors = FetchResult(hits=[hit()] if errors is None else errors, total=1)
        self.chain = FetchResult(hits=[hit()] if chain is None else chain, total=1)
        self.calls = []

    def latest_timestamp(self, window=None, *, before=None):
        if window:
            return window.start
        return before or datetime.now(timezone.utc)

    def fetch_errors(self, window):
        self.calls.append(("errors", window))
        return self.errors

    def fetch_chain(self, request_id, window, service=None):
        self.calls.append(("chain", request_id, window, service))
        if isinstance(self.chain, Exception):
            raise self.chain
        return self.chain


def test_end_to_end_grouping_global_chain_and_incremental_state():
    source = Source(
        errors=[hit("e1"), hit("e2", seconds=20), hit("noise", level="INFO")],
        chain=[hit("e2", seconds=20), hit("info", level="INFO", service="neutron", seconds=-10),
               hit("e1"), hit("wrong", request=REQUEST + "extra"), hit("e1")],
    )
    with StateStore(":memory:") as store:
        monitor = Monitor(source, store)
        report = monitor.analyze(WINDOW, advance_checkpoint=True)
        assert report.fetched_errors == report.new_errors == 2
        assert len(report.groups) == 1
        group = report.groups[0]
        assert group.group.count == 2
        assert group.chain.status == "retrieved"
        assert [event.evidence_id for event in group.chain.events] == ["logs/info", "logs/e1", "logs/e2"]
        assert group.analysis_status == "not_requested"
        assert group.llm_context["evidence"][0]["service"] == "neutron"
        assert source.calls[1][-1] is None
        assert store.get_checkpoint(source.source_key) == WINDOW.end
        repeated = monitor.analyze(WINDOW)
        assert repeated.new_errors == 0 and repeated.skipped_errors == 2


def test_replay_bypasses_seen_and_does_not_consume_state_or_checkpoint():
    source = Source()
    with StateStore(":memory:") as store:
        monitor = Monitor(source, store)
        report = monitor.analyze(WINDOW, replay=True)
        assert report.new_errors == 1
        assert store.seen(source.source_key, ["logs/error"]) == set()
        assert store.get_checkpoint(source.source_key) is None
        assert monitor.analyze(WINDOW).new_errors == 1


@pytest.mark.parametrize("problem", ["source_truncated", "chain_truncated", "chain_empty", "chain_failure", "invalid"])
def test_incomplete_runs_never_consume_events(problem):
    source = Source()
    if problem == "source_truncated":
        source.errors.truncated = True
    elif problem == "chain_truncated":
        source.chain.truncated = True
    elif problem == "chain_empty":
        source.chain.hits = []
    elif problem == "chain_failure":
        source.chain = RuntimeError("secret password must not appear in report")
    else:
        source.errors.hits.append({"_source": {"log_level": "ERROR"}})
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        assert store.get_checkpoint(source.source_key) is None
        assert not store.seen(source.source_key, ["logs/error"])
        assert "secret password" not in report.model_dump_json()


def test_missing_id_is_terminal_triage_with_isolated_evidence():
    source = Source(errors=[hit(request=None)])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        chain = report.groups[0].chain
        assert chain.status == "missing_request_id"
        assert len(chain.events) == 1
        assert not any(call[0] == "chain" for call in source.calls)
        assert store.seen(source.source_key, ["logs/error"]) == {"logs/error"}


def test_watch_can_acknowledge_delivered_error_with_incomplete_chain():
    source = Source(chain=[])
    delivered = []
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(
            WINDOW, advance_checkpoint=True, accept_partial_chains=True, on_report=delivered.append,
        )
        assert delivered == [report]
        assert report.watch_partial_chains and not report.complete
        assert store.get_checkpoint(source.source_key) == WINDOW.end
        assert store.seen(source.source_key, ["logs/error"], watch=True) == {"logs/error"}
        assert not store.seen(source.source_key, ["logs/error"])


def test_local_lookup_excludes_same_id_far_from_representative():
    error = hit(request=None, request_id=REQUEST)
    far = hit("far", request=None, request_id=REQUEST, level="INFO", seconds=1000)
    source = Source(errors=[error], chain=[error, far])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
    assert [event.evidence_id for event in report.groups[0].chain.events] == ["logs/error"]


def test_max_groups_is_explicit_and_does_not_advance_progress():
    source = Source(errors=[hit(), hit("other", Logger="keystone", request=None)])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW, max_groups=1)
        assert report.total_groups == 2 and len(report.groups) == 1
        assert report.truncated
        assert store.get_checkpoint(source.source_key) is None


def test_report_delivery_failure_does_not_consume_events():
    def broken_output(report):
        raise OSError("disk full")

    source = Source()
    with StateStore(":memory:") as store:
        monitor = Monitor(source, store)
        with pytest.raises(OSError, match="disk full"):
            monitor.analyze(WINDOW, on_report=broken_output)
        assert not store.seen(source.source_key, ["logs/error"])
        assert store.get_checkpoint(source.source_key) is None
        assert monitor.analyze(WINDOW).new_errors == 1


def test_cache_is_explicit_snapshot_and_replay_reloads_source():
    source = Source()
    with StateStore(":memory:") as store:
        monitor = Monitor(source, store)
        first = monitor.analyze(WINDOW, cache_ttl_seconds=60)
        count = len(source.calls)
        delivered = []
        second = monitor.analyze(WINDOW, cache_ttl_seconds=60, on_report=delivered.append)
        assert second.run_id == first.run_id and delivered
        assert "Cached snapshot" in second.warnings[-1]
        assert len(source.calls) == count
        monitor.analyze(WINDOW, replay=True, cache_ttl_seconds=60)
        assert len(source.calls) > count


def test_custom_timestamp_and_local_id_alias_with_global_primary():
    error = hit()
    error["_source"].pop("global_request_id")
    error["_source"]["request_id"] = REQUEST.removeprefix("req-")
    error["_source"]["custom_date"] = error["_source"].pop("@timestamp")
    chain = hit("info", level="INFO", request="global-other")
    chain["_source"]["request_id"] = REQUEST
    chain["_source"]["custom_date"] = chain["_source"].pop("@timestamp")
    source = Source(errors=[error], chain=[error, chain])
    source.config = SimpleNamespace(timestamp_field="custom_date")
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        assert report.groups[0].chain.status == "retrieved"
        assert len(report.groups[0].chain.events) == 2
        assert source.calls[1][-1] == "nova"


def test_stale_empty_window_does_not_claim_health():
    source = Source(errors=[])
    source.latest_timestamp = lambda: WINDOW.start - timedelta(days=1)
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        assert report.freshness == "stale"
        assert any("no current log coverage" in warning for warning in report.warnings)
        assert report.new_errors == 0


def test_uppercase_request_prefix_matches_bare_uuid():
    error = hit(request=REQUEST.upper())
    info = hit("info", level="INFO", request=REQUEST.removeprefix("req-"))
    source = Source(errors=[error], chain=[error, info])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        assert len(report.groups[0].chain.events) == 2


def test_cached_chain_validates_each_groups_original_error():
    one = hit("one", Payload="Failure type one")
    two = hit("two", Payload="Failure type two")
    source = Source(errors=[one, two], chain=[two])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW, advance_checkpoint=True)
        assert any(entry.chain.status == "truncated" for entry in report.groups)
        for entry in report.groups:
            assert entry.group.representative.evidence_id in {event.evidence_id for event in entry.chain.events}
        assert not store.seen(source.source_key, ["logs/one", "logs/two"])
        assert store.get_checkpoint(source.source_key) is None


@pytest.mark.parametrize("kind", ["global", "local"])
def test_correlation_namespaces_are_not_interchangeable(kind):
    error = hit()
    unrelated = hit("unrelated", request="unrelated-id")
    if kind == "global":
        unrelated["_source"]["request_id"] = REQUEST
    else:
        error["_source"].pop("global_request_id")
        error["_source"]["request_id"] = REQUEST
        unrelated["_source"]["global_request_id"] = REQUEST
        unrelated["_source"]["request_id"] = "different-local-id"
    source = Source(errors=[error], chain=[error, unrelated])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
        assert [event.evidence_id for event in report.groups[0].chain.events] == ["logs/error"]


def test_watch_does_not_use_cached_manual_report_for_progress():
    source = Source()
    with StateStore(":memory:") as store:
        monitor = Monitor(source, store)
        monitor.analyze(WINDOW, cache_ttl_seconds=60)
        assert store.get_checkpoint(source.source_key) is None
        call_count = len(source.calls)
        report = monitor.analyze(WINDOW, cache_ttl_seconds=60, advance_checkpoint=True)
        assert len(source.calls) > call_count and report.new_errors == 1
        assert store.get_checkpoint(source.source_key) == WINDOW.end
        assert not any("Cached snapshot" in warning for warning in report.warnings)


def test_fresh_source_can_have_empty_historical_window():
    source = Source(errors=[])
    source.latest_timestamp = lambda window=None, before=None: None if window else before or datetime.now(timezone.utc)
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
    assert report.freshness == "fresh"
    assert report.window_coverage == "empty"
    assert report.latest_window_timestamp is None
    assert any("No log timestamps" in warning for warning in report.warnings)


def test_positional_and_configured_ids_link_different_services():
    local = "req-22222222-2222-4222-8222-222222222222"
    error = hit(request=None, Payload=f"2026-11-15 06:00:10.123 321 ERROR nova.api [{REQUEST} {local} u p - - - -] failed")
    other = hit("other", level="INFO", service="neutron", request=None, correlation={"global": REQUEST})
    decoy = hit("decoy", level="INFO", service="nova", request=None, request_id=REQUEST)
    source = Source(errors=[error], chain=[other, decoy, error])
    source.config = SimpleNamespace(normalization_global_request_id_fields=("correlation.global",))
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
    group, = report.groups
    assert group.group.representative.global_request_id == REQUEST
    assert group.group.representative.local_request_id == local
    assert {e.evidence_id for e in group.chain.events} == {"logs/error", "logs/other"}
    assert source.calls[-1][-1] is None


def test_local_id_without_service_remains_isolated():
    error = hit(request=None, request_id=REQUEST)
    del error["_source"]["Logger"]
    source = Source(errors=[error])
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW)
    assert report.complete
    assert report.groups[0].chain.status == "missing_local_service"
    assert [event.evidence_id for event in report.groups[0].chain.events] == ["logs/error"]
    assert not any(call[0] == "chain" for call in source.calls)


def test_concurrent_runs_on_same_state_deliver_new_error_once(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier

    barrier = Barrier(2, timeout=5)
    path = tmp_path / "shared.sqlite3"

    class SharedSource(Source):
        def fetch_errors(self, window):
            barrier.wait()
            return self.errors

    def run():
        delivered = []
        with StateStore(path) as store:
            Monitor(SharedSource(), store).analyze(WINDOW, on_report=delivered.append)
        return delivered[0].new_errors

    with StateStore(path):
        pass
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert sorted(executor.map(lambda _: run(), range(2))) == [0, 1]


def test_future_timestamp_cannot_advance_watch_checkpoint():
    now = WINDOW.end + timedelta(minutes=20)
    actual_latest = WINDOW.start - timedelta(days=1)
    source = Source(errors=[])

    def latest_timestamp(window=None, *, before=None):
        assert window is None
        return actual_latest if before == now else now + timedelta(days=1)

    source.latest_timestamp = latest_timestamp
    with StateStore(":memory:") as store:
        report = Monitor(source, store).analyze(WINDOW, now=now, advance_checkpoint=True)
        assert store.get_checkpoint(source.source_key) is None
    assert report.watch_paused
    assert report.freshness == "stale" and report.window_coverage == "empty"
    assert report.latest_source_timestamp == actual_latest
    assert not source.calls
