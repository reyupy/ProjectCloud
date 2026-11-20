import pytest

from cloudru_monitor.cli import load_env, parser, source_config


def test_literal_dotenv_without_override_or_execution(tmp_path):
    path = tmp_path / ".env"
    path.write_text('A="literal $(danger)"\nB=local\n# ignored\n', encoding="utf-8")
    environment = {"B": "existing"}
    load_env(path, environment)
    assert environment == {"A": "literal $(danger)", "B": "existing"}


def test_invalid_dotenv_fails_without_value_leak(tmp_path):
    path = tmp_path / ".env"
    path.write_text("BROKEN secret", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1") as exc:
        load_env(path, {})
    assert "secret" not in str(exc.value)


def test_parser_requires_window_and_rejects_negative_limits():
    with pytest.raises(SystemExit):
        parser().parse_args(["analyze"])
    with pytest.raises(SystemExit):
        parser().parse_args(["watch", "--interval", "0"])
    with pytest.raises(SystemExit):
        parser().parse_args(["watch", "--overlap", "-1"])
    args = parser().parse_args(["watch", "--once"])
    assert args.once and args.interval == 60


def test_source_settings_explicit_environment(monkeypatch):
    monkeypatch.setenv("OPENSEARCH_URL", "https://example.invalid:9200")
    monkeypatch.delenv("OPENSEARCH_USERNAME", raising=False)
    monkeypatch.delenv("OPENSEARCH_PASSWORD", raising=False)
    monkeypatch.setenv("OPENSEARCH_REQUEST_ID_FIELDS", "context.request_id, global_request_id.keyword")
    monkeypatch.setenv("OPENSEARCH_GLOBAL_REQUEST_ID_FIELDS", "correlation.global")
    monkeypatch.setenv("OPENSEARCH_LOCAL_REQUEST_ID_FIELDS", "labels.request")
    config = source_config()
    assert config.request_id_fields == ("context.request_id", "global_request_id.keyword")
    assert config.normalization_global_request_id_fields == ("correlation.global",)
    assert config.normalization_local_request_id_fields == ("labels.request",)
    monkeypatch.setenv("OPENSEARCH_PAGE_SIZE", "0")
    with pytest.raises(ValueError):
        source_config()


def test_output_failure_does_not_consume_events(tmp_path, monkeypatch, capsys):
    from datetime import datetime, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult
    from cloudru_monitor.storage import StateStore

    class FakeSource:
        source_key = "output-test"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self):
            return datetime(2026, 11, 15, 9, tzinfo=timezone.utc)

        def fetch_errors(self, window):
            return FetchResult(hits=[{"_index": "flog-test", "_id": "one", "_source": {
                "@timestamp": "2026-11-15T09:00:00Z", "level": "ERROR",
                "message": "Keystone connection failed", "service": "keystone"}}], total=1)

    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "emit", lambda *_: (_ for _ in ()).throw(OSError("output unavailable")))
    state = tmp_path / "state.sqlite3"
    code = cli.main(["analyze", "--start", "2026-11-15T08:00:00Z", "--end", "2026-11-15T10:00:00Z",
                     "--state", str(state), "--env-file", str(tmp_path / "absent")])
    assert code == 1
    assert "output unavailable" in capsys.readouterr().err
    with StateStore(state) as store:
        assert store.get_checkpoint("output-test") is None
        assert store.connection.execute("SELECT count(*) FROM processed").fetchone()[0] == 0
        assert store.connection.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_sqlite_error_is_graceful(tmp_path, monkeypatch, capsys):
    from cloudru_monitor import cli
    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    assert cli.main(["watch", "--once", "--state", str(tmp_path),
                     "--env-file", str(tmp_path / "absent")]) == 1
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("failure", ["truncated", "exception"])
def test_watch_cold_start_preserves_pending_window_on_restart(tmp_path, monkeypatch, failure):
    from datetime import datetime, timedelta, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult
    from cloudru_monitor.storage import StateStore

    windows = []
    clock = [datetime(2026, 11, 17, 12, tzinfo=timezone.utc)]

    class Clock:
        @staticmethod
        def now(_):
            return clock[0]

    class FakeSource:
        source_key = "pending-test"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            return window.start if window else clock[0]

        def fetch_errors(self, window):
            windows.append(window)
            if failure == "exception":
                raise RuntimeError("source unavailable")
            return FetchResult(total=1, truncated=True)

    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "emit", lambda *_: None)
    state = tmp_path / "state.sqlite3"
    arguments = ["watch", "--once", "--lookback", "900", "--state", str(state),
                 "--env-file", str(tmp_path / "absent")]
    assert cli.main(arguments) == (1 if failure == "exception" else 2)
    original_start = clock[0] - timedelta(seconds=1500)
    clock[0] += timedelta(hours=2)
    assert cli.main(arguments) == (1 if failure == "exception" else 2)
    assert all(window.start == original_start for window in windows)
    if failure == "exception":
        assert len(windows) == 2
        assert windows[1].end > windows[0].end
    else:
        assert len(windows) > 2
        assert min(window.end - window.start for window in windows) <= timedelta(seconds=1)
    with StateStore(state) as store:
        assert store.get_checkpoint("pending-test") is None
        assert store.connection.execute("SELECT count(*) FROM processed").fetchone()[0] == 0


def test_continuous_watch_resumes_same_start_after_connection_loss(tmp_path, monkeypatch):
    from datetime import datetime, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult

    clock = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    windows = []

    class FakeSource:
        source_key = "connection-loss"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            return window.start if window else clock

        def fetch_errors(self, window):
            windows.append(window)
            if len(windows) == 1:
                raise source.RetryableSourceError("Connection lost")
            return FetchResult()

    class Clock:
        @staticmethod
        def now(_):
            return clock

    def sleep(_):
        if len(windows) == 2:
            raise KeyboardInterrupt

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli.time, "sleep", sleep)
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "source_config", lambda: None)
    assert cli.main(["watch", "--state", ":memory:", "--env-file", str(tmp_path / "absent")]) == 130
    assert len(windows) == 2 and windows[0] == windows[1]


def test_watch_reduces_limited_window_and_saves_smaller_checkpoint(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult
    from cloudru_monitor.storage import StateStore

    clock = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    windows = []

    class FakeSource:
        source_key = "limited-window"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            return window.start if window else clock

        def fetch_errors(self, window):
            windows.append(window)
            return FetchResult(truncated=window.end - window.start > timedelta(minutes=10))

    class Clock:
        @staticmethod
        def now(_):
            return clock

    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "source_config", lambda: None)
    state = tmp_path / "state.sqlite3"
    assert cli.main(["watch", "--once", "--state", str(state), "--env-file", str(tmp_path / "absent")]) == 0
    assert len(windows) == 2 and windows[0].start == windows[1].start
    assert windows[1].end < windows[0].end
    with StateStore(state) as store:
        assert store.get_checkpoint("limited-window") == windows[1].end


def test_manual_disjoint_analyze_preserves_watch_pending_window(tmp_path, monkeypatch):
    from datetime import datetime, timedelta, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult
    from cloudru_monitor.storage import StateStore

    windows = []
    pending = datetime(2026, 11, 15, tzinfo=timezone.utc)
    now = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(_):
            return now

    class FakeSource:
        source_key = "manual-watch-test"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            return window.start if window else now

        def fetch_errors(self, window):
            windows.append(window)
            return FetchResult()

    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "datetime", Clock)
    monkeypatch.setattr(cli, "emit", lambda *_: None)
    state = tmp_path / "state.sqlite3"
    with StateStore(state) as store:
        store.reserve_start("manual-watch-test", pending)
    shared = ["--state", str(state), "--env-file", str(tmp_path / "absent")]
    assert cli.main(["analyze", "--start", "2026-11-17T00:00:00Z", "--end", "2026-11-17T01:00:00Z", *shared]) == 0
    with StateStore(state) as store:
        assert store.get_checkpoint("manual-watch-test") is None
        row = store.connection.execute("SELECT start FROM pending_scans WHERE source=?", ("manual-watch-test",)).fetchone()
        assert datetime.fromisoformat(row[0]) == pending
    assert cli.main(["watch", "--once", *shared]) == 0
    assert windows[-1].start == pending
    with StateStore(state) as store:
        assert store.get_checkpoint("manual-watch-test") == pending + timedelta(hours=1)
        assert store.connection.execute("SELECT count(*) FROM pending_scans").fetchone()[0] == 0
    assert cli.main(["watch", "--once", "--max-window-seconds", "600", *shared]) == 0
    assert windows[-1].end == pending + timedelta(minutes=70)


@pytest.mark.parametrize("lag_seconds", [360, 7200])
def test_watch_recovers_delayed_errors_and_preserves_nonempty_report(tmp_path, monkeypatch, lag_seconds):
    import json
    from datetime import datetime, timedelta, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult
    from cloudru_monitor.storage import StateStore

    now = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    clock = [now]
    latest = [now - timedelta(seconds=lag_seconds)]
    rows = []
    error_time = now - timedelta(minutes=15)

    class Clock:
        @staticmethod
        def now(_):
            return clock[0]

    class FakeSource:
        source_key = "watch-delayed-test"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            if window is None:
                return latest[0]
            return max((stamp for stamp, _ in rows if window.start <= stamp < window.end), default=None)

        def fetch_errors(self, window):
            return self.fetch_chain("shared-id", window)

        def fetch_chain(self, request_id, window, service=None):
            hits = [hit for stamp, hit in rows if window.start <= stamp < window.end]
            return FetchResult(hits=hits, total=len(hits))

    def publish(stamp, level, service):
        rows.append((stamp, {"_index": "test", "_id": service, "_source": {
            "@timestamp": stamp.isoformat(), "log_level": level, "service": service,
            "global_request_id": "shared-id", "message": level + " synthetic event",
        }}))

    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "datetime", Clock)
    state = tmp_path / "state.sqlite3"
    output = tmp_path / "latest.json"
    args = ["watch", "--once", "--state", str(state), "--output", str(output),
            "--env-file", str(tmp_path / "absent")]

    assert cli.main(args) == 2
    paused = json.loads(output.read_text(encoding="utf-8"))
    assert paused["watch_paused"]
    assert paused["freshness"] == ("stale" if lag_seconds > 900 else "fresh")
    with StateStore(state) as store:
        assert store.get_checkpoint(FakeSource.source_key) is None

    # Documents arrive after the first run, outside its usual one-minute overlap.
    clock[0] = latest[0] = now + timedelta(minutes=5)
    publish(error_time, "ERROR", "nova")
    publish(error_time + timedelta(seconds=30), "INFO", "neutron")
    assert cli.main(args) == 0
    recovered = json.loads(output.read_text(encoding="utf-8"))
    assert recovered["window_coverage"] == "present"
    assert recovered["new_errors"] == 1
    assert {e["service"] for e in recovered["groups"][0]["chain"]["events"]} == {"nova", "neutron"}
    with StateStore(state) as store:
        assert store.get_checkpoint(FakeSource.source_key) == clock[0] - timedelta(minutes=10)

    clock[0] = latest[0] = now + timedelta(minutes=6)
    assert cli.main(args) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["new_errors"] == 0
    saved, = (tmp_path / "latest.json.history").glob("*.json")
    assert json.loads(saved.read_text(encoding="utf-8")) == recovered


def test_watch_waits_for_late_chain_events_even_after_manual_analysis(tmp_path, monkeypatch, capsys):
    import json
    from datetime import datetime, timedelta, timezone
    from cloudru_monitor import cli, source
    from cloudru_monitor.models import FetchResult

    now = datetime(2026, 11, 17, 12, tzinfo=timezone.utc)
    clock = [now]
    rows = []

    class Clock:
        @staticmethod
        def now(_):
            return clock[0]

    class FakeSource:
        source_key = "watch-settle-test"

        def __init__(self, config):
            self.config = config

        def latest_timestamp(self, window=None):
            if window is None:
                return clock[0]
            return max((stamp for stamp, _ in rows if window.start <= stamp < window.end), default=None)

        def fetch_errors(self, window):
            hits = [h for stamp, h in rows if window.start <= stamp < window.end and h["_id"] == "error"]
            return FetchResult(hits=hits, total=len(hits))

        def fetch_chain(self, request_id, window, service=None):
            hits = [h for stamp, h in rows if window.start <= stamp < window.end]
            return FetchResult(hits=hits, total=len(hits))

    def publish(stamp, identity, level):
        rows.append((stamp, {"_index": "test", "_id": identity, "_source": {
            "@timestamp": stamp.isoformat(), "log_level": level, "service": "nova",
            "global_request_id": "global-id", "message": "event " + identity,
        }}))

    monkeypatch.setenv("OPENSEARCH_URL", "http://example.invalid:9200")
    monkeypatch.setattr(source, "OpenSearchSource", FakeSource)
    monkeypatch.setattr(cli, "datetime", Clock)
    common = ["--state", str(tmp_path / "state.sqlite3"), "--env-file", str(tmp_path / "absent")]
    error_time = now - timedelta(minutes=9)
    publish(error_time, "error", "ERROR")

    args = ["analyze", "--start", (error_time - timedelta(minutes=1)).isoformat(),
            "--end", error_time.replace(second=5).isoformat(), *common]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["groups"][0]["chain"]["total"] == 1

    assert cli.main(["watch", "--once", *common]) == 0
    assert json.loads(capsys.readouterr().out)["new_errors"] == 0

    clock[0] += timedelta(minutes=2)
    publish(error_time + timedelta(seconds=30), "support", "INFO")
    assert cli.main(["watch", "--once", *common]) == 0
    chain = json.loads(capsys.readouterr().out)["groups"][0]["chain"]
    assert [event["evidence_id"] for event in chain["events"]] == ["test/error", "test/support"]
