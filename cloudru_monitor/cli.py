"""Command-line entry point. JSON reports on stdout, diagnostics on stderr."""
import argparse
import os
import re
import sqlite3
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import TimeWindow
from .storage import StateStore


def load_env(path: str | Path, environ=None) -> None:
    """Read literal KEY=value lines; no expansion, shell execution or env override."""
    env = os.environ if environ is None else environ
    file = Path(path)
    if not file.exists():
        return
    for number, line in enumerate(file.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key.strip()):
            raise ValueError(f"invalid .env assignment on line {number}")
        value = value.strip()
        if value.startswith(("'", '"')):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f"unclosed .env quote on line {number}")
            value = value[1:-1]
        env.setdefault(key.strip(), value)


def positive(value: str) -> int:
    integer = int(value)
    if integer <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return integer


def nonnegative(value: str) -> int:
    integer = int(value)
    if integer < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return integer


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="OpenSearch error grouping and request-chain collection; no LLM")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("analyze", "watch"):
        command = commands.add_parser(name)
        command.add_argument("--env-file", default=".env")
        command.add_argument("--state", default=".local/state.sqlite3")
        command.add_argument("--output", help="JSON file (watch overwrites atomically each run); stdout by default")
        command.add_argument("--freshness-seconds", type=positive, default=900)
        command.add_argument("--chain-padding-seconds", type=nonnegative, default=300)
        command.add_argument("--max-groups", type=positive, default=50)
        command.add_argument("--cache-ttl-seconds", type=nonnegative, default=0)
    analyze = commands.choices["analyze"]
    analyze.add_argument("--start", required=True, help="ISO timestamp including timezone")
    analyze.add_argument("--end", required=True, help="ISO timestamp including timezone (exclusive)")
    analyze.add_argument("--replay", action="store_true", help="include already processed errors; bypass cache")
    watch = commands.choices["watch"]
    watch.add_argument("--interval", type=positive, default=60)
    watch.add_argument("--lookback", type=positive, default=900)
    watch.add_argument("--overlap", type=nonnegative, default=60)
    watch.add_argument("--lateness-seconds", type=nonnegative, default=300,
                       help="expected maximum event-to-index delay before finalizing a window")
    watch.add_argument("--max-window-seconds", type=positive, default=3600,
                       help="maximum new seconds per watch pass when catching up")
    watch.add_argument("--once", action="store_true")
    return result


def source_config():
    from .source import SourceConfig
    url = os.getenv("OPENSEARCH_URL", "").strip()
    if not url:
        raise ValueError("OPENSEARCH_URL is required (environment or .env)")
    settings = dict(
        url=url, index_pattern=os.getenv("OPENSEARCH_INDEX", "flog-*"),
        username=os.getenv("OPENSEARCH_USERNAME") or None,
        password=os.getenv("OPENSEARCH_PASSWORD") or None,
        timeout_seconds=float(os.getenv("OPENSEARCH_TIMEOUT_SECONDS", "15")),
        page_size=int(os.getenv("OPENSEARCH_PAGE_SIZE", "500")),
        max_errors=int(os.getenv("OPENSEARCH_MAX_ERRORS", "10000")),
        max_chain_events=int(os.getenv("OPENSEARCH_MAX_CHAIN_EVENTS", "2000")),
        timestamp_field=os.getenv("OPENSEARCH_TIMESTAMP_FIELD", "@timestamp"),
        timestamp_fallback_fields=tuple(
            part.strip() for part in os.getenv("OPENSEARCH_TIMESTAMP_FALLBACK_FIELDS", "").split(",") if part.strip()
        ),
    )
    for setting in ("request_id_fields", "global_request_id_fields", "local_request_id_fields"):
        fields = os.getenv("OPENSEARCH_" + setting.upper())
        if fields:
            settings[setting] = tuple(part.strip() for part in fields.split(",") if part.strip())
    return SourceConfig(**settings)


def _write_report(destination: Path, value: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp",
            mode="w", encoding="utf-8", delete=False,
        ) as file:
            temporary = Path(file.name)
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def emit(report, output: str | None, retain_history: bool = False) -> None:
    value = report.model_dump_json(indent=2) + "\n"
    if output:
        destination = Path(output)
        _write_report(destination, value)
        if retain_history and (report.complete or report.watch_partial_chains) and report.groups:
            history = destination.with_name(destination.name + ".history")
            _write_report(history / (report.run_id + ".json"), value)
    else:
        print(value, end="", flush=True)
    statuses = {}
    for entry in report.groups:
        statuses[entry.chain.status] = statuses.get(entry.chain.status, 0) + 1
    chain_summary = ",".join(f"{key}:{count}" for key, count in sorted(statuses.items())) or "none"
    print(f"errors={report.new_errors} groups={len(report.groups)} freshness={report.freshness} "
          f"coverage={report.window_coverage} paused={report.watch_paused} "
          f"truncated={report.truncated} chains={chain_summary}", file=sys.stderr)
    for warning in report.warnings:
        print(f"warning: {warning}", file=sys.stderr)


def _analyze(monitor, args, window, *, now=None, watch_pause_reason=None, split_limited=False):
    watch = args.command == "watch"
    return monitor.analyze(window, replay=getattr(args, "replay", False), advance_checkpoint=watch,
                           freshness_seconds=args.freshness_seconds,
                           chain_padding_seconds=args.chain_padding_seconds,
                           max_groups=args.max_groups, cache_ttl_seconds=args.cache_ttl_seconds,
                           on_report=lambda report: emit(report, args.output, watch),
                           accept_partial_chains=watch and args.output is not None,
                           split_limited=split_limited,
                           now=now, watch_pause_reason=watch_pause_reason)


def _watch(monitor, args):
    store = monitor.store
    source_key = monitor.source.source_key
    delay = timedelta(seconds=args.lateness_seconds + args.chain_padding_seconds)
    lookback = timedelta(seconds=args.lookback)
    # Reserve the cold start in a separate transaction so a read or delivery
    # failure in the first pass cannot erase it.
    if store.get_checkpoint(source_key) is None:
        store.reserve_start(source_key, datetime.now(timezone.utc) - delay - lookback)
    with store.serialize_run():
        now = datetime.now(timezone.utc)
        end = now - delay
        checkpoint = store.get_checkpoint(source_key)
        if checkpoint is not None and checkpoint > now:
            raise ValueError("checkpoint is in the future; check system clock or use a new state file")
        start = (checkpoint - timedelta(seconds=args.overlap)) if checkpoint else (
            store.reserve_start(source_key, end - lookback))
        if start >= end:
            return _analyze(monitor, args, TimeWindow(start=end - lookback, end=end), now=now,
                            watch_pause_reason="the settled window has not reached the saved start")
        target = min(end, (checkpoint or start) + timedelta(seconds=args.max_window_seconds))
        end = target
        while True:
            can_split = end - start > timedelta(seconds=1)
            report = _analyze(monitor, args, TimeWindow(start=start, end=end), now=now,
                              split_limited=can_split)
            if report.error_scan_limited and can_split:
                end = start + (end - start) / 2
                continue
            if (checkpoint is None or end > checkpoint
                    or not (report.complete or report.watch_partial_chains) or end == target):
                return report
            # This slice covered only the overlap. Continue towards the saved
            # checkpoint without dropping late arrivals or moving it backwards.
            start, end = end, target


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.output and args.state != ":memory:" and Path(args.state).resolve() == Path(args.output).resolve():
            raise ValueError("report output must differ from the SQLite state path")
        load_env(args.env_file)
        from .source import OpenSearchSource, RetryableSourceError
        from .pipeline import Monitor
        source = OpenSearchSource(source_config())
        with StateStore(args.state) as store:
            monitor = Monitor(source, store)
            while True:
                if args.command == "analyze":
                    report = _analyze(monitor, args, TimeWindow(start=args.start, end=args.end))
                else:
                    try:
                        report = _watch(monitor, args)
                    except RetryableSourceError as exc:
                        if args.once:
                            raise
                        print(f"warning: {exc}; watch will retry the saved window", file=sys.stderr)
                        time.sleep(args.interval)
                        continue
                if args.command == "analyze" or args.once:
                    return 0 if report.complete else 2
                if report.error_scan_limited or report.invalid_errors:
                    print("error: watch cannot finish this window; check candidate and group limits or malformed records",
                          file=sys.stderr)
                    return 2
                if not report.complete and not report.watch_paused and not report.watch_partial_chains:
                    if all(entry.chain.status != "unavailable" for entry in report.groups):
                        print("error: incomplete chains; use --output to retain partial reports or adjust lookup limits",
                              file=sys.stderr)
                        return 2
                time.sleep(args.interval)
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        # Connector errors must already be sanitized; never display config repr.
        print(f"error: {exc}", file=sys.stderr)
        return 1
