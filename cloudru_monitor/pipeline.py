"""Deterministic incident triage. No model calls and no inferred root causes."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from uuid import uuid4
from collections.abc import Callable

from .models import AnalysisReport, ChainReport, FetchResult, GroupReport, LogEvent, TimeWindow
from .processing import build_llm_context, canonical_request_id, group_errors, normalize_hit
from .source import RetryableSourceError, SourceError
from .version import ALGORITHM_VERSION


def _matches_request(event: LogEvent, request_id: str, kind: str | None) -> bool:
    target = canonical_request_id(request_id)
    if kind == "global":
        values = (event.global_request_id, event.request_id if event.request_id_kind == "global" else None)
    else:
        values = (event.local_request_id, event.request_id if event.request_id_kind == "local" else None)
    return any(
        value is not None and canonical_request_id(value) == target
        for value in values
    )


class Monitor:
    def __init__(self, source, store):
        self.source = source
        self.store = store

    def analyze(
        self,
        window: TimeWindow,
        *,
        replay: bool = False,
        freshness_seconds: int = 900,
        chain_padding_seconds: int = 300,
        max_groups: int = 50,
        cache_ttl_seconds: int = 0,
        on_report: Callable[[AnalysisReport], None] | None = None,
        advance_checkpoint: bool = False,
        accept_partial_chains: bool = False,
        split_limited: bool = False,
        now: datetime | None = None,
        watch_pause_reason: str | None = None,
    ) -> AnalysisReport:
        if freshness_seconds <= 0 or chain_padding_seconds < 0 or max_groups <= 0 or cache_ttl_seconds < 0:
            raise ValueError("invalid analysis limits")
        now = now if now is not None else datetime.now(timezone.utc)
        cache_key = hashlib.sha256(json.dumps({
            "source": self.source.source_key, "window": window.model_dump(mode="json"),
            "freshness": freshness_seconds, "padding": chain_padding_seconds,
            "max_groups": max_groups, "algorithm": ALGORITHM_VERSION,
        }, sort_keys=True).encode()).hexdigest()
        # Replay means a fresh historical scan and never consumes incremental state.
        if not replay and not advance_checkpoint and cache_ttl_seconds:
            cached = self.store.cached_report(cache_key, cache_ttl_seconds)
            if cached is not None:
                cached.warnings.append("Cached snapshot: counts and freshness are from the original run.")
                if on_report is not None:
                    on_report(cached)
                return cached

        latest = self.source.latest_timestamp()
        future_timestamp = latest is not None and latest > now
        if future_timestamp:
            latest = self.source.latest_timestamp(before=now)
        freshness = "empty" if latest is None else (
            "stale" if (now - latest).total_seconds() > freshness_seconds else "fresh"
        )
        report = AnalysisReport(
            run_id=str(uuid4()), generated_at=now, source=self.source.source_key,
            window=window, latest_source_timestamp=latest, freshness=freshness,
        )
        if future_timestamp:
            report.warnings.append("Timestamps after the observation time were excluded from source freshness.")
        if freshness != "fresh":
            report.warnings.append(
                "Source has no timestamps." if latest is None else
                "Source is stale relative to current time; no errors does not mean a healthy live system."
            )
        if latest is not None and latest < window.start:
            report.warnings.append("Latest source timestamp precedes the analysis window; no current log coverage.")
        window_latest = None
        if latest is not None and latest >= window.start:
            window_latest = latest if latest < window.end else self.source.latest_timestamp(window=window)
        report.latest_window_timestamp = window_latest
        report.window_coverage = "present" if window_latest is not None else "empty"
        if window_latest is None:
            report.warnings.append("No log timestamps in the requested window; traffic and health are unconfirmed.")

        padding = timedelta(seconds=chain_padding_seconds)
        if advance_checkpoint and not replay:
            if freshness != "fresh":
                watch_pause_reason = "the source is not fresh"
            elif latest < window.end + padding:
                watch_pause_reason = "the source has not reached the end of the chain window"
            elif now < window.end + padding:
                watch_pause_reason = "the chain window extends past the current time"
            if watch_pause_reason:
                report.watch_paused = True
                report.warnings.append(f"Watch paused because {watch_pause_reason}. The checkpoint is unchanged.")
        fetched = FetchResult() if report.watch_paused else self.source.fetch_errors(window)
        report.fetched_candidates = len(fetched.hits)
        report.total_candidates = fetched.total
        report.truncated = fetched.truncated
        report.error_scan_limited = fetched.truncated
        report.warnings.extend(fetched.warnings)
        errors: dict[str, LogEvent] = {}
        for hit in fetched.hits:
            try:
                event = self._normalize(hit)
            except (ValueError, TypeError, KeyError):
                report.invalid_errors += 1
                continue
            # Source search is deliberately broad: words inside messages aren't levels.
            if event.level != "ERROR" or not window.start <= event.timestamp < window.end:
                continue
            errors[event.evidence_id] = event
        report.fetched_errors = len(errors)
        # Serialize acknowledgement and report delivery across processes sharing a state DB.
        with self.store.serialize_run():
            self._finish(report, errors, padding, max_groups, replay, advance_checkpoint,
                         accept_partial_chains, split_limited, on_report)
            if not replay and not advance_checkpoint and cache_ttl_seconds and report.complete:
                self.store.cache_report(cache_key, report)
        return report

    def _finish(self, report, errors, padding, max_groups, replay, advance_checkpoint,
                accept_partial_chains, split_limited, on_report):
        window = report.window
        seen = set() if replay else self.store.seen(self.source.source_key, list(errors), watch=advance_checkpoint)
        new_events = [event for key, event in errors.items() if key not in seen]
        report.skipped_errors = len(errors) - len(new_events)
        report.new_errors = len(new_events)
        groups = group_errors(new_events)
        report.total_groups = len(groups)
        if len(groups) > max_groups:
            report.truncated = True
            report.error_scan_limited = True
            report.warnings.append(
                f"Only {max_groups} of {len(groups)} groups were analyzed; state will not advance."
            )
        if split_limited and report.error_scan_limited:
            return
        chain_window = TimeWindow(
            start=window.start - padding,
            end=window.end + padding,
        )
        chain_cache = {}
        for group in groups[:max_groups]:
            representative = group.representative
            request_id = representative.request_id
            if request_id is None:
                chain = ChainReport(
                    request_id=None, status="missing_request_id", events=[representative], total=1,
                    warnings=["No request ID: this is an isolated error, not a reconstructed request chain."],
                )
            elif representative.request_id_kind == "local" and representative.service == "unknown":
                chain = ChainReport(
                    request_id=request_id, status="missing_local_service", events=[representative], total=1,
                    warnings=["Reliable correlation of a local request ID requires a known service."],
                )
            else:
                # Global IDs can span services. Local IDs are scoped to their origin.
                service = representative.service if representative.request_id_kind == "local" else None
                lookup = chain_window
                if service is not None:
                    lookup = TimeWindow(
                        start=representative.timestamp - padding,
                        end=representative.timestamp + padding + timedelta(microseconds=1),
                    )
                cache_id = (canonical_request_id(request_id), service, representative.request_id_kind,
                            lookup.start, lookup.end)
                if cache_id not in chain_cache:
                    chain_cache[cache_id] = self._chain(request_id, service, lookup, representative)
                chain = chain_cache[cache_id].model_copy(deep=True)
                chain.request_id = request_id
                # A cached lookup must still prove the current group's own error.
                if not any(event.evidence_id == representative.evidence_id for event in chain.events):
                    chain.events.append(representative)
                    chain.events.sort(key=lambda event: (event.timestamp, event.evidence_id))
                    chain.total = len(chain.events)
                    if chain.status == "retrieved":
                        chain.status = "truncated"
                    chain.warnings.append("Current group's error was absent from cached chain results; retry is required.")
            report.groups.append(GroupReport(
                group=group, chain=chain, llm_context=build_llm_context(group, chain),
            ))
            if chain.status == "truncated":
                report.truncated = True
        if report.invalid_errors:
            report.warnings.append(f"Rejected {report.invalid_errors} malformed candidates; state will not advance.")
        checkpoint = window.end if advance_checkpoint and not replay else None
        if (checkpoint is not None and accept_partial_chains and on_report is not None
                and not report.complete and not report.watch_paused and not report.error_scan_limited
                and not report.invalid_errors and all(entry.chain.status != "unavailable" for entry in report.groups)):
            report.watch_partial_chains = True
            report.warnings.append(
                "All ERRORs were delivered, with incomplete chains. Watch acknowledged these errors; "
                "use analyze --replay with adjusted limits or a smaller window to revisit the chains."
            )
        # Delivery first: a full disk or broken stdout must not consume errors.
        if on_report is not None:
            on_report(report)
        self.store.commit_run(
            self.source.source_key, report,
            [] if replay else [event.evidence_id for event in new_events], checkpoint,
        )

    def _normalize(self, hit) -> LogEvent:
        config = getattr(self.source, "config", None)
        return normalize_hit(
            hit, timestamp_field=getattr(config, "timestamp_field", "@timestamp"),
            timestamp_fields=getattr(config, "timestamp_fields", None),
            global_request_id_fields=getattr(config, "normalization_global_request_id_fields", ()),
            local_request_id_fields=getattr(config, "normalization_local_request_id_fields", ()),
        )

    def _chain(self, request_id, service, window, representative) -> ChainReport:
        warnings = [
            "Chain retrieval is bounded by the requested time window plus padding; completeness is not guaranteed."
        ]
        if service:
            warnings.append("Local request ID matches are limited to the reported service and time around the error.")
        try:
            fetched = self.source.fetch_chain(request_id, window, service=service)
        except RetryableSourceError as exc:
            return ChainReport(
                request_id=request_id, status="unavailable", events=[representative], total=1,
                warnings=warnings + [f"{exc}; chain lookup will need to be retried."],
            )
        except SourceError:
            raise
        except (RuntimeError, OSError, ValueError) as exc:
            # Persist a retryable failure without leaking URLs, credentials or raw exception data.
            return ChainReport(
                request_id=request_id, status="unavailable", events=[representative], total=1,
                warnings=warnings + [f"Source chain lookup failed ({type(exc).__name__}); retry is required."],
            )
        warnings.extend(fetched.warnings)
        if fetched.truncated:
            warnings.append(
                f"Chain lookup retrieved {len(fetched.hits)} of {fetched.total} source candidates; "
                "increase OPENSEARCH_MAX_CHAIN_EVENTS or narrow the analysis window."
            )
        events: dict[str, LogEvent] = {}
        invalid = 0
        for hit in fetched.hits:
            try:
                event = self._normalize(hit)
            except (ValueError, TypeError, KeyError):
                invalid += 1
                continue
            if not window.start <= event.timestamp < window.end or not _matches_request(event, request_id, representative.request_id_kind):
                continue
            if service and event.service != service:
                continue
            events[event.evidence_id] = event
        if invalid:
            warnings.append(f"{invalid} malformed chain candidates were rejected.")
        status = "truncated" if fetched.truncated or invalid else ("retrieved" if events else "empty")
        if not events:
            warnings.append("Lookup returned no matching chain; the original error is retained as evidence.")
        if representative.evidence_id not in events:
            if status == "retrieved":
                status = "truncated"
                warnings.append("Original error was absent from chain results; source results may have changed.")
            events[representative.evidence_id] = representative
        ordered = sorted(events.values(), key=lambda event: (event.timestamp, event.evidence_id))
        if len(ordered) == 1 and status == "retrieved":
            warnings.append("Only the original error matched within the chain window.")
        return ChainReport(
            request_id=request_id, status=status, events=ordered, total=len(ordered), warnings=warnings,
            fetched_candidates=len(fetched.hits), total_candidates=fetched.total,
        )
