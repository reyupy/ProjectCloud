from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json

import pytest

from cloudru_monitor.models import ChainReport
from cloudru_monitor.processing import build_llm_context, group_errors, normalize_hit
from cloudru_monitor.version import ALGORITHM_VERSION

R1 = "req-12345678-1234-1234-1234-123456789abc"
R2 = "req-87654321-1234-1234-1234-123456789abc"


def hit(message="ERROR failure", **fields):
    return {"_index": "flog-2026.11.15", "_id": "1", "_source": {"@timestamp": "2026-11-15T10:00:00+03:00", "Payload": message, "Logger": "openstack.nova", **fields}}


def test_application_level_over_syslog_severity():
    event = normalize_hit(hit(f"2026-11-15 10:00:00.123 121 ERROR nova.api [{R1}] failed\nTraceback (most recent call last):\nValueError: broken", Severity=6))
    assert event.level == "ERROR"
    assert event.request_id == R1
    assert event.request_id_kind == "local"
    assert "Traceback" in event.message


def test_global_wins_over_local():
    event = normalize_hit(hit(f"ERROR [{R1}] global_request_id={R2} failed", request_id=R1))
    assert event.request_id == R2
    assert event.request_id_kind == "global"
    assert event.global_request_id == R2
    assert event.local_request_id == R1


@pytest.mark.parametrize(("slots", "expected"), [
    (f"{R1} {R2} user project - - - -", (R1, R2)),
    (f"- {R2} user project - - - -", (None, R2)),
    (f"None {R2} user project - - - -", (None, R2)),
    (f"{R2} user project - - -", (None, R2)),
    (f"{R1} - user project - - - -", (R1, None)),
])
def test_oslo_log_reads_modern_and_legacy_request_contexts(slots, expected):
    event = normalize_hit(hit(f"2026-11-15 10:00:00.123 121 ERROR nova.api [{slots}] Failed"))
    assert (event.global_request_id, event.local_request_id) == expected
    assert event.request_id_kind == ("global" if expected[0] is not None else "local")


def test_structured_id_overrides_its_slot_without_affecting_other_namespace():
    event = normalize_hit(hit(f"ERROR nova.api [{R1} {R2} - - - - - -] failed", request_id="collector-local"))
    assert (event.global_request_id, event.local_request_id) == (R1, "collector-local")


@pytest.mark.parametrize("message", [
    f"ERROR failure accessing {R1}",
    f"ERROR failed [{R1}]",
    f"ERROR handler failed [{R1}]",
    f'ERROR Audit rejected body {{"global_request_id":"{R1}"}}: request header missing',
    f"ERROR failed\n2026-11-15 10:00:00.123 121 ERROR nova.api [{R1}] quoted traceback",
    f"ERROR failed\nValueError: quoted global_request_id={R1}",
])
def test_request_id_in_diagnostic_body_is_not_automatically_request_context(message):
    assert normalize_hit(hit(message)).request_id is None


def test_configured_id_paths_read_nested_and_flattened_sources():
    event = normalize_hit(
        hit(correlation={"external": R1}, **{"correlation.operation": R2}),
        global_request_id_fields=("correlation.external.keyword",),
        local_request_id_fields=("correlation.operation",),
    )
    assert (event.global_request_id, event.local_request_id, event.request_id_kind) == (R1, R2, "global")


def test_explicit_alias_classification_supersedes_default_namespace():
    event = normalize_hit(hit(request_id=R1), global_request_id_fields=("request_id.keyword",))
    assert (event.global_request_id, event.local_request_id) == (R1, None)


def test_structured_log_level_dominates_embedded_traceback():
    event = normalize_hit(hit("[parameters: {'id': 'dynamic'}]\n2026-11-15 10:00:00.123 24 ERROR keystone SQL error", log_level="CRITICAL", request_id=R1.removeprefix("req-")))
    assert event.level == "CRITICAL"
    assert event.request_id == R1.removeprefix("req-")
    assert event.local_request_id == R1.removeprefix("req-")
    assert group_errors([event]) == []


def test_configured_timestamp_priority_and_python_module():
    value = hit(**{"observed": {"time": "2026-11-16T00:00:00Z"}, "python_module": "keystone"})
    del value["_source"]["Logger"]
    event = normalize_hit(value, timestamp_field="observed.time")
    assert event.timestamp.day == 16
    assert event.service == "keystone"


@pytest.mark.parametrize("encode", [int, str])
def test_epoch_millis_timestamp(encode):
    expected = datetime(2026, 11, 15, 7, 0, 0, 123000, tzinfo=timezone.utc)
    timestamp = encode(int(expected.timestamp() * 1000))
    assert normalize_hit(hit(**{"@timestamp": timestamp})).timestamp == expected


def test_global_and_explicit_non_uuid_local_both_preserved():
    event = normalize_hit(hit("ERROR global_request_id=global-123 request_id=local-456 failed"))
    assert event.global_request_id == "global-123"
    assert event.local_request_id == "local-456"


def test_request_count_uuid_aliases():
    a = normalize_hit(hit("ERROR failed", request_id=R1))
    b = a.model_copy(update={"request_id": R1.removeprefix("req-").upper(), "evidence_id": "a/2"})
    assert group_errors([a, b])[0].request_count == 1


def test_request_count_keeps_global_and_local_namespaces_separate():
    a = normalize_hit(hit("ERROR failed", global_request_id=R1))
    value = hit("ERROR failed", request_id=R1.removeprefix("req-").upper())
    value["_id"] = "2"
    b = normalize_hit(value)
    assert group_errors([a, b])[0].request_count == 2


def test_fingerprint_uses_central_algorithm_version():
    event = normalize_hit(hit("ERROR failure"))
    group = group_errors([event])[0]
    expected = hashlib.sha256(json.dumps([ALGORITHM_VERSION, group.service, group.template], ensure_ascii=False).encode()).hexdigest()
    assert group.fingerprint == expected


def test_numeric_syslog_severity_text():
    event = normalize_hit(hit("a message", severity_text=3))
    assert event.level == "ERROR"


@pytest.mark.parametrize("invalid", [" ", {"raw": "not a severity"}, "UNKNOWN"])
def test_invalid_priority_fields_fall_back_to_valid_compatible_fields(invalid):
    event = normalize_hit(hit("failed", log_level=invalid, level=" eRrOr ", service={"metadata": True}))
    assert (event.level, event.service) == ("ERROR", "openstack.nova")


def test_unusable_flattened_field_falls_back_to_nested_value():
    event = normalize_hit(hit("failed", **{"log.level": " ", "log": {"level": "ERROR"}}))
    assert event.level == "ERROR"


def test_empty_context_prefix_groups_with_parsed_message_and_keeps_raw():
    message = "The broker has blocked the connection: connection blocked, see broker logs"
    a = normalize_hit(hit("[-] " + message, log_level="ERROR"))
    b_hit = hit(message, log_level="ERROR", request_id=R1)
    b_hit["_id"] = "2"
    b = normalize_hit(b_hit)
    groups = group_errors([a, b])
    assert len(groups) == 1
    assert groups[0].count == 2
    assert groups[0].representative.evidence_id == b.evidence_id
    assert a.message == "[-] " + message
    assert b.message == message


def test_uuid_is_not_request():
    assert normalize_hit(hit("ERROR instance 12345678-1234-1234-1234-123456789abc failed")).request_id is None


def test_logging_error_and_quoted_error_not_error():
    assert normalize_hit(hit("--- Logging error ---")).level == "UNKNOWN"
    assert normalize_hit(hit("INFO reported ERROR in previous request")).level == "INFO"


def test_structured_fields():
    event = normalize_hit({"_index": "a", "_id": "b", "_source": {"@timestamp": "2026-11-15T00:00:00Z", "message": "failure", "log": {"level": "error"}, "service": {"name": "api"}, "host": {"name": "lab"}, "global_request_id": R1}})
    assert (event.level, event.service, event.host, event.request_id_kind) == ("ERROR", "api", "lab", "global")


def test_uwsgi():
    event = normalize_hit(hit("GET /healthcheck => generated 2 bytes in 1 msecs (HTTP/1.1 200) 3 headers"))
    assert (event.http_method, event.http_uri, event.http_status, event.duration_ms) == ("GET", "/healthcheck", 200, 1)
    assert event.level != "ERROR"


@pytest.mark.parametrize("change", [{"@timestamp": "invalid"}, {"@timestamp": "2026-11-15T10:00:00"}])
def test_bad_timestamp(change):
    with pytest.raises(ValueError):
        normalize_hit(hit(**change))


def test_naive_timestamp_does_not_hide_timezone_aware_alias():
    event = normalize_hit(hit(**{"@timestamp": "2026-11-15T10:00:00", "timestamp": "2026-11-15T10:00:00+03:00"}))
    assert event.timestamp.isoformat() == "2026-11-15T10:00:00+03:00"


def test_missing_evidence_id():
    value = hit()
    del value["_id"]
    with pytest.raises(ValueError):
        normalize_hit(value)


def test_grouping_keeps_codes_and_services_but_removes_request_pid():
    a = normalize_hit(hit(f"2026-11-15 10:00:00.123 121 ERROR nova.api [{R1}] HTTP 404 ValueError"))
    b_hit = hit(f"2026-11-15 10:01:00.456 999 ERROR nova.api [{R2}] HTTP 404 ValueError")
    b_hit["_id"] = "2"
    b = normalize_hit(b_hit)
    c = b.model_copy(update={"evidence_id": "a/3", "message": b.message.replace("404", "500")})
    d = b.model_copy(update={"evidence_id": "a/4", "service": "other"})
    groups = group_errors([b, a, c, d, a])
    assert len(groups) == 3
    assert groups[0].count == 2
    assert groups[0].request_count == 2
    assert group_errors([a, b])[0].fingerprint == group_errors([b, a])[0].fingerprint


def test_representative_prefers_id_and_context_preserves_chain():
    a = normalize_hit(hit("ERROR operation failed"))
    b = a.model_copy(update={"evidence_id": "a/2", "request_id": R1, "request_id_kind": "local"})
    group = group_errors([a, b])[0]
    assert group.representative.request_id == R1
    chain = ChainReport(request_id=R1, status="retrieved", total=2, events=[b, a])
    before = deepcopy(chain)
    context = build_llm_context(group, chain)
    assert len(context["evidence"]) == 1
    summary = context["evidence"][0]
    assert summary["count"] == 2
    assert summary["sample_doc_id"] == b.evidence_id
    assert context["analysis_status"] == "not_requested"
    assert chain == before


def test_bounded_context_marks_clipping():
    a = normalize_hit(hit("ERROR " + "x" * 10000))
    group = group_errors([a])[0]
    context = build_llm_context(group, ChainReport(request_id=None, status="missing_request_id", events=[a], total=1))
    assert context["context_truncated"]
    assert context["evidence"][0]["message_truncated"]
    assert len(context["evidence"][0]["message"]) == 8000


@pytest.mark.parametrize("quote", ['"', "'"])
def test_quoted_explicit_global_and_local_ids(quote):
    event = normalize_hit(hit(f"ERROR {{{quote}global_request_id{quote}: {quote}{R1}{quote}, {quote}request_id{quote}: {quote}{R2}{quote}}}"))
    assert event.request_id_kind == "global"
    assert event.global_request_id == R1
    assert event.local_request_id == R2


def test_explicit_global_only_not_duplicated_as_local():
    event = normalize_hit(hit(f'ERROR {{"global_request_id": "{R1}"}}'))
    assert event.global_request_id == R1
    assert event.local_request_id is None


def test_explicit_global_token_not_truncated_at_uuid_prefix():
    event = normalize_hit(hit(f"ERROR global_request_id={R1}X failed"))
    assert event.global_request_id == R1 + "X"


def test_explicit_id_template_masks_non_uuid_but_retains_status_and_limit():
    a = normalize_hit(hit('ERROR "request_id": "local-123" HTTP 404 limit 72'))
    value = hit('ERROR "request_id": "local-456" HTTP 404 limit 72')
    value["_id"] = "2"
    b = normalize_hit(value)
    c = b.model_copy(update={"evidence_id": "a/3", "message": b.message.replace("404", "500")})
    groups = group_errors([a, b, c])
    assert len(groups) == 2
    assert groups[0].count == 2
    assert "404" in groups[0].template and "72" in groups[0].template


def test_context_keeps_error_and_both_timeline_endpoints_under_limit():
    error = normalize_hit(hit("ERROR failed"))
    early = [error.model_copy(update={"evidence_id": f"a/info-{i:03}", "timestamp": error.timestamp - timedelta(seconds=200-i), "level": "INFO", "message": f"INFO work {i}"}) for i in range(150)]
    group = group_errors([error])[0]
    chain = ChainReport(request_id=None, status="retrieved", events=early+[error], total=151)
    context = build_llm_context(group, chain)
    ids = [record["evidence_id"] for record in context["evidence"]]
    assert error.evidence_id in ids
    assert early[0].evidence_id in ids
    assert len(ids) <= 100 and len(ids) == len(set(ids))
    assert context["omitted_evidence_count"] == 151-len(ids)
    assert context["context_truncated"]
    assert ids[-1] == error.evidence_id
    assert chain.events == early + [error]


def test_context_aggregates_repeats_by_service_and_level_with_exact_counts():
    error = normalize_hit(hit("ERROR failed"))
    group = group_errors([error])[0]
    first = error.model_copy(update={"evidence_id": "a/earlier", "level": "WARNING", "message": "pid=123 retry",
                                     "timestamp": error.timestamp + timedelta(seconds=1)})
    last = first.model_copy(update={"evidence_id": "a/later", "timestamp": first.timestamp + timedelta(seconds=30),
                                    "message": "pid=432 retry"})
    other = last.model_copy(update={"evidence_id": "a/different-service", "service": "neutron",
                                    "timestamp": last.timestamp + timedelta(seconds=1)})
    chain = ChainReport(request_id=None, status="retrieved", events=[error, last, first, other], total=4)
    context = build_llm_context(group, chain)
    assert context["represented_event_count"] == 4
    assert context["included_evidence_count"] == 3
    assert context["omitted_evidence_count"] == 0
    assert not context["context_truncated"]
    repeat, = [row for row in context["evidence"] if row["count"] == 2]
    assert repeat["sample_doc_id"] == first.evidence_id
    assert repeat["template"] == "pid=<id> retry"
    assert repeat["first_seen"] == first.timestamp.isoformat()
    assert repeat["last_seen"] == last.timestamp.isoformat()


def test_context_long_error_retains_exception_tail():
    event = normalize_hit(hit("ERROR start " + "x" * 10000 + " ValueError: final reason"))
    group = group_errors([event])[0]
    context = build_llm_context(group, ChainReport(request_id=None, status="retrieved", events=[event], total=1))
    assert context["evidence"][0]["message"].endswith("ValueError: final reason")
    assert context["truncated_message_count"] == 1


def test_context_keeps_repeated_operations_on_both_sides_of_error():
    error = normalize_hit(hit("ERROR failed"))
    before = error.model_copy(update={"evidence_id": "before", "level": "INFO", "message": "retry",
                                      "timestamp": error.timestamp - timedelta(seconds=1), "http_uri": "/" * 50_000})
    after = before.model_copy(update={"evidence_id": "after", "timestamp": error.timestamp + timedelta(seconds=1)})
    group = group_errors([error])[0]
    interleaved = after.model_copy(update={"evidence_id": "other", "message": "woke up",
                                          "timestamp": after.timestamp + timedelta(seconds=1)})
    repeated = after.model_copy(update={"evidence_id": "later", "timestamp": after.timestamp + timedelta(seconds=2)})
    chain = ChainReport(request_id=None, status="retrieved", events=[before, error, after, interleaved, repeated], total=5)
    evidence = build_llm_context(group, chain)["evidence"]
    assert [row["phase"] for row in evidence] == ["before_error", "at_error", "after_error", "after_error", "after_error"]
    assert [row["message"] for row in evidence[2:]] == ["retry", "woke up", "retry"]
    assert all("http_uri" not in row for row in evidence)


def test_hexadecimal_codes_and_non_http_durations_keep_their_meaning():
    first = normalize_hit(hit("ERROR code 0x400000000 in 21 msecs"))
    second = normalize_hit(hit("ERROR code 0x800000000 in 21 msecs"))
    second.evidence_id += "-second"
    assert first.duration_ms is None
    assert len(group_errors([first, second])) == 2


@pytest.mark.parametrize("fields", [{"event": {"original": "ERROR original"}}, {"log": "ERROR scalar"}, {"log": {"message": "ERROR nested", "level": "ERROR"}}])
def test_additional_message_schema_aliases(fields):
    value = hit()
    del value["_source"]["Payload"]
    value["_source"].update(fields)
    event = normalize_hit(value)
    assert event.level == "ERROR"
    assert event.message.startswith("ERROR")
