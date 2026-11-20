from datetime import datetime, timezone
import re
from unittest.mock import Mock

import pytest

from cloudru_monitor.models import TimeWindow
from cloudru_monitor.source import OpenSearchSource, SourceConfig, SourceError


WINDOW = TimeWindow(start=datetime(2026, 1, 1, tzinfo=timezone.utc), end=datetime(2026, 1, 2, tzinfo=timezone.utc))


def page(ids, total=3, scroll="cursor"):
    return {"_scroll_id": scroll, "_shards": {"failed": 0}, "timed_out": False,
            "hits": {"total": {"value": total, "relation": "eq"},
                     "hits": [{"_index": "flog-1", "_id": i, "_source": {"@timestamp": "2026-01-01T00:00:00Z"}} for i in ids]}}


def test_scroll_handles_equal_timestamps_and_cleans_latest_cursor():
    source = OpenSearchSource(SourceConfig("http://localhost:9200", page_size=2))
    source._request = Mock(side_effect=[page(["a", "b"]), page(["c"], scroll="next"), {}])
    result = source.fetch_errors(WINDOW)
    assert [h["_id"] for h in result.hits] == ["a", "b", "c"]
    assert result.total == 3 and not result.truncated
    calls = source._request.call_args_list
    body = calls[0].args[2]
    assert body["query"]["bool"]["filter"][0]["range"]["@timestamp"] == {
        "gte": WINDOW.start.isoformat(), "lt": WINDOW.end.isoformat()}
    assert body["track_total_hits"] is True
    assert calls[-1].args == ("DELETE", "/_search/scroll", {"scroll_id": ["next"]})
    assert calls[1].kwargs.get("retry") is not True


def test_truncation_is_explicit():
    source = OpenSearchSource(SourceConfig("http://localhost", max_errors=2))
    source._request = Mock(side_effect=[page(["a", "b"], 10), {}])
    result = source.fetch_errors(WINDOW)
    assert result.truncated and result.total == 10 and len(result.hits) == 2


@pytest.mark.parametrize("bad", [{"timed_out": True}, {"_shards": {"failed": 1}}])
def test_incomplete_search_raises_and_cleans(bad):
    source = OpenSearchSource(SourceConfig("http://localhost"))
    response = page(["a"])
    response.update(bad)
    source._request = Mock(side_effect=[response, {}])
    with pytest.raises(SourceError):
        source.fetch_errors(WINDOW)
    assert source._request.call_args.args[0] == "DELETE"


def test_premature_end_is_not_success():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page(["a"]), page([]), {}])
    with pytest.raises(SourceError, match="ended before"):
        source.fetch_errors(WINDOW)


def test_global_chain_query_does_not_filter_service_and_preserves_range():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page([], 0), {}])
    source.fetch_chain("req-123", WINDOW)
    query = source._request.call_args_list[0].args[2]["query"]
    assert query["bool"]["filter"] == [source._range(WINDOW)]
    assert "nova" not in str(query)
    with pytest.raises(ValueError):
        source.fetch_chain("* OR password:*", WINDOW)


def test_local_chain_leaves_normalized_service_check_to_pipeline():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page([], 0), {}])
    source.fetch_chain("req-123", WINDOW, "nova.compute.manager")
    query = source._request.call_args_list[0].args[2]["query"]
    assert query["bool"]["filter"] == [source._range(WINDOW)]
    assert "nova.compute.manager" not in str(query)


@pytest.mark.parametrize("prefix", ["", "req-"])
def test_chain_uuid_aliases_and_all_processing_id_fields(prefix):
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page([], 0), {}])
    bare = "12345678-1234-1234-1234-123456789abc"
    source.fetch_chain(prefix + bare, WINDOW)
    should = source._request.call_args_list[0].args[2]["query"]["bool"]["should"]
    matches = [item["multi_match"] for item in should if "multi_match" in item]
    assert {bare, "req-" + bare} <= {item["query"] for item in matches}
    for name in ("global-request-id", "Fields.global_request_id", "Fields.request_id", "request.request_id", "context.request_id.keyword"):
        assert all(name in item["fields"] for item in matches)


@pytest.mark.parametrize("request_token", [
    "REQ-AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
    "req-aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
    "AAAAAAAA-AAAA-4AAA-8AAA-AAAAAAAAAAAA",
])
def test_uuid_candidates_cover_mixed_case_and_padding_in_keyword_fields(request_token):
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._scan = Mock(return_value=None)
    source.fetch_chain(request_token, WINDOW)
    should = source._scan.call_args.args[0]["bool"]["should"]
    by_field = [item["query_string"] for item in should if "query_string" in item]
    assert all(item["analyzer"] == "keyword" and item["lenient"] for item in by_field)
    structured = next(item for item in by_field if "request_id.keyword" in item["fields"])
    unstructured = next(item for item in by_field if "Payload" in item["fields"])
    lower = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"
    mixed = "ReQ-aAaAaAaA-aAaA-4AaA-8AAa-AaAaAaAaAaAa"
    for value in (lower, "req-" + lower, lower.upper(), "REQ-" + lower.upper(), " \t" + mixed + "\n"):
        assert re.fullmatch(structured["query"][1:-1], value)
    assert "Payload.keyword" in unstructured["fields"]
    assert re.fullmatch(unstructured["query"][1:-1], f"nova.api [ {mixed} ] failed")


def test_non_uuid_ids_remain_case_sensitive():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._scan = Mock(return_value=None)
    source.fetch_chain("Case.Sensitive:ID-1", WINDOW)
    should = source._scan.call_args.args[0]["bool"]["should"]
    values = {item["multi_match"]["query"] for item in should if "multi_match" in item}
    assert values == {"Case.Sensitive:ID-1"}
    structured = next(item["query_string"] for item in should if "query_string" in item)
    regex = structured["query"][1:-1]
    assert re.fullmatch(regex, " Case.Sensitive:ID-1 ")
    assert not re.fullmatch(regex, "case.Sensitive:ID-1")
    assert not re.fullmatch(regex, "CaseXSensitive:ID-1")


def test_error_candidates_support_level_aliases_and_keyword_schemas():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page([], 0), {}])
    source.fetch_errors(WINDOW)
    should = source._request.call_args_list[0].args[2]["query"]["bool"]["should"]
    matches = [item["multi_match"] for item in should if "multi_match" in item]
    assert {m["query"] for m in matches} == {"ERROR", "error", "Error", "ERR", "err", "Err", "3"}
    for name in ("severity_text", "Level", "log_level.keyword", "log.message"):
        assert all(name in item["fields"] for item in matches if item["query"] != "3")
    expressions = [item["query_string"] for item in should if "query_string" in item]
    level = next(item for item in expressions if "log_level" in item["fields"])
    message = next(item for item in expressions if "Payload" in item["fields"])
    for value in ("\t eRrOr ", " ErR", "\u00a03"):
        assert re.fullmatch(level["query"][1:-1], value)
    assert not re.fullmatch(level["query"][1:-1], "not ERROR")
    for value in ("[eRrOr] failed", "2026-01-01 09:00:00.000 123 ErRoR nova.api failed", "ERROR:\tfailed"):
        assert re.fullmatch(message["query"][1:-1], value)
    assert not re.fullmatch(message["query"][1:-1], "terror happened")


@pytest.mark.parametrize("name", ["log_level", "log.level", "level", "Level", "severity_text", "severity", "Severity"])
def test_numeric_error_level_candidates_cover_all_normalized_fields(name):
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._scan = Mock(return_value=None)
    source.fetch_errors(WINDOW)
    should = source._scan.call_args.args[0]["bool"]["should"]
    numeric = next(item["multi_match"] for item in should if item.get("multi_match", {}).get("query") == "3")
    assert name in numeric["fields"] and name + ".keyword" in numeric["fields"]
    assert "Payload" not in numeric["fields"]


def test_custom_request_fields_extend_search_and_respect_explicit_scope():
    config = SourceConfig(
        "http://localhost",
        request_id_fields=("request_id.keyword", "custom.global-request-id", "correlation.id", "collector.legacy.keyword"),
        global_request_id_fields=("correlation.id.keyword",),
        local_request_id_fields=("custom.global-request-id",),
    )
    assert config.normalization_global_request_id_fields == ("correlation.id",)
    assert config.normalization_local_request_id_fields == ("custom.global-request-id", "collector.legacy")
    source = OpenSearchSource(config)
    source._scan = Mock(return_value=None)
    source.fetch_chain("request-123", WINDOW)
    should = source._scan.call_args.args[0]["bool"]["should"]
    fields = should[0]["multi_match"]["fields"]
    assert {"global_request_id", "request_id.keyword", "custom.global-request-id.keyword",
            "correlation.id", "correlation.id.keyword", "collector.legacy", "collector.legacy.keyword"} <= set(fields)
    assert "correlation.id.keyword.keyword" not in fields


def test_nonstandard_legacy_global_alias_is_inferred_without_reclassifying_defaults():
    config = SourceConfig("http://localhost", request_id_fields=("global_request_id.keyword", "otel.global_request_id.keyword"))
    assert config.normalization_global_request_id_fields == ("otel.global_request_id",)
    assert config.normalization_local_request_id_fields == ()


def test_source_key_includes_algorithm_version(monkeypatch):
    source = OpenSearchSource(SourceConfig("http://localhost"))
    previous = source.source_key
    monkeypatch.setattr("cloudru_monitor.source.ALGORITHM_VERSION", "next")
    assert previous != source.source_key


def test_cleanup_failure_returns_warning():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[page([], 0), SourceError("no cleanup")])
    assert "cleanup" in source.fetch_errors(WINDOW).warnings[0]


@pytest.mark.parametrize("kwargs", [
    {"url": "file:///secret"}, {"url": "http://user:secret@localhost"},
    {"url": "http://localhost?a=b"}, {"url": "http://localhost", "index_pattern": "../_all"},
    {"url": "http://localhost", "page_size": 0}, {"url": "http://localhost", "timeout_seconds": float("nan")},
    {"url": "http://localhost", "username": "user"},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        SourceConfig(**kwargs)


def test_source_key_contains_no_secrets_and_is_stable():
    a = OpenSearchSource(SourceConfig("https://example.test", username="user", password="SECRET"))
    b = OpenSearchSource(SourceConfig("https://example.test/", username="user", password="OTHER"))
    assert a.source_key == b.source_key
    assert "SECRET" not in repr(a.config) and "SECRET" not in a.source_key


def test_latest_timestamp_and_empty():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(side_effect=[{"aggregations": {"latest": {"value": None}}},
                                       {"aggregations": {"latest": {"value": 1, "value_as_string": "2026-01-01T00:00:00Z"}}}])
    assert source.latest_timestamp() is None
    assert source.latest_timestamp() == WINDOW.start


def test_latest_timestamp_can_check_coverage_in_exact_window():
    source = OpenSearchSource(SourceConfig("http://localhost", timestamp_field="event.created"))
    source._request = Mock(return_value={"aggregations": {"latest": {"value": None}}})
    assert source.latest_timestamp(WINDOW) is None
    body = source._request.call_args.args[2]
    assert body["query"] == {"range": {"event.created": {"gte": WINDOW.start.isoformat(), "lt": WINDOW.end.isoformat()}}}
    assert body["aggs"]["latest"]["max"]["field"] == "event.created"


def test_latest_timestamp_can_ignore_future_documents():
    source = OpenSearchSource(SourceConfig("http://localhost"))
    source._request = Mock(return_value={"aggregations": {"latest": {"value": None}}})
    assert source.latest_timestamp(before=WINDOW.end) is None
    body = source._request.call_args.args[2]
    assert body["query"] == {"range": {"@timestamp": {"lte": WINDOW.end.isoformat()}}}


def test_timestamp_fallback_uses_consistent_precedence_for_scan_and_latest():
    source = OpenSearchSource(SourceConfig("http://localhost", timestamp_fallback_fields=("timestamp",)))
    query = source._range(WINDOW)["bool"]
    primary, fallback = query["should"]
    assert "@timestamp" in primary["range"]
    assert fallback["bool"]["must_not"] == [{"exists": {"field": "@timestamp"}}]
    assert "timestamp" in fallback["bool"]["filter"][0]["range"]
    source._request = Mock(side_effect=[
        {"aggregations": {"latest": {"value": None}}},
        {"aggregations": {"latest": {"value": 1, "value_as_string": WINDOW.start.isoformat()}}},
    ])
    assert source.latest_timestamp(WINDOW) == WINDOW.start
    assert source._request.call_args.args[2]["query"] == fallback
