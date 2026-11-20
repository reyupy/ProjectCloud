"""Bounded, read-only OpenSearch access (TLS validation stays enabled)."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, build_opener, HTTPRedirectHandler

from .models import FetchResult, TimeWindow
from .version import ALGORITHM_VERSION, SOURCE_QUERY_VERSION

_LEVEL_FIELDS = ("log_level", "log.level", "level", "Level", "severity_text", "severity", "Severity")
_MESSAGE_FIELDS = ("Payload", "message", "Message", "log.message", "log", "event.original")


def _with_keyword_fields(names: tuple[str, ...]) -> tuple[str, ...]:
    bases = tuple(dict.fromkeys(name.removesuffix(".keyword") for name in names))
    return (*bases, *(name + ".keyword" for name in bases))


_STANDARD_REQUEST_ID_FIELDS = tuple(
    prefix + name
    for prefix in ("", "context.", "request.", "Fields.")
    for name in ("global_request_id", "global-request-id", "request_id", "request-id")
)
_GLOBAL_ID_NAMES = {"global_request_id", "global-request-id"}
_REQUEST_UUID = re.compile(r"(?:req-)?([0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12})", re.I)
# Lucene requires literal whitespace characters inside this character class.
_SPACE = "[ \t\n\r\f\v\x1c\x1d\x1e\x1f\u0085\u00a0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]"
_ERROR_WORD = "[Ee][Rr][Rr]([Oo][Rr])?"


def _regex_candidates(pattern: str, names: tuple[str, ...]) -> dict:
    # Query-string regexp terms support leniency per field on mixed mappings
    # (e.g. keyword severity in one index, numeric severity in another).
    return {"query_string": {"query": "/" + pattern + "/", "fields": list(names), "analyzer": "keyword",
                             "lenient": True, "rewrite": "constant_score"}}


class SourceError(RuntimeError):
    """A source request failed; messages intentionally exclude response bodies."""


class RetryableSourceError(SourceError):
    """The scan can be restarted safely from its saved window."""


@dataclass(frozen=True)
class SourceConfig:
    url: str
    index_pattern: str = "flog-*"
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    timeout_seconds: float = 15
    page_size: int = 500
    max_errors: int = 10000
    max_chain_events: int = 2000
    timestamp_field: str = "@timestamp"
    request_id_fields: tuple[str, ...] = _with_keyword_fields(_STANDARD_REQUEST_ID_FIELDS)
    global_request_id_fields: tuple[str, ...] = ()
    local_request_id_fields: tuple[str, ...] = ()
    timestamp_fallback_fields: tuple[str, ...] = ()

    def __post_init__(self):
        try:
            parts = urlsplit(self.url)
            valid = parts.scheme in ("http", "https") and parts.hostname and parts.port != 0
        except ValueError:
            valid = False
        if not valid or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("OpenSearch URL must be http(s), without credentials, query or fragment")
        if not re.fullmatch(r"[a-z0-9_*.,-]+", self.index_pattern) or ".." in self.index_pattern:
            raise ValueError("invalid index pattern")
        if not math.isfinite(self.timeout_seconds) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("timeout_seconds must be between 0 and 300")
        for name in ("page_size", "max_errors", "max_chain_events"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.page_size > 10000:
            raise ValueError("page_size must not exceed 10000")
        for name in (*self.timestamp_fields, *self.request_id_fields, *self.global_request_id_fields,
                     *self.local_request_id_fields):
            if not re.fullmatch(r"[@A-Za-z_][@A-Za-z0-9_.-]*", name):
                raise ValueError("invalid field name")
        global_fields = {name.removesuffix(".keyword") for name in self.global_request_id_fields}
        local_fields = {name.removesuffix(".keyword") for name in self.local_request_id_fields}
        if global_fields & local_fields:
            raise ValueError("a request ID field cannot be configured as both global and local")
        if (self.username is None) != (self.password is None):
            raise ValueError("both username and password are required")

    @property
    def timestamp_fields(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.timestamp_field, *self.timestamp_fallback_fields)))

    @property
    def normalization_global_request_id_fields(self) -> tuple[str, ...]:
        return self._normalization_request_fields(global_ids=True)

    @property
    def normalization_local_request_id_fields(self) -> tuple[str, ...]:
        return self._normalization_request_fields(global_ids=False)

    def _normalization_request_fields(self, *, global_ids: bool) -> tuple[str, ...]:
        explicit_global = dict.fromkeys(name.removesuffix(".keyword") for name in self.global_request_id_fields)
        explicit_local = dict.fromkeys(name.removesuffix(".keyword") for name in self.local_request_id_fields)
        explicit = explicit_global if global_ids else explicit_local
        other = explicit_local if global_ids else explicit_global
        for configured in self.request_id_fields:
            name = configured.removesuffix(".keyword")
            if name in _STANDARD_REQUEST_ID_FIELDS or name in other:
                continue
            if (name.rsplit(".", 1)[-1] in _GLOBAL_ID_NAMES) == global_ids:
                explicit.setdefault(name, None)
        return tuple(explicit)


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward logs or authorization to an unexpected destination.
        raise SourceError("OpenSearch redirect refused")


class OpenSearchSource:
    def __init__(self, config: SourceConfig):
        self.config = config
        self._opener = build_opener(_NoRedirect())

    @property
    def source_key(self) -> str:
        parts = urlsplit(self.config.url)
        public_url = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))
        # Include scope configuration; accounts can have distinct document visibility.
        identity = json.dumps(["algorithm", ALGORITHM_VERSION, "query", SOURCE_QUERY_VERSION,
                               public_url, self.config.index_pattern, self.config.timestamp_fields,
                               self.config.request_id_fields, self.config.global_request_id_fields,
                               self.config.local_request_id_fields, self.config.username], sort_keys=True)
        return "opensearch:" + hashlib.sha256(identity.encode()).hexdigest()[:24]

    def _request(self, method: str, path: str, body: dict | None = None, *, retry: bool = False) -> dict:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.config.username is not None:
            token = base64.b64encode(f"{self.config.username}:{self.config.password}".encode()).decode()
            headers["Authorization"] = "Basic " + token
        request = Request(self.config.url.rstrip("/") + path,
                          data=None if body is None else json.dumps(body).encode(), headers=headers, method=method)
        for attempt in range(3 if retry else 1):
            try:
                with self._opener.open(request, timeout=self.config.timeout_seconds) as response:
                    result = json.load(response)
                if not isinstance(result, dict):
                    raise SourceError("OpenSearch returned an invalid response")
                return result
            except HTTPError as exc:
                code = exc.code
                exc.close()
                if retry and code in (429, 502, 503, 504) and attempt < 2:
                    time.sleep(0.2 * 2**attempt)
                    continue
                error = RetryableSourceError if code in (408, 429, 502, 503, 504) else SourceError
                raise error(f"OpenSearch HTTP {code}") from None
            except (URLError, TimeoutError, OSError):
                # Network failures may happen after scroll advancement; don't retry cursors.
                raise RetryableSourceError("OpenSearch connection failed (check URL, VPN and credentials)") from None
            except (ValueError, UnicodeError):
                raise SourceError("OpenSearch returned invalid JSON") from None
        raise SourceError("OpenSearch request failed")

    @staticmethod
    def _check_response(response: dict) -> None:
        if response.get("error"):
            raise SourceError("OpenSearch search failed; results are incomplete")
        if response.get("timed_out"):
            raise RetryableSourceError("OpenSearch search timed out; results are incomplete")
        shards = response.get("_shards", {})
        if shards.get("failed", 0):
            raise RetryableSourceError("OpenSearch shard failure; results are incomplete")

    def _index_path(self) -> str:
        return "/" + quote(self.config.index_pattern, safe="*,-_")

    def latest_timestamp(self, window: TimeWindow | None = None, *, before: datetime | None = None) -> datetime | None:
        if before is not None and (before.tzinfo is None or before.utcoffset() is None):
            raise ValueError("timestamp upper bound must include a timezone")
        bounds = self._bounds(window)
        if before is not None:
            bounds["lte"] = before.isoformat()
        timestamps = (self._latest_for_field(field, bounds, preceding)
                      for field, preceding in self._time_fields())
        return max((stamp for stamp in timestamps if stamp is not None), default=None)

    def _latest_for_field(self, field, bounds, preceding) -> datetime | None:
        body = {
            "size": 0, "aggs": {"latest": {"max": {"field": field}}}
        }
        if bounds or preceding:
            body["query"] = self._time_filter(field, bounds, preceding)
        response = self._request("POST", self._index_path() + "/_search", body, retry=True)
        self._check_response(response)
        try:
            latest = response["aggregations"]["latest"]
            if latest["value"] is None:
                return None
            if latest.get("value_as_string"):
                value = datetime.fromisoformat(latest["value_as_string"].replace("Z", "+00:00"))
                if value.tzinfo is None:
                    raise ValueError
                return value
            return datetime.fromtimestamp(latest["value"] / 1000, tz=timezone.utc)
        except (KeyError, TypeError, ValueError, OverflowError):
            raise SourceError("OpenSearch latest timestamp response is invalid") from None

    def _time_fields(self):
        fields = self.config.timestamp_fields
        return ((field, fields[:position]) for position, field in enumerate(fields))

    @staticmethod
    def _bounds(window: TimeWindow | None) -> dict:
        return {"gte": window.start.isoformat(), "lt": window.end.isoformat()} if window else {}

    @staticmethod
    def _time_filter(field, bounds, preceding) -> dict:
        query = {"range": {field: bounds}} if bounds else {"exists": {"field": field}}
        if preceding:
            return {"bool": {"filter": [query], "must_not": [{"exists": {"field": name}} for name in preceding]}}
        return query

    def _range(self, window: TimeWindow) -> dict:
        bounds = self._bounds(window)
        clauses = [self._time_filter(field, bounds, preceding) for field, preceding in self._time_fields()]
        return clauses[0] if len(clauses) == 1 else {"bool": {"should": clauses, "minimum_should_match": 1}}

    def fetch_errors(self, window: TimeWindow) -> FetchResult:
        level_fields = _with_keyword_fields(_LEVEL_FIELDS)
        message_fields = _with_keyword_fields(_MESSAGE_FIELDS)
        candidates = [
            {"multi_match": {"query": level, "fields": [*level_fields, *message_fields], "lenient": True}}
            for level in ("ERROR", "error", "Error", "ERR", "err", "Err")
        ]
        candidates.extend((
            # App levels are exact normalized values. Restrict numeric matches
            # to those fields to avoid scanning ordinary messages containing 3.
            {"multi_match": {"query": "3", "fields": list(level_fields), "lenient": True}},
            _regex_candidates(_SPACE + "*(" + _ERROR_WORD + "|3)" + _SPACE + "*", level_fields),
            # Analyzed fields match above. This branch also finds error headers
            # inside a single keyword token. Normalization validates the header.
            _regex_candidates(r"(.*[^a-zA-Z0-9_])?" + _ERROR_WORD + r"((\]|:|" + _SPACE + ").*)?", message_fields),
        ))
        query = {"bool": {"filter": [self._range(window)], "minimum_should_match": 1, "should": candidates}}
        return self._scan(query, self.config.max_errors)

    def fetch_chain(self, request_id: str, window: TimeWindow, service: str | None = None) -> FetchResult:
        if not request_id or len(request_id) > 256 or not re.fullmatch(r"[A-Za-z0-9_.:-]+", request_id):
            raise ValueError("invalid request ID")
        if service is not None and (not service or len(service) > 512):
            raise ValueError("invalid service")
        parsed_uuid = _REQUEST_UUID.fullmatch(request_id)
        aliases = (request_id,)
        # Dot is the only permitted ID character with special meaning here.
        pattern = request_id.replace(".", r"\.")
        if parsed_uuid:
            bare = parsed_uuid[1]
            lower, upper = bare.lower(), bare.upper()
            aliases = tuple(dict.fromkeys((request_id, lower, upper, "req-" + lower, "REQ-" + upper)))
            # Canonicalize UUID case even for non-normalized keyword fields.
            pattern = "([Rr][Ee][Qq]-)?" + "".join(
                f"[{char}{char.upper()}]" if char.isalpha() else char for char in lower
            )
        request_fields = _with_keyword_fields((
            *_STANDARD_REQUEST_ID_FIELDS, *self.config.request_id_fields,
            *self.config.global_request_id_fields, *self.config.local_request_id_fields,
        ))
        message_fields = _with_keyword_fields(_MESSAGE_FIELDS)
        candidates = [
            {"multi_match": {"query": alias, "type": "phrase", "fields": [*request_fields, *message_fields], "lenient": True}}
            for alias in aliases
        ]
        candidates.extend((
            _regex_candidates(_SPACE + "*" + pattern + _SPACE + "*", request_fields),
            _regex_candidates(".*" + pattern + ".*", message_fields),
        ))
        # The pipeline applies exact service scope and ID namespace after
        # normalization. An indexed service filter can lose valid candidates
        # when a collector pads the value or derives it from a secondary field.
        query = {"bool": {"filter": [self._range(window)], "minimum_should_match": 1, "should": candidates}}
        return self._scan(query, self.config.max_chain_events)

    def _scan(self, query: dict, limit: int) -> FetchResult:
        scroll_id = None
        warnings = []
        result = None
        try:
            response = self._request("POST", self._index_path() + "/_search?scroll=1m", {
                "query": query, "size": min(self.config.page_size, limit),
                "sort": ["_doc"], "track_total_hits": True,
            }, retry=True)
            hits = []
            seen = set()
            total = None
            while True:
                scroll_id = response.get("_scroll_id", scroll_id)
                self._check_response(response)
                try:
                    block = response["hits"]
                    if total is None:
                        raw_total = block["total"]
                        if isinstance(raw_total, dict):
                            if raw_total.get("relation", "eq") != "eq":
                                raise SourceError("OpenSearch returned an inexact hit count")
                            total = int(raw_total["value"])
                        else:
                            total = int(raw_total)
                    page = block["hits"]
                    if not isinstance(page, list):
                        raise ValueError
                    for hit in page:
                        key = (hit["_index"], hit["_id"])
                        if key in seen:
                            raise SourceError("OpenSearch scroll repeated a document")
                        seen.add(key)
                        if len(hits) < limit:
                            hits.append(hit)
                except (KeyError, TypeError, ValueError):
                    raise SourceError("OpenSearch hit response is invalid") from None
                if len(hits) >= min(total, limit):
                    break
                if not page or not scroll_id:
                    raise SourceError("OpenSearch scan ended before all documents were retrieved")
                response = self._request("POST", "/_search/scroll", {"scroll": "1m", "scroll_id": scroll_id})
            result = FetchResult(hits=hits, total=total, truncated=total > limit, warnings=warnings)
            return result
        finally:
            if scroll_id:
                try:
                    self._request("DELETE", "/_search/scroll", {"scroll_id": [scroll_id]})
                except SourceError:
                    if result is not None:
                        result.warnings.append("Scroll cleanup failed; cursor expires automatically")
