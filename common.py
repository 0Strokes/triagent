"""Shared utilities for triagent: config loading, SQLite schema, and the
HackerOne API client. All entry points (ingest, index, classify) import from
here so configuration and storage stay consistent.
"""
from __future__ import annotations

import os
import re
import sqlite3
import time
import warnings
from pathlib import Path
from typing import Any, Iterator

# Silence the harmless LibreSSL notice urllib3 emits on macOS' system Python.
# Must run before `import requests` (which imports urllib3).
warnings.filterwarnings("ignore", message=r"urllib3 v2 only supports OpenSSL")

import requests
import yaml
from rich.markup import escape

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"
STORE_DIR = BASE_DIR / "store"
DB_PATH = STORE_DIR / "reports.db"
EMBEDDINGS_PATH = STORE_DIR / "embeddings.npy"
EMBEDDING_MAP_PATH = STORE_DIR / "embedding_map.json"

H1_API_BASE = "https://api.hackerone.com/v1"

# States considered "closed" / historical for ingestion.
CLOSED_STATES = ["resolved", "not-applicable", "informative", "duplicate", "spam"]
# States considered "pending" / awaiting triage for classification.
PENDING_STATES = ["new", "triaged"]


# ---------------------------------------------------------------------------
# Untrusted text
# ---------------------------------------------------------------------------
# C0/C1 control characters except tab and newline: ESC-based terminal sequences
# (screen clears, cursor moves, OSC 8 hyperlinks), carriage returns, etc.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def strip_control(value: Any) -> str:
    """Remove terminal control characters from untrusted text."""
    return _CONTROL_CHARS.sub("", str(value))


def safe(value: Any) -> str:
    """Make untrusted text (report fields, model output, API errors) safe to
    print with rich: strip control characters and escape rich markup, so a
    report can't fake lines in the UI or hide text from the reviewer.
    """
    return escape(strip_control(value))


def safe_line(value: Any) -> str:
    """Like safe(), but also flattens newlines for single-line fields (titles,
    usernames) so they can't push fake lines into the UI."""
    return " ".join(safe(value).split())


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and minimally validate config.yaml."""
    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found at {path}. Copy the template and fill it in."
        )
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}

    # Allow the Anthropic key to come from the environment as a fallback.
    anthropic_cfg = config.setdefault("anthropic", {})
    if not anthropic_cfg.get("api_key"):
        anthropic_cfg["api_key"] = os.environ.get("ANTHROPIC_API_KEY", "")

    return config


def require_h1_credentials(config: dict[str, Any]) -> tuple[str, str, str]:
    """Return (identifier, token, program_handle) or raise a clear error."""
    h1 = config.get("hackerone", {})
    identifier = h1.get("api_identifier", "")
    token = h1.get("api_token", "")
    handle = h1.get("program_handle", "")
    missing = [
        name
        for name, value in (
            ("api_identifier", identifier),
            ("api_token", token),
            ("program_handle", handle),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing HackerOne config value(s): "
            + ", ".join(missing)
            + ". Set them in config.yaml under 'hackerone'."
        )
    return identifier, token, handle


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id              INTEGER PRIMARY KEY,
    title           TEXT,
    description     TEXT,
    vulnerability_type TEXT,
    severity        TEXT,
    state           TEXT,
    reporter        TEXT,
    reporter_reputation REAL,
    reporter_signal     REAL,
    created_at      TEXT,
    closed_at       TEXT,
    team_comments   TEXT,        -- newline-joined team activity comments
    indexed         INTEGER NOT NULL DEFAULT 0  -- 1 once an embedding exists
);

CREATE TABLE IF NOT EXISTS classifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    report_id       INTEGER NOT NULL,
    fp_score        INTEGER,
    suggested_state TEXT,
    suggested_comment TEXT,
    reasoning       TEXT,        -- JSON array
    similar_report_ids TEXT,     -- JSON array
    action_taken    TEXT,        -- approved | skipped | discarded | auto-approved | dry-run
    final_state     TEXT,        -- the state the report was actually closed as (if any)
    classified_at   TEXT,
    mode            TEXT         -- interactive | auto-approve | dry-run
);

-- Tracks reports the user has locally dismissed ([d]iscard) so we don't resurface them.
CREATE TABLE IF NOT EXISTS reviewed (
    report_id       INTEGER PRIMARY KEY,
    reviewed_at     TEXT
);

CREATE INDEX IF NOT EXISTS idx_reports_state ON reports(state);
CREATE INDEX IF NOT EXISTS idx_reports_indexed ON reports(indexed);
CREATE INDEX IF NOT EXISTS idx_classifications_report ON classifications(report_id);
"""


def get_db(path: Path = DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the SQLite database with the schema applied."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# HackerOne API client
# ---------------------------------------------------------------------------
class HackerOneError(RuntimeError):
    """Raised for non-retryable HackerOne API failures."""


class HackerOneClient:
    """Thin wrapper over the HackerOne REST API with retries and pagination.

    Auth is HTTP Basic: api_identifier as username, api_token as password.
    """

    def __init__(
        self,
        api_identifier: str,
        api_token: str,
        program_handle: str,
        max_retries: int = 4,
        timeout: float = 30.0,
    ):
        self.program_handle = program_handle
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.auth = (api_identifier, api_token)
        self.session.headers.update({"Accept": "application/json"})

    # -- low level ----------------------------------------------------------
    def _request(self, method: str, url: str, **kwargs) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(
                    method, url, timeout=self.timeout, **kwargs
                )
            except requests.RequestException as exc:
                last_exc = exc
                self._backoff(attempt)
                continue

            if resp.status_code in (429, 500, 502, 503, 504):
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else self._backoff_delay(attempt)
                time.sleep(wait)
                last_exc = HackerOneError(
                    f"{resp.status_code} from {url}: {resp.text[:200]}"
                )
                continue

            if resp.status_code == 401:
                raise HackerOneError(
                    "HackerOne returned 401 Unauthorized. Check api_identifier / api_token."
                )
            if not resp.ok:
                raise HackerOneError(
                    f"{resp.status_code} from {url}: {resp.text[:500]}"
                )
            if resp.status_code == 204 or not resp.text:
                return {}
            return resp.json()

        raise HackerOneError(
            f"Request to {url} failed after {self.max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        return min(2.0 ** attempt, 30.0)

    def _backoff(self, attempt: int) -> None:
        time.sleep(self._backoff_delay(attempt))

    # -- reports ------------------------------------------------------------
    def iter_reports(
        self,
        states: list[str],
        created_after: str | None = None,
        page_size: int = 100,
    ) -> Iterator[dict[str, Any]]:
        """Yield raw report objects for the configured program filtered by state.

        `created_after` is an ISO 8601 timestamp; only reports created strictly
        after it are returned (used for incremental sync).
        """
        # HackerOne expects filter[state][] repeated once per state; requests
        # serializes a list value into exactly that repeated form.
        params: dict[str, Any] = {
            "filter[program][]": self.program_handle,
            "filter[state][]": states,
            "page[size]": page_size,
        }
        if created_after:
            params["filter[created_at__gt]"] = created_after

        url = f"{H1_API_BASE}/reports"
        while url:
            data = self._request("GET", url, params=params)
            params = {}  # subsequent pages use the full `next` link
            for item in data.get("data", []):
                yield item
            url = data.get("links", {}).get("next")
            # Never send our Basic auth credentials anywhere but the H1 API.
            if url and not url.startswith(f"{H1_API_BASE}/"):
                raise HackerOneError(f"Refusing to follow unexpected pagination link: {url}")

    def get_report(self, report_id: int) -> dict[str, Any]:
        url = f"{H1_API_BASE}/reports/{report_id}"
        data = self._request("GET", url)
        return data.get("data", data)

    # -- actions ------------------------------------------------------------
    def post_comment(self, report_id: int, message: str, internal: bool = False) -> dict[str, Any]:
        """Post a (public by default) comment on a report."""
        url = f"{H1_API_BASE}/reports/{report_id}/activities"
        payload = {
            "data": {
                "type": "activity-comment",
                "attributes": {"message": message, "internal": internal},
            }
        }
        return self._request("POST", url, json=payload)

    def change_state(
        self, report_id: int, state: str, message: str
    ) -> dict[str, Any]:
        """Change a report's state. `message` is posted publicly with the change."""
        url = f"{H1_API_BASE}/reports/{report_id}/state_changes"
        payload = {
            "data": {
                "type": "state-change",
                "attributes": {"message": message, "state": state},
            }
        }
        return self._request("POST", url, json=payload)


# ---------------------------------------------------------------------------
# Report normalization
# ---------------------------------------------------------------------------
def normalize_report(raw: dict[str, Any]) -> dict[str, Any]:
    """Flatten a HackerOne JSON:API report object into our storage shape."""
    attrs = raw.get("attributes", {})
    relationships = raw.get("relationships", {})

    # Reporter
    reporter_data = (
        relationships.get("reporter", {}).get("data", {}).get("attributes", {})
    )
    reporter = reporter_data.get("username", "") or ""
    reputation = reporter_data.get("reputation")
    signal = reporter_data.get("signal")

    # Severity (rating lives on the severity relationship)
    severity_attrs = (
        relationships.get("severity", {}).get("data", {}).get("attributes", {})
    )
    severity = severity_attrs.get("rating") or attrs.get("severity_rating") or ""

    # Weakness / vulnerability type
    weakness_attrs = (
        relationships.get("weakness", {}).get("data", {}).get("attributes", {})
    )
    vuln_type = weakness_attrs.get("name") or ""

    # Team comments: public, non-reporter activity comments. These are sent to
    # the LLM as precedent for *other* reports, so exclude:
    #   - internal comments: they could leak into a public suggested comment;
    #   - the reporter's own comments: attacker-controlled prompt-injection text.
    # Fail closed: a comment is kept only if it's explicitly marked non-internal.
    comments: list[str] = []
    activities = relationships.get("activities", {}).get("data", []) or []
    for act in activities:
        a_attrs = act.get("attributes", {})
        if act.get("type") != "activity-comment":
            continue
        if a_attrs.get("internal") is not False:
            continue
        actor = (
            act.get("relationships", {}).get("actor", {}).get("data", {}) or {}
        ).get("attributes", {}) or {}
        if reporter and actor.get("username") == reporter:
            continue
        msg = a_attrs.get("message")
        if msg:
            comments.append(msg)

    return {
        "id": int(raw.get("id")),
        "title": attrs.get("title", "") or "",
        "description": attrs.get("vulnerability_information", "") or "",
        "vulnerability_type": vuln_type,
        "severity": severity,
        "state": attrs.get("state", "") or "",
        "reporter": reporter,
        "reporter_reputation": reputation,
        "reporter_signal": signal,
        "created_at": attrs.get("created_at", "") or "",
        "closed_at": attrs.get("closed_at") or "",
        "team_comments": "\n---\n".join(comments),
    }


def embedding_text(report: dict[str, Any]) -> str:
    """Build the combined text used to embed a report:
    title + vulnerability type + first 500 chars of description.
    """
    title = report.get("title") or ""
    vuln = report.get("vulnerability_type") or ""
    desc = (report.get("description") or "")[:500]
    return f"{title}\n{vuln}\n{desc}".strip()
