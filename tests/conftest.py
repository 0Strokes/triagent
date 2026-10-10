"""Shared fixtures. Tests never touch HackerOne, Claude, or the embedding model."""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import common  # noqa: E402


@pytest.fixture
def conn():
    """In-memory SQLite database with the real schema."""
    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.executescript(common.SCHEMA)
    yield db
    db.close()


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    """Make retry/backoff/countdown loops instant."""
    monkeypatch.setattr("time.sleep", lambda *_: None)


@pytest.fixture
def report():
    return {
        "id": 4242,
        "title": "Stored XSS in display name",
        "description": "Payload executes when another user views the profile.",
        "vulnerability_type": "Cross-site Scripting (XSS) - Stored",
        "severity": "high",
        "reporter": "researcher",
        "reporter_reputation": 120.0,
        "reporter_signal": 4.1,
    }


@pytest.fixture
def verdict():
    return {
        "fp_score": 95,
        "reasoning": ["Matches previously closed reports."],
        "suggested_state": "informative",
        "suggested_comment": "Thank you for your report.",
        "similar_report_ids": [],
    }
