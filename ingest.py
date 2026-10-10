"""ingest.py: sync closed HackerOne reports into the local SQLite store.

Usage:
    python ingest.py --sync     # full first sync (all closed reports)
    python ingest.py            # incremental (only reports newer than the last stored one)
"""
from __future__ import annotations

import argparse
import sqlite3
import sys

from rich.console import Console

import common

console = Console()


def latest_created_at(conn: sqlite3.Connection) -> str | None:
    """Return the created_at of the most recent stored report, or None."""
    row = conn.execute(
        "SELECT created_at FROM reports ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    return row["created_at"] if row and row["created_at"] else None


def upsert_report(conn: sqlite3.Connection, report: dict) -> bool:
    """Insert or update a report. Returns True if it was newly inserted.

    Re-syncing an existing report resets `indexed` to 0 only if its content
    changed, so index.py --update will refresh stale embeddings.
    """
    existing = conn.execute(
        "SELECT title, description, vulnerability_type FROM reports WHERE id = ?",
        (report["id"],),
    ).fetchone()

    content_changed = True
    if existing is not None:
        content_changed = (
            existing["title"] != report["title"]
            or existing["description"] != report["description"]
            or existing["vulnerability_type"] != report["vulnerability_type"]
        )

    conn.execute(
        """
        INSERT INTO reports (
            id, title, description, vulnerability_type, severity, state,
            reporter, reporter_reputation, reporter_signal, created_at,
            closed_at, team_comments, indexed
        ) VALUES (
            :id, :title, :description, :vulnerability_type, :severity, :state,
            :reporter, :reporter_reputation, :reporter_signal, :created_at,
            :closed_at, :team_comments, 0
        )
        ON CONFLICT(id) DO UPDATE SET
            title=excluded.title,
            description=excluded.description,
            vulnerability_type=excluded.vulnerability_type,
            severity=excluded.severity,
            state=excluded.state,
            reporter=excluded.reporter,
            reporter_reputation=excluded.reporter_reputation,
            reporter_signal=excluded.reporter_signal,
            created_at=excluded.created_at,
            closed_at=excluded.closed_at,
            team_comments=excluded.team_comments,
            indexed=CASE WHEN :reset_index THEN 0 ELSE reports.indexed END
        """,
        {**report, "reset_index": content_changed},
    )
    return existing is None


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync closed HackerOne reports.")
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Full sync of all closed reports (default is incremental).",
    )
    args = parser.parse_args()

    try:
        config = common.load_config()
        identifier, token, handle = common.require_h1_credentials(config)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return 1

    client = common.HackerOneClient(identifier, token, handle)
    conn = common.get_db()

    created_after: str | None = None
    if not args.sync:
        created_after = latest_created_at(conn)
        if created_after:
            console.print(
                f"[cyan]Incremental sync[/]: fetching reports created after "
                f"[bold]{created_after}[/]."
            )
        else:
            console.print(
                "[yellow]No existing reports found; performing a full sync.[/]"
            )
    else:
        console.print("[cyan]Full sync[/]: fetching all closed reports.")

    inserted = 0
    updated = 0
    processed = 0

    try:
        # We can't know the total up-front from the HackerOne API without an
        # extra count call, so we render a live counter instead of a bar.
        with console.status("[bold green]Syncing reports…") as status:
            for raw in client.iter_reports(
                states=common.CLOSED_STATES, created_after=created_after
            ):
                report = common.normalize_report(raw)
                processed += 1
                status.update(
                    f"[{processed}] Syncing report #{report['id']}…"
                )
                console.print(
                    f"[{processed}] Syncing report #{report['id']}: "
                    f"[dim]{common.safe_line(report['title'][:60])}[/]"
                )
                if upsert_report(conn, report):
                    inserted += 1
                else:
                    updated += 1
                # Commit periodically so a crash doesn't lose everything.
                if processed % 50 == 0:
                    conn.commit()
        conn.commit()
    except common.HackerOneError as exc:
        conn.commit()
        console.print(f"[bold red]HackerOne API error:[/] {common.safe(exc)}")
        console.print(
            f"[yellow]Saved {inserted} new / {updated} updated reports before the error.[/]"
        )
        return 1
    except KeyboardInterrupt:
        conn.commit()
        console.print("\n[yellow]Interrupted, progress saved.[/]")
        return 130
    finally:
        conn.close()

    console.print(
        f"\n[bold green]Done.[/] {processed} reports processed "
        f"([green]{inserted} new[/], [blue]{updated} updated[/])."
    )
    console.print("Run [bold]python index.py --update[/] to embed new reports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
