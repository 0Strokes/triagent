"""index.py: build the embedding index over stored reports.

Embeddings are generated locally with sentence-transformers (offline, free).
Vectors are stored in store/embeddings.npy and the array-index -> report-id
mapping in store/embedding_map.json.

Usage:
    python index.py --build     # full rebuild of the whole index
    python index.py --update    # only embed reports not yet indexed
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys

import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

import common

console = Console()


def load_model(name: str):
    """Import sentence-transformers lazily (heavy import) and load the model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        console.print(
            "[bold red]sentence-transformers is not installed.[/] "
            "Run: pip install -r requirements.txt"
        )
        raise SystemExit(1)
    console.print(f"[cyan]Loading embedding model[/] [bold]{name}[/]…")
    return SentenceTransformer(name)


def fetch_reports(conn: sqlite3.Connection, only_unindexed: bool) -> list[sqlite3.Row]:
    query = (
        "SELECT id, title, description, vulnerability_type, state FROM reports"
    )
    if only_unindexed:
        query += " WHERE indexed = 0"
    query += " ORDER BY id"
    return conn.execute(query).fetchall()


def embed_reports(model, reports: list[sqlite3.Row]) -> np.ndarray:
    texts = [
        common.embedding_text(
            {
                "title": r["title"],
                "vulnerability_type": r["vulnerability_type"],
                "description": r["description"],
            }
        )
        # Never feed an empty string to the model (yields a zero/NaN vector).
        or (r["title"] or f"report {r['id']}")
        for r in reports
    ]
    vectors: list[np.ndarray] = []
    batch_size = 64
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Embedding reports", total=len(texts))
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            emb = model.encode(
                batch, convert_to_numpy=True, normalize_embeddings=True
            )
            # Empty/whitespace text yields a zero vector → NaN after
            # normalization; scrub so the saved index stays clean.
            emb = np.nan_to_num(emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            vectors.append(emb)
            progress.advance(task, len(batch))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32)
    return np.vstack(vectors)


def print_stats(conn: sqlite3.Connection, total_indexed: int) -> None:
    rows = conn.execute(
        """
        SELECT state, COUNT(*) AS n
        FROM reports WHERE indexed = 1
        GROUP BY state
        """
    ).fetchall()
    counts = {r["state"]: r["n"] for r in rows}

    def c(state: str) -> int:
        return counts.get(state, 0)

    parts = [
        f"{c('not-applicable')} N/A",
        f"{c('informative')} Informative",
        f"{c('resolved')} Resolved",
        f"{c('duplicate')} Duplicate",
    ]
    extra = c("spam")
    if extra:
        parts.append(f"{extra} Spam")
    console.print(
        f"\n[bold green]Indexed {total_indexed} reports.[/] " + ", ".join(parts) + "."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the report embedding index.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--build", action="store_true", help="Full rebuild.")
    group.add_argument(
        "--update", action="store_true", help="Only embed unindexed reports."
    )
    args = parser.parse_args()

    if not args.build and not args.update:
        console.print("[yellow]Specify --build (full) or --update (incremental).[/]")
        return 1

    try:
        config = common.load_config()
    except FileNotFoundError as exc:
        console.print(f"[bold red]Configuration error:[/] {exc}")
        return 1

    model_name = config.get("embeddings", {}).get("model", "all-MiniLM-L6-v2")
    conn = common.get_db()

    try:
        reports = fetch_reports(conn, only_unindexed=args.update)
        if not reports:
            if args.update:
                console.print("[green]Index is already up to date.[/]")
            else:
                console.print("[yellow]No reports to index. Run ingest.py first.[/]")
            return 0

        model = load_model(model_name)
        new_vectors = embed_reports(model, reports)
        new_ids = [int(r["id"]) for r in reports]

        if args.build:
            matrix = new_vectors
            id_map = new_ids
        else:
            # --update: append to (or create) the existing index.
            matrix, id_map = _merge_with_existing(new_vectors, new_ids)

        np.save(common.EMBEDDINGS_PATH, matrix)
        with open(common.EMBEDDING_MAP_PATH, "w", encoding="utf-8") as fh:
            json.dump({str(i): rid for i, rid in enumerate(id_map)}, fh, indent=2)

        # Mark embedded reports as indexed.
        conn.executemany(
            "UPDATE reports SET indexed = 1 WHERE id = ?",
            [(rid,) for rid in new_ids],
        )
        conn.commit()

        print_stats(conn, total_indexed=len(id_map))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted before the index was saved.[/]")
        return 130
    finally:
        conn.close()

    return 0


def _merge_with_existing(
    new_vectors: np.ndarray, new_ids: list[int]
) -> tuple[np.ndarray, list[int]]:
    """Append new vectors to the existing index, replacing any re-indexed ids."""
    if not common.EMBEDDINGS_PATH.exists() or not common.EMBEDDING_MAP_PATH.exists():
        return new_vectors, new_ids

    existing_matrix = np.load(common.EMBEDDINGS_PATH)
    with open(common.EMBEDDING_MAP_PATH, "r", encoding="utf-8") as fh:
        existing_map = json.load(fh)
    existing_ids = [int(existing_map[str(i)]) for i in range(len(existing_map))]

    # Drop any existing entries that are being re-indexed (content changed).
    new_id_set = set(new_ids)
    keep_rows = [i for i, rid in enumerate(existing_ids) if rid not in new_id_set]
    kept_matrix = existing_matrix[keep_rows] if keep_rows else np.empty(
        (0, new_vectors.shape[1]), dtype=np.float32
    )
    kept_ids = [existing_ids[i] for i in keep_rows]

    matrix = np.vstack([kept_matrix, new_vectors]) if kept_ids else new_vectors
    id_map = kept_ids + new_ids
    return matrix, id_map


if __name__ == "__main__":
    sys.exit(main())
