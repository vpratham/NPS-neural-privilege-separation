"""Host-owned authority, durable one-use grants, and transactional note effects.

The database and this process are trusted. The model provider has no access to
either. Never expose import_record/issue to untrusted callers or model tools.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .contracts import Decision, canonical_json, content, digest, identifier, parse_proposal, strict_json


@dataclass(frozen=True)
class Grant:
    token: str = field(repr=False)
    subject: str
    run_id: str
    source_id: str
    record_id: str
    value: str
    source_sha256: str
    note_id: str


class Gateway:
    def __init__(self, database: str | Path, *, clock: Callable[[], float] = time.time):
        self.database = Path(database)
        self.clock = clock
        self.database.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # Do not follow a symlink when creating a new authority store.
        try:
            fd = os.open(self.database, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if self.database.is_symlink() or not self.database.is_file():
                raise ValueError("invalid_database_path")
        else:
            os.close(fd)
        with self._connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    source_id TEXT NOT NULL, record_id TEXT NOT NULL,
                    value TEXT NOT NULL, sha256 TEXT NOT NULL,
                    PRIMARY KEY(source_id, record_id)
                );
                CREATE TABLE IF NOT EXISTS grants (
                    token_hash TEXT PRIMARY KEY, subject TEXT NOT NULL,
                    run_id TEXT NOT NULL UNIQUE, source_id TEXT NOT NULL,
                    record_id TEXT NOT NULL, source_sha256 TEXT NOT NULL,
                    note_id TEXT NOT NULL, arguments_hash TEXT NOT NULL,
                    expected_version INTEGER NOT NULL, expires REAL NOT NULL,
                    consumed INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS notes (
                    subject TEXT NOT NULL, note_id TEXT NOT NULL, content TEXT NOT NULL,
                    source_id TEXT NOT NULL, record_id TEXT NOT NULL,
                    source_sha256 TEXT NOT NULL, version INTEGER NOT NULL,
                    run_id TEXT NOT NULL, PRIMARY KEY(subject, note_id)
                );
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL NOT NULL,
                    run_id TEXT NOT NULL, subject TEXT NOT NULL, status TEXT NOT NULL,
                    reason TEXT NOT NULL, proposal_sha256 TEXT, side_effect INTEGER NOT NULL
                );
            """)

    @contextmanager
    def _connection(self):
        db = sqlite3.connect(self.database, timeout=10, isolation_level=None)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA synchronous=FULL")
            yield db
        finally:
            db.close()

    @contextmanager
    def _transaction(self):
        with self._connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.commit()
            except BaseException:
                db.rollback()
                raise

    def import_record(self, raw: bytes, *, expected_sha256: str) -> str:
        """Import a locally approved source snapshot pinned by the host operator.

        A digest verifies equality to approved bytes, not the author's identity.
        Obtain the expected digest from an independent trusted channel.
        """
        if not isinstance(raw, bytes) or len(raw) > 65_536:
            raise ValueError("source_too_large")
        actual = hashlib.sha256(raw).hexdigest()
        if (not isinstance(expected_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
                or not hmac.compare_digest(actual, expected_sha256)):
            raise ValueError("source_digest_mismatch")
        try:
            record = strict_json(raw.decode("utf-8"), limit=65_536)
        except UnicodeError as exc:
            raise ValueError("invalid_source") from exc
        if not isinstance(record, dict) or set(record) != {"source_id", "record_id", "value"}:
            raise ValueError("source_schema")
        source_id, record_id = identifier(record["source_id"]), identifier(record["record_id"])
        value = content(record["value"])
        with self._transaction() as db:
            old = db.execute("SELECT sha256 FROM records WHERE source_id=? AND record_id=?", (source_id, record_id)).fetchone()
            if old and old["sha256"] != actual:
                raise ValueError("source_is_immutable_use_new_record_id")
            db.execute("INSERT OR IGNORE INTO records VALUES (?, ?, ?, ?)", (source_id, record_id, value, actual))
        return actual

    def issue(self, subject: str, source_id: str, record_id: str, note_id: str, *, ttl_seconds: int = 300) -> Grant:
        """Authorize the host-selected copy workflow before model generation."""
        for value in (subject, source_id, record_id, note_id):
            identifier(value)
        if type(ttl_seconds) is not int or not 1 <= ttl_seconds <= 3600:
            raise ValueError("invalid_ttl")
        token, run_id = secrets.token_urlsafe(32), secrets.token_hex(16)
        with self._transaction() as db:
            record = db.execute("SELECT * FROM records WHERE source_id=? AND record_id=?", (source_id, record_id)).fetchone()
            if record is None:
                raise ValueError("unknown_trusted_source")
            note = db.execute("SELECT version FROM notes WHERE subject=? AND note_id=?", (subject, note_id)).fetchone()
            args_hash = digest(canonical_json({"note_id": note_id, "content": record["value"]}))
            db.execute("INSERT INTO grants VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                       (digest(token), subject, run_id, source_id, record_id, record["sha256"], note_id,
                        args_hash, note["version"] if note else 0, self.clock() + ttl_seconds))
            self._event(db, Decision("issued", "source_bound", run_id), subject)
        return Grant(token, subject, run_id, source_id, record_id, record["value"], record["sha256"], note_id)

    def _event(self, db, decision: Decision, subject: str, proposal_hash: str | None = None):
        db.execute("INSERT INTO events(timestamp,run_id,subject,status,reason,proposal_sha256,side_effect) VALUES(?,?,?,?,?,?,?)",
                   (self.clock(), decision.run_id, subject, decision.status, decision.reason, proposal_hash, int(decision.side_effect)))

    def execute(self, subject: str, token: str, raw_proposal: str) -> Decision:
        """Spend the grant and commit a note + audit in the same SQLite transaction.

        Invalid proposals also spend a matching grant. A process crash either
        rolls back all three changes or preserves all three; never retry an
        external side effect using this transaction pattern.
        """
        identifier(subject)
        proposal_hash = None
        try:
            proposal = parse_proposal(raw_proposal)
            proposal_hash = digest(raw_proposal)
            parse_error = None
        except ValueError as exc:
            proposal, parse_error = None, str(exc)
        token_hash = digest(token) if isinstance(token, str) and len(token) <= 256 and token.isascii() else ""
        with self._transaction() as db:
            grant = db.execute("SELECT * FROM grants WHERE token_hash=?", (token_hash,)).fetchone()
            run_id = grant["run_id"] if grant and grant["subject"] == subject else "unknown"

            def finish(reason: str, *, allowed: bool = False):
                result = Decision("executed" if allowed else "blocked", reason, run_id, allowed)
                self._event(db, result, subject, proposal_hash)
                return result

            if grant is None:
                return finish("unknown_capability")
            if grant["subject"] != subject:
                return finish("subject_mismatch")
            if grant["consumed"]:
                return finish("replay_or_revoked")
            db.execute("UPDATE grants SET consumed=1 WHERE token_hash=?", (token_hash,))
            if grant["expires"] <= self.clock():
                return finish("expired")
            if parse_error:
                return finish(parse_error)
            args = proposal["arguments"]
            if args["note_id"] != grant["note_id"]:
                return finish("scope_mismatch")
            if not hmac.compare_digest(grant["arguments_hash"], digest(canonical_json(args))):
                return finish("argument_binding_mismatch")
            note = db.execute("SELECT version FROM notes WHERE subject=? AND note_id=?", (subject, grant["note_id"])).fetchone()
            if (note["version"] if note else 0) != grant["expected_version"]:
                return finish("stale_destination")
            db.execute("""INSERT INTO notes VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                          ON CONFLICT(subject,note_id) DO UPDATE SET
                          content=excluded.content, source_id=excluded.source_id,
                          record_id=excluded.record_id, source_sha256=excluded.source_sha256,
                          version=excluded.version, run_id=excluded.run_id""",
                       (subject, args["note_id"], args["content"], grant["source_id"], grant["record_id"],
                        grant["source_sha256"], grant["expected_version"] + 1, run_id))
            return finish("authorized_source_copy", allowed=True)

    def revoke(self, grant: Grant, *, reason: str = "host_revoked", error: bool = False) -> Decision:
        # Restrict audit reasons to host-selected constants; never log provider error bodies.
        if reason not in {"host_revoked", "provider_error", "monitor_error", "monitor_veto", "invalid_context"}:
            raise ValueError("invalid_revocation_reason")
        with self._transaction() as db:
            row = db.execute("SELECT * FROM grants WHERE token_hash=? AND subject=?", (digest(grant.token), grant.subject)).fetchone()
            if row is None or row["run_id"] != grant.run_id:
                raise ValueError("unknown_capability")
            if row["consumed"]:
                decision = Decision("blocked", "replay_or_revoked", row["run_id"])
            else:
                db.execute("UPDATE grants SET consumed=1 WHERE token_hash=?", (digest(grant.token),))
                decision = Decision("error" if error else "blocked", reason, row["run_id"])
            self._event(db, decision, grant.subject)
            return decision

    def notes(self, subject: str) -> list[dict]:
        identifier(subject)
        with self._connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM notes WHERE subject=? ORDER BY note_id", (subject,))]

    def audit(self, subject: str, *, limit: int = 100) -> list[dict]:
        identifier(subject)
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("invalid_limit")
        with self._connection() as db:
            return [dict(row) for row in db.execute("SELECT * FROM events WHERE subject=? ORDER BY sequence DESC LIMIT ?", (subject, limit))]
