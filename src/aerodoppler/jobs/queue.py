"""Persistent job queue (SQLite) and lifecycle state machine.

Every long-running operation — flight imports, audio ingest, spectrograms, ridge detection,
geolocation searches, benchmarks — is a row in ``jobs``. The row, not the worker process, is
the source of truth; a worker owns a job only while it holds an unexpired lease *and* its
recorded process identity (PID + creation time + a per-claim token) still matches a live
process. Rows marked ``running`` are never counted as "live workers" by themselves.

State machine (every transition is one guarded UPDATE inside ``BEGIN IMMEDIATE``)::

    queued ──claim──▶ claimed ──start──▶ running ──▶ succeeded | abstained | failed | cancelled
       │                 │                  │
       │cancel           └── lease lost ────┴──▶ interrupted  (or queued again when the job
       ▼                                                     type is safe to resume and
    cancelled                                                attempts remain)

``failed`` jobs whose error is transient go back to ``queued`` with ``next_attempt_at`` set
(bounded exponential backoff, ``max_attempts``). Events (status/stage/progress changes) are
appended to ``job_events`` with a global monotonically increasing ``seq`` so a client can
reconnect (SSE ``Last-Event-ID``) without missing state changes.

The :class:`JobQueue` protocol is the interface the rest of the code uses; SQLite is the only
backend. Swapping it (e.g. for a server deployment) does not touch the analysis engine.
"""

from __future__ import annotations

import json
import os
import secrets
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

ACTIVE = ("queued", "claimed", "running")
LIVE = ("claimed", "running")
TERMINAL = ("succeeded", "abstained", "failed", "cancelled", "interrupted")

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
  id TEXT PRIMARY KEY,
  type TEXT NOT NULL,
  status TEXT NOT NULL,
  stage TEXT,
  params_json TEXT NOT NULL,
  idempotency_key TEXT,
  resource_class TEXT NOT NULL,
  memory_mb REAL NOT NULL DEFAULT 0,
  priority INTEGER NOT NULL DEFAULT 0,
  recording_id TEXT,
  dataset_id TEXT,
  mode TEXT NOT NULL DEFAULT 'worker',
  created_at REAL NOT NULL,
  claimed_at REAL,
  started_at REAL,
  updated_at REAL NOT NULL,
  finished_at REAL,
  worker_token TEXT,
  worker_host TEXT,
  pid INTEGER,
  pid_started_at REAL,
  lease_expires_at REAL,
  heartbeat_at REAL,
  attempt INTEGER NOT NULL DEFAULT 0,
  max_attempts INTEGER NOT NULL DEFAULT 3,
  next_attempt_at REAL,
  cancel_requested INTEGER NOT NULL DEFAULT 0,
  cancel_requested_at REAL,
  progress_json TEXT NOT NULL DEFAULT '{}',
  resources_json TEXT NOT NULL DEFAULT '{}',
  error_code TEXT,
  error_message TEXT,
  error_details TEXT,
  result_json TEXT,
  log_path TEXT,
  work_dir TEXT,
  legacy INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_type ON jobs(type, status);
CREATE INDEX IF NOT EXISTS idx_jobs_idem ON jobs(idempotency_key);
CREATE INDEX IF NOT EXISTS idx_jobs_rec ON jobs(recording_id);
CREATE TABLE IF NOT EXISTS job_events (
  seq INTEGER PRIMARY KEY AUTOINCREMENT,
  job_id TEXT NOT NULL,
  ts REAL NOT NULL,
  kind TEXT NOT NULL,
  data_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_job_events_job ON job_events(job_id, seq);
CREATE TABLE IF NOT EXISTS scheduler_lock (
  name TEXT PRIMARY KEY,
  owner TEXT NOT NULL,
  pid INTEGER NOT NULL,
  pid_started_at REAL,
  lease_expires_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS schema_migrations (
  version INTEGER PRIMARY KEY,
  applied_at REAL NOT NULL,
  note TEXT
);
"""

LEASE_S = 30.0


class JobError(RuntimeError):
    """Raised by handlers. ``transient`` errors may be retried automatically."""

    def __init__(self, message: str, code: str = "error", transient: bool = False, **details: Any):
        super().__init__(message)
        self.code = code
        self.transient = transient
        self.details = details


class JobCancelled(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    type: str
    status: str
    stage: str | None
    params: dict[str, Any]
    idempotency_key: str | None
    resource_class: str
    memory_mb: float
    recording_id: str | None
    dataset_id: str | None
    mode: str
    created_at: float
    claimed_at: float | None
    started_at: float | None
    updated_at: float
    finished_at: float | None
    worker_token: str | None
    pid: int | None
    pid_started_at: float | None
    lease_expires_at: float | None
    heartbeat_at: float | None
    attempt: int
    max_attempts: int
    next_attempt_at: float | None
    cancel_requested: bool
    progress: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None
    error_message: str | None = None
    error_details: Any = None
    result: Any = None
    log_path: str | None = None
    work_dir: str | None = None
    legacy: bool = False

    @classmethod
    def from_row(cls, r: sqlite3.Row) -> Job:
        d = dict(r)
        return cls(
            id=d["id"],
            type=d["type"],
            status=d["status"],
            stage=d["stage"],
            params=json.loads(d["params_json"] or "{}"),
            idempotency_key=d["idempotency_key"],
            resource_class=d["resource_class"],
            memory_mb=float(d["memory_mb"] or 0.0),
            recording_id=d["recording_id"],
            dataset_id=d["dataset_id"],
            mode=d["mode"],
            created_at=d["created_at"],
            claimed_at=d["claimed_at"],
            started_at=d["started_at"],
            updated_at=d["updated_at"],
            finished_at=d["finished_at"],
            worker_token=d["worker_token"],
            pid=d["pid"],
            pid_started_at=d["pid_started_at"],
            lease_expires_at=d["lease_expires_at"],
            heartbeat_at=d["heartbeat_at"],
            attempt=int(d["attempt"]),
            max_attempts=int(d["max_attempts"]),
            next_attempt_at=d["next_attempt_at"],
            cancel_requested=bool(d["cancel_requested"]),
            progress=json.loads(d["progress_json"] or "{}"),
            resources=json.loads(d["resources_json"] or "{}"),
            error_code=d["error_code"],
            error_message=d["error_message"],
            error_details=json.loads(d["error_details"]) if d["error_details"] else None,
            result=json.loads(d["result_json"]) if d["result_json"] else None,
            log_path=d["log_path"],
            work_dir=d["work_dir"],
            legacy=bool(d["legacy"]),
        )

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d.pop("worker_token", None)  # a capability; never exposed
        return d


@dataclass
class Capacity:
    """Admission limits: per-class concurrency plus one shared memory budget."""

    class_limits: dict[str, int]
    memory_budget_mb: float


class JobQueue(Protocol):  # the interface the rest of AeroDoppler uses
    def submit(self, job_type: str, params: dict[str, Any], **kw: Any) -> tuple[Job, bool]: ...
    def get(self, job_id: str) -> Job | None: ...
    def list(self, **filters: Any) -> list[Job]: ...
    def claim_next(self, capacity: Capacity, now: float | None = None) -> Job | None: ...
    def request_cancel(self, job_id: str) -> Job | None: ...


def new_job_id() -> str:
    return "job_" + secrets.token_hex(6)


class SqliteJobQueue:
    def __init__(self, db_path: Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        with self._conn() as c:
            c.execute("BEGIN IMMEDIATE")
            try:
                yield c
            except BaseException:
                c.execute("ROLLBACK")
                raise
            c.execute("COMMIT")

    # ---- events ------------------------------------------------------------------------
    @staticmethod
    def _event(c: sqlite3.Connection, job_id: str, kind: str, data: dict[str, Any]) -> None:
        c.execute(
            "INSERT INTO job_events(job_id, ts, kind, data_json) VALUES (?,?,?,?)",
            (job_id, time.time(), kind, json.dumps(data, default=str)),
        )

    def events(
        self, job_id: str | None = None, after: int = 0, limit: int = 500
    ) -> list[dict[str, Any]]:
        with self._conn() as c:
            if job_id:
                rows = c.execute(
                    "SELECT * FROM job_events WHERE job_id=? AND seq>? ORDER BY seq LIMIT ?",
                    (job_id, after, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM job_events WHERE seq>? ORDER BY seq LIMIT ?", (after, limit)
                ).fetchall()
        return [
            {
                "seq": r["seq"],
                "job_id": r["job_id"],
                "ts": r["ts"],
                "kind": r["kind"],
                "data": json.loads(r["data_json"]),
            }
            for r in rows
        ]

    # ---- submission --------------------------------------------------------------------
    def submit(
        self,
        job_type: str,
        params: dict[str, Any],
        *,
        resource_class: str,
        memory_mb: float,
        idempotency_key: str | None = None,
        recording_id: str | None = None,
        dataset_id: str | None = None,
        max_attempts: int = 3,
        priority: int = 0,
        mode: str = "worker",
        job_id: str | None = None,
    ) -> tuple[Job, bool]:
        """Insert a queued job, or return the existing active (or succeeded) job with the same
        idempotency key. Returns ``(job, created)``. Atomic: concurrent duplicate submissions
        resolve to one row."""
        now = time.time()
        jid = job_id or new_job_id()
        with self._tx() as c:
            if idempotency_key:
                row = c.execute(
                    "SELECT * FROM jobs WHERE idempotency_key=? AND status IN "
                    "('queued','claimed','running') ORDER BY created_at LIMIT 1",
                    (idempotency_key,),
                ).fetchone()
                if row:
                    return Job.from_row(row), False
            c.execute(
                "INSERT INTO jobs(id, type, status, stage, params_json, idempotency_key, "
                "resource_class, memory_mb, priority, recording_id, dataset_id, mode, created_at, "
                "updated_at, max_attempts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    jid,
                    job_type,
                    "queued",
                    "queued",
                    json.dumps(params, default=str),
                    idempotency_key,
                    resource_class,
                    float(memory_mb),
                    priority,
                    recording_id,
                    dataset_id,
                    mode,
                    now,
                    now,
                    max_attempts,
                ),
            )
            self._event(c, jid, "status", {"status": "queued", "type": job_type})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        return Job.from_row(row), True

    # ---- reads -------------------------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row) if row else None

    def list(
        self,
        *,
        types: list[str] | None = None,
        statuses: list[str] | None = None,
        recording_id: str | None = None,
        dataset_id: str | None = None,
        created_after: float | None = None,
        created_before: float | None = None,
        limit: int = 200,
    ) -> list[Job]:
        conds, params = [], []
        if types:
            conds.append("type IN (" + ",".join("?" * len(types)) + ")")
            params += types
        if statuses:
            conds.append("status IN (" + ",".join("?" * len(statuses)) + ")")
            params += statuses
        if recording_id:
            conds.append("recording_id=?")
            params.append(recording_id)
        if dataset_id:
            conds.append("dataset_id=?")
            params.append(dataset_id)
        if created_after is not None:
            conds.append("created_at>=?")
            params.append(created_after)
        if created_before is not None:
            conds.append("created_at<=?")
            params.append(created_before)
        q = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(conds) if conds else "")
        q += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as c:
            rows = c.execute(q, params).fetchall()
        return [Job.from_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._conn() as c:
            rows = c.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status").fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    # ---- claiming / admission ----------------------------------------------------------
    def claim_next(
        self, capacity: Capacity, now: float | None = None, types: list[str] | None = None
    ) -> Job | None:
        """Atomically claim the oldest queued job that fits: its class below its concurrency
        limit and its memory reservation within the shared budget (reservations of every
        claimed/running job are summed). Head-of-line protection: a job that does not fit only
        for lack of memory keeps its reservation, so younger jobs may start only in what is
        left after it — big jobs are not starved by a stream of small ones."""
        now = now or time.time()
        with self._tx() as c:
            live = c.execute(
                "SELECT resource_class, memory_mb FROM jobs WHERE status IN ('claimed','running')"
            ).fetchall()
            used_mem = sum(float(r["memory_mb"]) for r in live)
            per_class: dict[str, int] = {}
            for r in live:
                per_class[r["resource_class"]] = per_class.get(r["resource_class"], 0) + 1
            q = (
                "SELECT * FROM jobs WHERE status='queued' AND mode='worker' AND cancel_requested=0 "
                "AND (next_attempt_at IS NULL OR next_attempt_at<=?)"
            )
            args: list[Any] = [now]
            if types:
                q += " AND type IN (" + ",".join("?" * len(types)) + ")"
                args += types
            q += " ORDER BY priority DESC, created_at"
            reserved = 0.0
            for row in c.execute(q, args).fetchall():
                cls, mem = row["resource_class"], float(row["memory_mb"])
                if per_class.get(cls, 0) >= capacity.class_limits.get(cls, 1):
                    continue
                if used_mem + reserved + mem > capacity.memory_budget_mb:
                    reserved += mem  # head-of-line reservation
                    continue
                token = secrets.token_hex(8)
                n = c.execute(
                    "UPDATE jobs SET status='claimed', stage='claimed', claimed_at=?, updated_at=?, "
                    "worker_token=?, lease_expires_at=?, attempt=attempt+1, pid=NULL, "
                    "pid_started_at=NULL WHERE id=? AND status='queued'",
                    (now, now, token, now + LEASE_S, row["id"]),
                ).rowcount
                if n:
                    self._event(c, row["id"], "status", {"status": "claimed"})
                    out = c.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
                    return Job.from_row(out)
            return None

    def claim_inline(
        self, job_id: str, capacity: Capacity, pid: int, pid_started_at: float | None
    ) -> bool:
        """Admission for an inline (in-process) job: same class and memory rules as workers."""
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                "SELECT * FROM jobs WHERE id=? AND status='queued'", (job_id,)
            ).fetchone()
            if not row:
                return False
            live = c.execute(
                "SELECT resource_class, memory_mb FROM jobs WHERE status IN ('claimed','running')"
            ).fetchall()
            used = sum(float(r["memory_mb"]) for r in live)
            n_cls = sum(1 for r in live if r["resource_class"] == row["resource_class"])
            if n_cls >= capacity.class_limits.get(row["resource_class"], 1):
                return False
            if used + float(row["memory_mb"]) > capacity.memory_budget_mb:
                return False
            token = secrets.token_hex(8)
            c.execute(
                "UPDATE jobs SET status='running', stage='starting', claimed_at=?, started_at=?, "
                "updated_at=?, worker_token=?, lease_expires_at=?, heartbeat_at=?, pid=?, "
                "pid_started_at=?, attempt=attempt+1 WHERE id=?",
                (now, now, now, token, now + LEASE_S, now, pid, pid_started_at, job_id),
            )
            self._event(c, job_id, "status", {"status": "running", "mode": "inline"})
        return True

    # ---- worker side -------------------------------------------------------------------
    def start(
        self, job_id: str, token: str, pid: int, pid_started_at: float | None, host: str
    ) -> Job | None:
        now = time.time()
        with self._tx() as c:
            n = c.execute(
                "UPDATE jobs SET status='running', stage='starting', started_at=?, updated_at=?, "
                "pid=?, pid_started_at=?, worker_host=?, heartbeat_at=?, lease_expires_at=? "
                "WHERE id=? AND worker_token=? AND status='claimed'",
                (now, now, pid, pid_started_at, host, now, now + LEASE_S, job_id, token),
            ).rowcount
            if not n:
                return None
            self._event(c, job_id, "status", {"status": "running", "pid": pid})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row)

    def set_process(self, job_id: str, token: str, pid: int, pid_started_at: float | None) -> None:
        """Record the spawned worker process right after spawning (before it starts)."""
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET pid=?, pid_started_at=?, updated_at=? WHERE id=? AND worker_token=?",
                (pid, pid_started_at, time.time(), job_id, token),
            )

    def heartbeat(
        self, job_id: str, token: str, resources: dict[str, Any] | None = None
    ) -> tuple[bool, bool]:
        """Renew the lease. Returns ``(still_owner, cancel_requested)``."""
        now = time.time()
        with self._conn() as c:
            sets = "heartbeat_at=?, lease_expires_at=?, updated_at=?"
            args: list[Any] = [now, now + LEASE_S, now]
            if resources is not None:
                sets += ", resources_json=?"
                args.append(json.dumps(resources, default=str))
            n = c.execute(
                f"UPDATE jobs SET {sets} WHERE id=? AND worker_token=? AND status IN ('claimed','running')",
                (*args, job_id, token),
            ).rowcount
            row = c.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return n > 0, bool(row and row["cancel_requested"])

    def progress(self, job_id: str, token: str, stage: str, info: dict[str, Any]) -> bool:
        now = time.time()
        with self._tx() as c:
            row = c.execute(
                "SELECT progress_json, stage FROM jobs WHERE id=? AND worker_token=? AND status='running'",
                (job_id, token),
            ).fetchone()
            if not row:
                return False
            prog = json.loads(row["progress_json"] or "{}")
            stages = prog.setdefault("stages", {})
            stages[stage] = info
            prog["stage"] = stage
            c.execute(
                "UPDATE jobs SET stage=?, progress_json=?, updated_at=?, heartbeat_at=? WHERE id=?",
                (stage, json.dumps(prog, default=str), now, now, job_id),
            )
            self._event(
                c,
                job_id,
                "progress",
                {"stage": stage, "info": info, "stage_changed": row["stage"] != stage},
            )
        return True

    def finish(
        self,
        job_id: str,
        token: str | None,
        status: str,
        *,
        result: Any = None,
        error_code: str | None = None,
        error_message: str | None = None,
        error_details: Any = None,
        resources: dict[str, Any] | None = None,
        dataset_id: str | None = None,
        retry_at: float | None = None,
    ) -> bool:
        """Terminal transition by the owning worker (``token``) or, with ``token=None``, by
        the scheduler/reconciler. Idempotent: only an active row changes. ``retry_at`` turns a
        failure into a re-queue (transient errors, attempts left)."""
        if status not in TERMINAL:
            raise ValueError(f"not a terminal status: {status}")
        now = time.time()
        with self._tx() as c:
            owner = "AND worker_token=?" if token else ""
            args: list[Any] = [job_id] + ([token] if token else [])
            row = c.execute(
                f"SELECT * FROM jobs WHERE id=? {owner} AND status IN ('queued','claimed','running')",
                args,
            ).fetchone()
            if not row:
                return False
            new_status = "queued" if retry_at is not None else status
            c.execute(
                "UPDATE jobs SET status=?, stage=?, finished_at=?, updated_at=?, result_json=?, "
                "error_code=?, error_message=?, error_details=?, resources_json=COALESCE(?, resources_json), "
                "dataset_id=COALESCE(?, dataset_id), next_attempt_at=?, worker_token=NULL, "
                "lease_expires_at=NULL WHERE id=?",
                (
                    new_status,
                    "waiting_to_retry" if retry_at is not None else status,
                    None if retry_at is not None else now,
                    now,
                    json.dumps(result, default=str) if result is not None else row["result_json"],
                    error_code,
                    error_message,
                    json.dumps(error_details, default=str) if error_details is not None else None,
                    json.dumps(resources, default=str) if resources is not None else None,
                    dataset_id,
                    retry_at,
                    job_id,
                ),
            )
            self._event(
                c,
                job_id,
                "status",
                {
                    "status": new_status,
                    "error_code": error_code,
                    "error": error_message,
                    "retry_at": retry_at,
                },
            )
        return True

    def set_paths(self, job_id: str, log_path: str | None, work_dir: str | None) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET log_path=COALESCE(?, log_path), work_dir=COALESCE(?, work_dir) WHERE id=?",
                (log_path, work_dir, job_id),
            )

    def set_links(
        self, job_id: str, recording_id: str | None = None, dataset_id: str | None = None
    ) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE jobs SET recording_id=COALESCE(?, recording_id), dataset_id=COALESCE(?, dataset_id) WHERE id=?",
                (recording_id, dataset_id, job_id),
            )

    # ---- cancellation / retry ----------------------------------------------------------
    def request_cancel(self, job_id: str) -> Job | None:
        """Queued jobs are cancelled at once (no worker ever starts); claimed/running jobs get
        a flag the worker polls, and the scheduler escalates if the worker does not stop."""
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if not row:
                return None
            if row["status"] == "queued":
                c.execute(
                    "UPDATE jobs SET status='cancelled', stage='cancelled', cancel_requested=1, "
                    "cancel_requested_at=?, finished_at=?, updated_at=?, error_message=? WHERE id=?",
                    (now, now, now, "cancelled before it started", job_id),
                )
                self._event(c, job_id, "status", {"status": "cancelled"})
            elif row["status"] in LIVE and not row["cancel_requested"]:
                c.execute(
                    "UPDATE jobs SET cancel_requested=1, cancel_requested_at=?, updated_at=? WHERE id=?",
                    (now, now, job_id),
                )
                self._event(c, job_id, "cancel_requested", {})
            out = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(out)

    def requeue(self, job_id: str) -> Job | None:
        """Manual retry of a terminal job (failed/interrupted/cancelled): same id and history."""
        now = time.time()
        with self._tx() as c:
            n = c.execute(
                "UPDATE jobs SET status='queued', stage='queued', cancel_requested=0, "
                "cancel_requested_at=NULL, finished_at=NULL, updated_at=?, next_attempt_at=NULL, "
                "error_code=NULL, error_message=NULL, error_details=NULL, max_attempts=attempt+? "
                "WHERE id=? AND status IN ('failed','interrupted','cancelled') AND mode='worker'",
                (now, 3, job_id),
            ).rowcount
            if not n:
                return None
            self._event(c, job_id, "status", {"status": "queued", "retry": "manual"})
            row = c.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return Job.from_row(row)

    def requeue_lost(self, job_id: str, reason: str) -> bool:
        """Put a job whose worker was lost back in the queue (resume-safe types only)."""
        now = time.time()
        with self._tx() as c:
            n = c.execute(
                "UPDATE jobs SET status='queued', stage='queued', worker_token=NULL, "
                "lease_expires_at=NULL, pid=NULL, pid_started_at=NULL, updated_at=?, "
                "error_message=? WHERE id=? AND status IN ('claimed','running') AND attempt<max_attempts",
                (now, reason, job_id),
            ).rowcount
            if n:
                self._event(c, job_id, "status", {"status": "queued", "requeued_after": reason})
        return n > 0

    # ---- scheduler lock ----------------------------------------------------------------
    def acquire_scheduler_lock(
        self, owner: str, pid: int, pid_started_at: float | None, ttl_s: float = 15.0
    ) -> bool:
        """Only one scheduler (API process or headless CLI) dispatches at a time."""
        now = time.time()
        with self._tx() as c:
            row = c.execute("SELECT * FROM scheduler_lock WHERE name='main'").fetchone()
            if row and row["owner"] != owner and row["lease_expires_at"] > now:
                return False
            c.execute(
                "INSERT INTO scheduler_lock(name, owner, pid, pid_started_at, lease_expires_at) "
                "VALUES ('main',?,?,?,?) ON CONFLICT(name) DO UPDATE SET owner=excluded.owner, "
                "pid=excluded.pid, pid_started_at=excluded.pid_started_at, "
                "lease_expires_at=excluded.lease_expires_at",
                (owner, pid, pid_started_at, now + ttl_s),
            )
        return True

    def release_scheduler_lock(self, owner: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM scheduler_lock WHERE name='main' AND owner=?", (owner,))

    # ---- migrations --------------------------------------------------------------------
    def applied_migrations(self) -> set[int]:
        with self._conn() as c:
            return {int(r["version"]) for r in c.execute("SELECT version FROM schema_migrations")}

    def record_migration(self, version: int, note: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT OR IGNORE INTO schema_migrations(version, applied_at, note) VALUES (?,?,?)",
                (version, time.time(), note),
            )


def host_id() -> str:
    try:
        return os.uname().nodename
    except AttributeError:  # pragma: no cover - non-POSIX
        return "localhost"
