"""SQLite metadata and job-state store (SPEC §3, §11, §12).

Tables: recordings, events, searches (job state with cancellation flag), audit. Large
artefacts (audio, spectrograms, run outputs) live on disk under ``data/`` and are referenced by
path. All writes are small and transactional; the store is safe to open from the API process
and from job subprocesses concurrently (WAL mode).
"""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from aerodoppler.models import DopplerEvent, Recording, SearchRequest, utc_now_iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS recordings (
  id TEXT PRIMARY KEY, created_at TEXT NOT NULL, json TEXT NOT NULL,
  dir TEXT NOT NULL, spectrogram_path TEXT, deleted INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, created_at TEXT NOT NULL, json TEXT NOT NULL,
  FOREIGN KEY(recording_id) REFERENCES recordings(id)
);
CREATE INDEX IF NOT EXISTS idx_events_rec ON events(recording_id);
CREATE TABLE IF NOT EXISTS searches (
  id TEXT PRIMARY KEY, recording_id TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
  status TEXT NOT NULL, stage TEXT, request_json TEXT NOT NULL, progress_json TEXT,
  result_dir TEXT, error TEXT, pid INTEGER, cancel_requested INTEGER NOT NULL DEFAULT 0,
  cpu_seconds REAL, peak_rss_mb REAL
);
CREATE INDEX IF NOT EXISTS idx_searches_rec ON searches(recording_id);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, action TEXT NOT NULL,
  subject TEXT, detail TEXT
);
"""

JOB_STATES = ("queued", "running", "succeeded", "failed", "cancelled", "abstained")


class Store:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA busy_timeout=30000")
        con.row_factory = sqlite3.Row
        try:
            yield con
        finally:
            con.close()

    # ---- audit -------------------------------------------------------------------------
    def audit(self, action: str, subject: str | None = None, detail: Any = None) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO audit(ts, action, subject, detail) VALUES (?,?,?,?)",
                (
                    utc_now_iso(),
                    action,
                    subject,
                    json.dumps(detail, default=str) if detail is not None else None,
                ),
            )

    def audit_log(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT * FROM audit ORDER BY seq DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---- recordings --------------------------------------------------------------------
    def add_recording(self, rec: Recording, directory: Path) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO recordings(id, created_at, json, dir) VALUES (?,?,?,?)",
                (rec.id, rec.created_at, rec.model_dump_json(), str(directory)),
            )
        self.audit(
            "recording.ingest",
            rec.id,
            {"filename": rec.original_filename, "sha256": rec.sha256, "synthetic": rec.synthetic},
        )

    def get_recording(self, rec_id: str) -> Recording | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT json FROM recordings WHERE id=? AND deleted=0", (rec_id,)
            ).fetchone()
        return Recording.model_validate_json(row["json"]) if row else None

    def recording_dir(self, rec_id: str) -> Path | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT dir FROM recordings WHERE id=? AND deleted=0", (rec_id,)
            ).fetchone()
        return Path(row["dir"]) if row else None

    def set_spectrogram_path(self, rec_id: str, path: Path) -> None:
        with self._conn() as c:
            c.execute("UPDATE recordings SET spectrogram_path=? WHERE id=?", (str(path), rec_id))

    def spectrogram_path(self, rec_id: str) -> Path | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT spectrogram_path FROM recordings WHERE id=? AND deleted=0", (rec_id,)
            ).fetchone()
        return Path(row["spectrogram_path"]) if row and row["spectrogram_path"] else None

    def list_recordings(self) -> list[Recording]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT json FROM recordings WHERE deleted=0 ORDER BY created_at DESC"
            ).fetchall()
        return [Recording.model_validate_json(r["json"]) for r in rows]

    def mark_recording_deleted(self, rec_id: str) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE recordings SET deleted=1, spectrogram_path=NULL, json=json_set(json,'$.original_filename','<purged>') WHERE id=?",
                (rec_id,),
            )
            c.execute("DELETE FROM events WHERE recording_id=?", (rec_id,))
        self.audit("recording.purge", rec_id)

    # ---- events ------------------------------------------------------------------------
    def add_event(self, ev: DopplerEvent) -> DopplerEvent:
        if not ev.id:
            from aerodoppler.util import new_id

            ev.id = new_id("evt")
        with self._conn() as c:
            c.execute(
                "INSERT OR REPLACE INTO events(id, recording_id, created_at, json) VALUES (?,?,?,?)",
                (ev.id, ev.recording_id, ev.created_at, ev.model_dump_json()),
            )
        self.audit(
            "event.add",
            ev.id,
            {
                "recording_id": ev.recording_id,
                "method": ev.detection_method,
                "n_points": len(ev.ridge_points),
            },
        )
        return ev

    def get_event(self, ev_id: str) -> DopplerEvent | None:
        with self._conn() as c:
            row = c.execute("SELECT json FROM events WHERE id=?", (ev_id,)).fetchone()
        return DopplerEvent.model_validate_json(row["json"]) if row else None

    def list_events(self, recording_id: str) -> list[DopplerEvent]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT json FROM events WHERE recording_id=? ORDER BY created_at", (recording_id,)
            ).fetchall()
        return [DopplerEvent.model_validate_json(r["json"]) for r in rows]

    def delete_event(self, ev_id: str) -> bool:
        with self._conn() as c:
            n = c.execute("DELETE FROM events WHERE id=?", (ev_id,)).rowcount
        self.audit("event.delete", ev_id)
        return n > 0

    # ---- searches ----------------------------------------------------------------------
    def create_search(self, search_id: str, req: SearchRequest, result_dir: Path) -> None:
        now = utc_now_iso()
        with self._conn() as c:
            c.execute(
                "INSERT INTO searches(id, recording_id, created_at, updated_at, status, stage, request_json, progress_json, result_dir) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    search_id,
                    req.recording_id,
                    now,
                    now,
                    "queued",
                    "queued",
                    req.model_dump_json(),
                    json.dumps({}),
                    str(result_dir),
                ),
            )
        self.audit(
            "search.create",
            search_id,
            {
                "mode": str(req.mode),
                "recording_id": req.recording_id,
                "bbox": req.bbox.as_tuple() if req.bbox else None,
            },
        )

    def update_search(self, search_id: str, **fields: Any) -> None:
        if not fields:
            return
        if "progress" in fields:
            fields["progress_json"] = json.dumps(fields.pop("progress"), default=str)
        fields["updated_at"] = utc_now_iso()
        cols = ", ".join(f"{k}=?" for k in fields)
        with self._conn() as c:
            c.execute(f"UPDATE searches SET {cols} WHERE id=?", (*fields.values(), search_id))

    def get_search(self, search_id: str) -> dict[str, Any] | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM searches WHERE id=?", (search_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["request"] = json.loads(d.pop("request_json"))
        d["progress"] = json.loads(d.pop("progress_json") or "{}")
        d["cancel_requested"] = bool(d["cancel_requested"])
        return d

    def list_searches(self, recording_id: str | None = None) -> list[dict[str, Any]]:
        with self._conn() as c:
            if recording_id:
                rows = c.execute(
                    "SELECT id FROM searches WHERE recording_id=? ORDER BY created_at DESC",
                    (recording_id,),
                ).fetchall()
            else:
                rows = c.execute("SELECT id FROM searches ORDER BY created_at DESC").fetchall()
        return [s for s in (self.get_search(r["id"]) for r in rows) if s]

    def request_cancel(self, search_id: str) -> bool:
        with self._conn() as c:
            n = c.execute(
                "UPDATE searches SET cancel_requested=1, updated_at=? WHERE id=? AND status IN ('queued','running')",
                (utc_now_iso(), search_id),
            ).rowcount
        self.audit("search.cancel", search_id)
        return n > 0

    def cancel_requested(self, search_id: str) -> bool:
        with self._conn() as c:
            row = c.execute(
                "SELECT cancel_requested FROM searches WHERE id=?", (search_id,)
            ).fetchone()
        return bool(row and row["cancel_requested"])

    def count_active(self) -> int:
        with self._conn() as c:
            row = c.execute(
                "SELECT COUNT(*) AS n FROM searches WHERE status IN ('running')"
            ).fetchone()
        return int(row["n"])

    def delete_searches_for_recording(self, recording_id: str) -> list[Path]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT result_dir FROM searches WHERE recording_id=?", (recording_id,)
            ).fetchall()
            c.execute("DELETE FROM searches WHERE recording_id=?", (recording_id,))
        return [Path(r["result_dir"]) for r in rows if r["result_dir"]]
