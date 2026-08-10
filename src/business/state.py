"""业务状态数据库和跨进程锁。"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


class PipelineState:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._connection() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS observations (
                    observation_time TEXT PRIMARY KEY,
                    last_seen_counts TEXT,
                    last_processed_counts TEXT,
                    status TEXT NOT NULL,
                    error TEXT,
                    outputs TEXT,
                    metrics TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(observations)")}
            if "metrics" not in columns:
                conn.execute("ALTER TABLE observations ADD COLUMN metrics TEXT")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS observation_queue (
                    observation_time TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    updated_at TEXT NOT NULL
                )"""
            )

    def should_process(self, observation_time: datetime, counts: dict[str, int]) -> bool:
        key = _time_key(observation_time)
        with self._connection() as conn:
            row = conn.execute(
                "SELECT last_processed_counts FROM observations WHERE observation_time = ?", (key,)
            ).fetchone()
        if not row or not row["last_processed_counts"]:
            return any(counts.values())
        previous = json.loads(row["last_processed_counts"])
        return any(int(counts.get(name, 0)) > int(previous.get(name, 0)) for name in counts)

    def seen(self, observation_time: datetime, counts: dict[str, int], status: str = "seen") -> None:
        self._upsert(observation_time, counts, status=status)

    def success(
        self,
        observation_time: datetime,
        counts: dict[str, int],
        outputs: list[str],
        metrics: dict[str, object] | None = None,
    ) -> None:
        key = _time_key(observation_time)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO observations
                   (observation_time,last_seen_counts,last_processed_counts,status,error,outputs,metrics,updated_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(observation_time) DO UPDATE SET
                     last_seen_counts=excluded.last_seen_counts,
                     last_processed_counts=excluded.last_processed_counts,
                     status=excluded.status,error=NULL,outputs=excluded.outputs,metrics=excluded.metrics,updated_at=excluded.updated_at""",
                (
                    key,
                    json.dumps(counts),
                    json.dumps(counts),
                    "success",
                    None,
                    json.dumps(outputs),
                    json.dumps(metrics or {}, ensure_ascii=False),
                    _now(),
                ),
            )

    def failure(self, observation_time: datetime, counts: dict[str, int] | None, error: str) -> None:
        self._upsert(observation_time, counts, status="failed", error=error)

    def _upsert(
        self,
        observation_time: datetime,
        counts: dict[str, int] | None,
        *,
        status: str,
        error: str | None = None,
    ) -> None:
        key = _time_key(observation_time)
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO observations
                   (observation_time,last_seen_counts,last_processed_counts,status,error,outputs,metrics,updated_at)
                   VALUES (?,?,?,?,?,?,?,?)
                   ON CONFLICT(observation_time) DO UPDATE SET
                     last_seen_counts=COALESCE(excluded.last_seen_counts, observations.last_seen_counts),
                     status=excluded.status,error=excluded.error,updated_at=excluded.updated_at""",
                (key, json.dumps(counts) if counts is not None else None, None, status, error, None, None, _now()),
            )

    def enqueue(self, observation_times: list[datetime]) -> None:
        now = _now()
        with self._connection() as conn:
            conn.executemany(
                """INSERT INTO observation_queue(observation_time,status,attempts,error,updated_at)
                   VALUES (?, 'pending', 0, NULL, ?)
                   ON CONFLICT(observation_time) DO UPDATE SET
                     status=CASE WHEN status='processing' THEN status ELSE 'pending' END,
                     error=CASE WHEN status='processing' THEN error ELSE NULL END,
                     updated_at=excluded.updated_at""",
                [(_time_key(value), now) for value in observation_times],
            )

    def claim_backfill(self, *, exclude: datetime, limit: int) -> list[datetime]:
        if limit <= 0:
            return []
        excluded = _time_key(exclude)
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT observation_time FROM observation_queue
                   WHERE observation_time <> ? AND status IN ('pending', 'failed')
                   ORDER BY observation_time ASC LIMIT ?""",
                (excluded, limit),
            ).fetchall()
            now = _now()
            for row in rows:
                conn.execute(
                    """UPDATE observation_queue
                       SET status='processing', attempts=attempts + 1, updated_at=?
                       WHERE observation_time=?""",
                    (now, row["observation_time"]),
                )
        return [_parse_time_key(row["observation_time"]) for row in rows]

    def queue_result(self, observation_time: datetime, *, success: bool, error: str | None = None) -> None:
        with self._connection() as conn:
            conn.execute(
                """UPDATE observation_queue
                   SET status=?, error=?, updated_at=?
                   WHERE observation_time=?""",
                ("done" if success else "failed", error, _now(), _time_key(observation_time)),
            )


def _time_key(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_time_key(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def process_lock(path: Path) -> Iterator[bool]:
    """使用平台文件锁，保证常驻进程和外部补跑不会并发处理。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    acquired = False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                acquired = False
        else:
            import fcntl

            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                acquired = False
        yield acquired
    finally:
        if acquired:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                try:
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
