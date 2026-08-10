"""常驻业务进程的本地健康检查服务。"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .config import BusinessConfig


@dataclass
class HealthState:
    """保存不含敏感信息的常驻服务状态。"""

    config: BusinessConfig
    _started: bool = False
    _stopping: bool = False
    _phase: str = "starting"
    _last_activity: datetime | None = None
    _last_result: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def mark_started(self) -> None:
        with self._lock:
            self._started = True
            self._phase = "idle"
            self._last_activity = _now()

    def mark_processing(self) -> None:
        with self._lock:
            self._phase = "processing"
            self._last_activity = _now()

    def mark_cycle_complete(self, result: str = "completed") -> None:
        with self._lock:
            self._phase = "idle"
            self._last_result = result
            self._last_activity = _now()

    def mark_stopping(self) -> None:
        with self._lock:
            self._stopping = True
            self._phase = "stopping"
            self._last_activity = _now()

    def live_payload(self) -> tuple[HTTPStatus, dict[str, Any]]:
        with self._lock:
            payload = self._payload()
            status = HTTPStatus.SERVICE_UNAVAILABLE if self._stopping else HTTPStatus.OK
        return status, payload

    def ready_payload(self) -> tuple[HTTPStatus, dict[str, Any]]:
        with self._lock:
            payload = self._payload()
            started = self._started
            stopping = self._stopping

        checks = {
            "dem": self.config.dem_path.is_file(),
            "data_root": _is_writable_location(self.config.state_path.parent),
            "outputs": all(
                _is_writable_location(path)
                for path in (
                    self.config.csv_national_root,
                    self.config.csv_combined_root,
                    self.config.nc_national_root,
                    self.config.nc_combined_root,
                    self.config.vis_img_root,
                )
            ),
            "boundary": self.config.guangdong_boundary_path.is_file(),
        }
        payload["checks"] = checks
        if not checks["boundary"]:
            payload["status"] = "degraded"
        else:
            payload["status"] = "ready"
        healthy = started and not stopping and all(
            checks[name] for name in ("dem", "data_root", "outputs")
        )
        return (HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE), payload

    def _payload(self) -> dict[str, Any]:
        return {
            "status": "stopping" if self._stopping else "live",
            "phase": self._phase,
            "last_activity_utc": self._last_activity.isoformat() if self._last_activity else None,
            "last_result": self._last_result,
        }


class HealthServer:
    """管理独立线程中的只读 HTTP 健康服务。"""

    def __init__(self, host: str, port: int, state: HealthState):
        handler = _handler_for(state)
        self._server = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="business-health",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    @property
    def port(self) -> int:
        return int(self._server.server_address[1])

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)


def start_health_server(host: str, port: int, state: HealthState) -> HealthServer:
    server = HealthServer(host, port, state)
    server.start()
    return server


def _handler_for(state: HealthState) -> type[BaseHTTPRequestHandler]:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path == "/health/live":
                status, payload = state.live_payload()
            elif self.path == "/health/ready":
                status, payload = state.ready_payload()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            self.send_error(HTTPStatus.METHOD_NOT_ALLOWED)

        do_PUT = do_POST
        do_DELETE = do_POST
        do_PATCH = do_POST

        def log_message(self, format: str, *args: object) -> None:
            return

    return HealthHandler


def _is_writable_location(path: Path) -> bool:
    """不创建目录地检查目标位置或其最近已有父目录的可写性。"""
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current.is_dir() and os.access(current, os.W_OK | os.X_OK)


def _now() -> datetime:
    return datetime.now(timezone.utc)
