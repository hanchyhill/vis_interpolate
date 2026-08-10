"""业务流程命令行入口。"""

from __future__ import annotations

import argparse
import signal
import threading
from datetime import datetime

from .config import BusinessConfig
from .health import HealthState, start_health_server
from .pipeline import run_forever, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="能见度业务化API采集、估算与IDW处理")
    parser.add_argument("command", choices=["serve", "run-once"])
    parser.add_argument("--config", default=None, help="API账号和数据目录配置JSON路径")
    parser.add_argument("--dem-path", default=None, help="merged_dem_data.nc路径")
    parser.add_argument("--province", default=None, help="单省份覆盖（兼容参数）")
    parser.add_argument("--provinces", default=None, help="省份覆盖，逗号分隔")
    parser.add_argument("--now", default=None, help="run-once使用的时间，ISO-8601格式")
    parser.add_argument("--health-host", default=None, help="serve健康检查监听地址")
    parser.add_argument("--health-port", type=int, default=None, help="serve健康检查监听端口")
    args = parser.parse_args()
    config = BusinessConfig.from_file(
        args.config, dem_path=args.dem_path, province=args.province, provinces=args.provinces
    )
    if args.command == "serve":
        stop_event = threading.Event()
        health_state = HealthState(config)

        def request_stop(_signum: int, _frame: object) -> None:
            health_state.mark_stopping()
            stop_event.set()

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        health_server = None
        if args.health_port is not None:
            if not 1 <= args.health_port <= 65535:
                parser.error("--health-port 必须在 1 到 65535 之间")
            health_server = start_health_server(args.health_host or "127.0.0.1", args.health_port, health_state)
        try:
            run_forever(config, stop_event=stop_event, health_state=health_state)
        finally:
            health_state.mark_stopping()
            if health_server is not None:
                health_server.close()
    else:
        now = datetime.fromisoformat(args.now) if args.now else None
        run_once(now=now, config=config, include_cldas=True)


if __name__ == "__main__":
    main()
