"""业务流程命令行入口。"""

from __future__ import annotations

import argparse
from datetime import datetime

from .config import BusinessConfig
from .pipeline import run_forever, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="能见度业务化API采集、估算与IDW处理")
    parser.add_argument("command", choices=["serve", "run-once"])
    parser.add_argument("--config", default=None, help="API账号和数据目录配置JSON路径")
    parser.add_argument("--dem-path", default=None, help="merged_dem_data.nc路径")
    parser.add_argument("--province", default=None, help="单省份覆盖（兼容参数）")
    parser.add_argument("--provinces", default=None, help="省份覆盖，逗号分隔")
    parser.add_argument("--now", default=None, help="run-once使用的时间，ISO-8601格式")
    args = parser.parse_args()
    config = BusinessConfig.from_file(
        args.config, dem_path=args.dem_path, province=args.province, provinces=args.provinces
    )
    if args.command == "serve":
        run_forever(config)
    else:
        now = datetime.fromisoformat(args.now) if args.now else None
        run_once(now=now, config=config)


if __name__ == "__main__":
    main()
