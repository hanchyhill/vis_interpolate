"""能见度业务流程、调度和文件发布。"""

from __future__ import annotations

import logging
import logging.handlers
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import xarray as xr

from .algorithms import estimate_both
from .api import StationBatch, VisibilityApiClient
from .config import BusinessConfig
from .idw import create_visibility_grid
from .plot import plot_visibility
from .state import PipelineState, process_lock


def configure_logging(config: BusinessConfig) -> logging.Logger:
    logger = logging.getLogger("vis_interpolate.business")
    if logger.handlers:
        return logger
    config.log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.handlers.RotatingFileHandler(
        config.log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console)
    return logger


def close_logging() -> None:
    """关闭业务日志句柄，便于测试或进程重载时释放日志文件。"""
    logger = logging.getLogger("vis_interpolate.business")
    for handler in list(logger.handlers):
        handler.flush()
        handler.close()
        logger.removeHandler(handler)


def window_times(now: datetime) -> list[datetime]:
    """返回严格滚动30分钟且距当前至少5分钟的5分钟UTC时次。"""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    start = _ceil_five(now - timedelta(minutes=30))
    end = _floor_five(now - timedelta(minutes=5))
    result: list[datetime] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(minutes=5)
    return result


def run_once(now: datetime | None = None, config: BusinessConfig | None = None) -> list[dict[str, Any]]:
    config = config or BusinessConfig.from_file()
    logger = configure_logging(config)
    state = PipelineState(config.state_path)
    with process_lock(config.lock_path) as acquired:
        if not acquired:
            logger.warning("已有业务任务执行中，本轮跳过")
            return [{"status": "lock_skipped"}]
        client = VisibilityApiClient(config.api)
        results = []
        current = now or datetime.now().astimezone()
        for observation_time in window_times(current):
            results.append(_process_time(observation_time, client, state, config, logger))
        return results


def _process_time(
    observation_time: datetime,
    client: VisibilityApiClient,
    state: PipelineState,
    config: BusinessConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    label = observation_time.strftime("%Y%m%d%H%M")
    try:
        batch = client.fetch(observation_time)
        counts = batch.sample_counts
        state.seen(observation_time, counts)
        if not state.should_process(observation_time, counts):
            logger.info("%s 样本数无增长，跳过", label)
            return {"time": observation_time, "status": "skipped", "counts": counts}
        outputs = _build_outputs(observation_time, batch, config)
        state.success(observation_time, counts, outputs)
        if not config.guangdong_boundary_path.exists():
            logger.warning("%s 未找到广东边界文件，已完成数据处理但未生成图片: %s", label, config.guangdong_boundary_path)
        logger.info("%s 处理成功 counts=%s", label, counts)
        return {"time": observation_time, "status": "success", "counts": counts, "outputs": outputs}
    except Exception as exc:  # noqa: BLE001 - one时次失败不影响窗口其余时次
        state.failure(observation_time, locals().get("counts"), str(exc))
        logger.exception("%s 处理失败: %s", label, exc)
        return {"time": observation_time, "status": "failed", "error": str(exc)}


def _build_outputs(observation_time: datetime, batch: StationBatch, config: BusinessConfig) -> list[str]:
    estimates = estimate_both(batch.national, batch.regional)
    if not config.dem_path.exists():
        raise FileNotFoundError(f"DEM文件不存在: {config.dem_path}")
    with xr.open_dataset(config.dem_path) as dem:
        grids = {
            source: create_visibility_grid(frame, dem)
            for source, frame in estimates.items()
        }

    date_parts = (observation_time.strftime("%Y"), observation_time.strftime("%m"), observation_time.strftime("%d"))
    stamp = observation_time.strftime("%Y%m%d%H%M")
    csv_paths = {
        "national": config.csv_national_root.joinpath(*date_parts, f"station_vis_all_estimated_{stamp}.csv"),
        "national_and_regional": config.csv_combined_root.joinpath(
            *date_parts, f"station_vis_all_estimated_{stamp}_national_and_regional.csv"
        ),
    }
    nc_paths = {
        "national": config.nc_national_root.joinpath(*date_parts, f"visibility_anisotropic_idw_{stamp}.nc"),
        "national_and_regional": config.nc_combined_root.joinpath(
            *date_parts, f"visibility_anisotropic_idw_{stamp}.nc"
        ),
    }
    image_paths = {
        "national": config.vis_img_root.joinpath(*date_parts, f"visibility_national_{stamp}.png"),
        "national_and_regional": config.vis_img_root.joinpath(
            *date_parts, f"visibility_national_and_regional_{stamp}.png"
        ),
    }
    for path in [*csv_paths.values(), *nc_paths.values(), *image_paths.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"business_{stamp}_", dir=config.state_path.parent) as temp_dir:
        temp = Path(temp_dir)
        staged: list[tuple[Path, Path]] = []
        for source, frame in estimates.items():
            csv_temp = temp / f"{source}.csv"
            nc_temp = temp / f"{source}.nc"
            frame.to_csv(csv_temp, index=False, encoding="utf-8-sig")
            grids[source].attrs.update(
                {
                    "observation_time_utc": observation_time.astimezone(timezone.utc).isoformat(),
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source_type": source,
                    "station_count": int(len(frame)),
                }
            )
            grids[source].to_netcdf(nc_temp)
            staged.extend([(csv_temp, csv_paths[source]), (nc_temp, nc_paths[source])])
            if config.guangdong_boundary_path.exists():
                image_temp = temp / f"{source}.png"
                plot_visibility(
                    nc_temp,
                    config.guangdong_boundary_path,
                    image_temp,
                    title=f"广东省能见度 - {source} - {stamp}",
                )
                staged.append((image_temp, image_paths[source]))
        for source_temp, destination in staged:
            source_temp.replace(destination)
    return [str(path) for _, path in staged]


def run_forever(config: BusinessConfig | None = None) -> None:
    config = config or BusinessConfig.from_file()
    logger = configure_logging(config)
    last_slot: str | None = None
    logger.info("业务调度启动，分钟表=%s", config.schedule_minutes)
    while True:
        now = datetime.now().astimezone()
        slot = now.strftime("%Y%m%d%H%M")
        if now.minute in config.schedule_minutes and slot != last_slot:
            last_slot = slot
            run_once(now=now, config=config)
        time.sleep(10)


def _floor_five(value: datetime) -> datetime:
    value = value.replace(second=0, microsecond=0)
    return value - timedelta(minutes=value.minute % 5)


def _ceil_five(value: datetime) -> datetime:
    floor = _floor_five(value)
    return floor if floor == value else floor + timedelta(minutes=5)
