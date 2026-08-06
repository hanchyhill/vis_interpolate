"""能见度业务流程、调度和文件发布。"""

from __future__ import annotations

import atexit
import logging
import logging.handlers
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import xarray as xr

from .algorithms import estimate_both
from .api import StationBatch, VisibilityApiClient
from .config import BusinessConfig
from .idw import create_visibility_grid
from .plot import plot_cldas_visibility, plot_visibility
from .state import PipelineState, process_lock

from src.evaluate_visibility import build_filenames_and_urls


_PLOT_EXECUTOR: ThreadPoolExecutor | None = None
_PLOT_EXECUTOR_LOCK = threading.Lock()
_CLDAS_PLOT_EXECUTOR: ThreadPoolExecutor | None = None
_CLDAS_PLOT_EXECUTOR_LOCK = threading.Lock()
_CLDAS_PENDING_OUTPUTS: set[str] = set()
_BEIJING_TIMEZONE = timezone(timedelta(hours=8))


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


def window_times(now: datetime, ready_delay_minutes: int = 5) -> list[datetime]:
    """返回滚动30分钟且距当前达到就绪延迟的5分钟UTC时次。"""
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    start = _ceil_five(now - timedelta(minutes=30))
    end = _floor_five(now - timedelta(minutes=max(0, ready_delay_minutes)))
    result: list[datetime] = []
    current = start
    while current <= end:
        result.append(current)
        current += timedelta(minutes=5)
    return result


def run_once(
    now: datetime | None = None,
    config: BusinessConfig | None = None,
    *,
    include_cldas: bool = False,
) -> list[dict[str, Any]]:
    config = config or BusinessConfig.from_file()
    logger = configure_logging(config)
    state = PipelineState(config.state_path)
    with process_lock(config.lock_path) as acquired:
        if not acquired:
            logger.warning("已有业务任务执行中，本轮跳过")
            return [{"status": "lock_skipped"}]
        current = now or datetime.now().astimezone()
        candidates = window_times(current, config.source_ready_delay_minutes)
        if not candidates:
            if include_cldas:
                run_cldas_visibility_backfill(current, config)
            return []
        state.enqueue(candidates)
        latest = max(candidates)
        backfill = state.claim_backfill(
            exclude=latest,
            limit=config.max_backfill_slots_per_cycle,
        )
        processing_order = [latest, *backfill] if config.latest_first else [*backfill, latest]
        results = []
        for observation_time in processing_order:
            client = VisibilityApiClient(config.api)
            result = _process_time(observation_time, client, state, config, logger)
            state.queue_result(
                observation_time,
                success=result.get("status") in {"success", "skipped"},
                error=result.get("error"),
            )
            results.append(result)
        if include_cldas:
            run_cldas_visibility_backfill(current, config)
        return results


def _process_time(
    observation_time: datetime,
    client: VisibilityApiClient,
    state: PipelineState,
    config: BusinessConfig,
    logger: logging.Logger,
) -> dict[str, Any]:
    label = observation_time.strftime("%Y%m%d%H%M")
    started = time.perf_counter()
    timings: dict[str, Any] = {}
    counts: dict[str, int] | None = None
    try:
        api_started = time.perf_counter()
        batch = client.fetch(observation_time)
        timings["api_seconds"] = round(time.perf_counter() - api_started, 3)
        if batch.request_timings:
            timings["api_request_max_seconds"] = round(max(batch.request_timings.values()), 3)
            timings["api_request_total_seconds"] = round(sum(batch.request_timings.values()), 3)
        timings["parse_seconds"] = round(batch.parse_seconds, 3)
        counts = batch.sample_counts
        state.seen(observation_time, counts)
        if not state.should_process(observation_time, counts):
            timings["total_seconds"] = round(time.perf_counter() - started, 3)
            logger.info("%s 样本数无增长，跳过 timings=%s", label, timings)
            return {"time": observation_time, "status": "skipped", "counts": counts, "metrics": timings}
        outputs = _build_outputs(observation_time, batch, config, timings)
        timings["total_seconds"] = round(time.perf_counter() - started, 3)
        if batch.source_update_time is not None:
            timings["source_update_time_utc"] = batch.source_update_time.isoformat()
            timings["data_delay_seconds"] = round(
                max(0.0, (datetime.now(timezone.utc) - batch.source_update_time).total_seconds()), 3
            )
        state.success(observation_time, counts, outputs, timings)
        if not config.guangdong_boundary_path.exists():
            logger.warning("%s 未找到广东边界文件，已完成数据处理但未生成图片: %s", label, config.guangdong_boundary_path)
        if batch.errors:
            logger.warning("%s 部分接口失败 errors=%s", label, batch.errors)
        logger.info(
            "%s 处理成功 counts=%s source_update_time=%s timings=%s",
            label,
            counts,
            batch.source_update_time.isoformat() if batch.source_update_time else None,
            timings,
        )
        return {
            "time": observation_time,
            "status": "success",
            "counts": counts,
            "outputs": outputs,
            "metrics": timings,
            "api_errors": list(batch.errors),
        }
    except Exception as exc:  # noqa: BLE001 - one时次失败不影响窗口其余时次
        timings["total_seconds"] = round(time.perf_counter() - started, 3)
        state.failure(observation_time, counts, str(exc))
        logger.exception("%s 处理失败: %s", label, exc)
        return {"time": observation_time, "status": "failed", "error": str(exc), "metrics": timings}


def _build_outputs(
    observation_time: datetime,
    batch: StationBatch,
    config: BusinessConfig,
    timings: dict[str, Any] | None = None,
) -> list[str]:
    timings = timings if timings is not None else {}
    estimate_started = time.perf_counter()
    estimates = estimate_both(batch.national, batch.regional)
    timings["estimate_seconds"] = round(time.perf_counter() - estimate_started, 3)
    if not config.dem_path.exists():
        raise FileNotFoundError(f"DEM文件不存在: {config.dem_path}")
    idw_started = time.perf_counter()
    with xr.open_dataset(config.dem_path) as dem:
        grids = {
            source: create_visibility_grid(frame, dem)
            for source, frame in estimates.items()
        }
    timings["idw_seconds"] = round(time.perf_counter() - idw_started, 3)

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

    generated_at = datetime.now(timezone.utc).isoformat()
    timings["generated_at_utc"] = generated_at
    data_outputs: list[str] = []
    publish_started = time.perf_counter()
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
                    "generated_at_utc": generated_at,
                    "source_update_time_utc": (
                        batch.source_update_time.isoformat() if batch.source_update_time else ""
                    ),
                    "source_type": source,
                    "station_count": int(len(frame)),
                }
            )
            grids[source].to_netcdf(nc_temp)
            staged.extend([(csv_temp, csv_paths[source]), (nc_temp, nc_paths[source])])
        for source_temp, destination in staged:
            source_temp.replace(destination)
            data_outputs.append(str(destination))
    timings["publish_seconds"] = round(time.perf_counter() - publish_started, 3)

    if config.guangdong_boundary_path.exists():
        plot_started = time.perf_counter()
        title_stamp = _beijing_timestamp(observation_time, "%Y%m%d%H%M")
        for source, nc_path in nc_paths.items():
            if config.async_plots:
                _submit_plot(
                    nc_path,
                    config.guangdong_boundary_path,
                    image_paths[source],
                    f"广东省能见度 - {source} - {title_stamp}",
                )
            else:
                _render_plot_job(
                    nc_path,
                    config.guangdong_boundary_path,
                    image_paths[source],
                    f"广东省能见度 - {source} - {title_stamp}",
                )
        timings["plot_submit_seconds"] = round(time.perf_counter() - plot_started, 3)
    return [*data_outputs, *[str(path) for path in image_paths.values() if config.guangdong_boundary_path.exists()]]


def _submit_plot(nc_path: Path, boundary_path: Path, output_path: Path, title: str) -> None:
    global _PLOT_EXECUTOR
    with _PLOT_EXECUTOR_LOCK:
        if _PLOT_EXECUTOR is None:
            _PLOT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="visibility-plot")
        _PLOT_EXECUTOR.submit(_render_plot_job, nc_path, boundary_path, output_path, title)


def _render_plot_job(nc_path: Path, boundary_path: Path, output_path: Path, title: str) -> None:
    started = time.perf_counter()
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_", dir=output_path.parent) as temp_dir:
            temporary_output = Path(temp_dir) / output_path.name
            plot_visibility(nc_path, boundary_path, temporary_output, title=title)
            temporary_output.replace(output_path)
        logging.getLogger("vis_interpolate.business").info(
            "%s 图片生成完成 plot_seconds=%.3f",
            output_path.name,
            time.perf_counter() - started,
        )
    except Exception:
        logging.getLogger("vis_interpolate.business").exception("图片生成失败: %s", output_path)


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
            # 小时产品也随现有5分钟调度检查；目标图片存在时不会读取远程NC。
            run_once(now=now, config=config, include_cldas=True)
        time.sleep(config.poll_interval_seconds)


def hourly_visibility_times(now: datetime, sample_count: int = 2) -> list[datetime]:
    """返回最近两个已完成的整点 UTC 时次，避免按5分钟频率重复取产品。"""
    if sample_count <= 0:
        return []
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    current_hour = now.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    return [current_hour - timedelta(hours=offset) for offset in range(1, sample_count + 1)]


def run_cldas_visibility_backfill(
    now: datetime,
    config: BusinessConfig,
    *,
    async_plot: bool = True,
) -> list[str]:
    """补画过去两个小时的国家局 5km 能见度产品。

    产品文件尚未生成时，读取函数会返回 None；此处将其视为本轮可恢复的
    缺测，只记录日志，不影响 5 分钟站点流程。后续调度轮次会再次尝试。
    """
    logger = logging.getLogger("vis_interpolate.business")
    if not config.guangdong_boundary_path.exists():
        logger.warning("未找到广东边界文件，跳过国家局5km能见度图片: %s", config.guangdong_boundary_path)
        return []

    outputs: list[str] = []
    for observation_time in hourly_visibility_times(now, sample_count=2):
        stamp = observation_time.strftime("%Y%m%d%H")
        title_stamp = _beijing_timestamp(observation_time, "%Y%m%d%H")
        output_path = config.vis_img_root.joinpath(
            observation_time.strftime("%Y"),
            observation_time.strftime("%m"),
            observation_time.strftime("%d"),
            f"visibility_cldas_5km_{stamp}.png",
        )
        output_key = str(output_path.resolve())
        with _CLDAS_PLOT_EXECUTOR_LOCK:
            if output_path.exists() or output_key in _CLDAS_PENDING_OUTPUTS:
                continue
            if async_plot:
                _CLDAS_PENDING_OUTPUTS.add(output_key)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data_url = build_filenames_and_urls(observation_time)["cldas_vis"]["url"]
        title = f"国家局5km能见度实况融合 - {title_stamp}"
        if async_plot:
            _submit_cldas_plot(data_url, config.guangdong_boundary_path, output_path, title)
            outputs.append(str(output_path))
        else:
            if _render_cldas_plot_job(data_url, config.guangdong_boundary_path, output_path, title):
                outputs.append(str(output_path))
    return outputs


def _submit_cldas_plot(data_url: str, boundary_path: Path, output_path: Path, title: str) -> None:
    global _CLDAS_PLOT_EXECUTOR
    with _CLDAS_PLOT_EXECUTOR_LOCK:
        if _CLDAS_PLOT_EXECUTOR is None:
            _CLDAS_PLOT_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cldas-visibility-plot")
        _CLDAS_PLOT_EXECUTOR.submit(_render_cldas_plot_job, data_url, boundary_path, output_path, title)


def _render_cldas_plot_job(data_url: str, boundary_path: Path, output_path: Path, title: str) -> bool:
    started = time.perf_counter()
    logger = logging.getLogger("vis_interpolate.business")
    output_key = str(output_path.resolve())
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}_", dir=output_path.parent) as temp_dir:
            temporary_output = Path(temp_dir) / output_path.name
            plot_cldas_visibility(data_url, boundary_path, temporary_output, title=title)
            temporary_output.replace(output_path)
        logger.info("%s 图片生成完成 plot_seconds=%.3f", output_path.name, time.perf_counter() - started)
        return True
    except FileNotFoundError as exc:
        logger.warning("国家局5km能见度数据未生成，跳过图片 %s: %s", output_path.name, exc)
    except Exception as exc:  # noqa: BLE001 - 单个小时产品失败不影响主流程
        logger.warning("国家局5km能见度图片生成失败，跳过 %s: %s", output_path.name, exc)
    finally:
        with _CLDAS_PLOT_EXECUTOR_LOCK:
            _CLDAS_PENDING_OUTPUTS.discard(output_key)
    return False


@atexit.register
def _shutdown_plot_executor() -> None:
    if _PLOT_EXECUTOR is not None:
        _PLOT_EXECUTOR.shutdown(wait=True)
    if _CLDAS_PLOT_EXECUTOR is not None:
        _CLDAS_PLOT_EXECUTOR.shutdown(wait=True)


def _floor_five(value: datetime) -> datetime:
    value = value.replace(second=0, microsecond=0)
    return value - timedelta(minutes=value.minute % 5)


def _beijing_timestamp(value: datetime, format_string: str) -> str:
    """将业务时次按 UTC 解释，并格式化为北京时间显示文本。"""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(_BEIJING_TIMEZONE).strftime(format_string)


def _ceil_five(value: datetime) -> datetime:
    floor = _floor_five(value)
    return floor if floor == value else floor + timedelta(minutes=5)
