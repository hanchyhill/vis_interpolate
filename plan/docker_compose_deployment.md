# 能见度业务流水线 Docker Compose 部署方案

## 摘要

- 将现有 `python -m src.business serve` 封装为 Python 3.13 Linux 容器，由 Docker Compose 单实例守护，不在容器内运行 PM2。
- 镜像在服务器从仓库本地构建；依赖严格使用现有 `uv.lock`，首版不拆分生产与 Notebook 依赖。
- 配置文件只读挂载，DEM、边界文件、SQLite 状态、日志和生成结果统一挂载到宿主机 `/srv/vis-interpolate/data`。
- 增加轻量 HTTP 健康服务，仅映射到宿主机 `127.0.0.1`，不提供管理或补跑 API。

## 实现变更

### 容器与 Compose

- 新增 Dockerfile、`.dockerignore`、`compose.yaml` 和不包含敏感信息的 `.env.example`。
- 基于 `python:3.13-slim-bookworm`，使用固定版本 uv 执行 `uv sync --frozen --no-dev`。
- 安装 `fonts-noto-cjk`、`fontconfig`、`libgomp1`、CA 证书等运行库，保证中文绘图和科学计算依赖可用。
- 使用固定非 root UID/GID `10001`，启用 `init`、只读根文件系统、`tmpfs /tmp` 和 `restart: unless-stopped`。
- Compose 将 `${VIS_CONFIG_FILE}` 挂载为 `/run/config/business.config.json:ro`，`${VIS_DATA_DIR}` 挂载为 `/data`。
- 健康端口在容器内使用 `8181`，宿主机映射为 `127.0.0.1:${VIS_HEALTH_PORT:-8181}`。
- 首版不设置 CPU 和内存硬限制，先采集真实运行峰值。
- 设置 5 分钟停止宽限期，避免正在发布的 CSV、NetCDF 或图片被立即中断。

### 服务生命周期

- `serve` 增加 `--health-host`、`--health-port` 参数，Compose 使用 `0.0.0.0:8181`。
- 处理 SIGTERM 和 SIGINT：停止接收新调度，等待当前处理及绘图任务收尾，关闭 HTTP 服务和日志后退出。
- 使用可中断事件替代固定 `time.sleep`，确保空闲时可以快速停止。
- 记录调度阶段、最近活动时间、当前任务开始时间和最近一轮结果。
- 上游 API 失败只反映在状态中，不以容器重启代替业务重试，避免重启风暴。

### 健康接口

- `GET /health/live`：健康线程可以响应且服务未进入关闭状态时返回 HTTP 200。
- `GET /health/ready`：主调度仍活跃、DEM 存在、状态与输出目录可访问时返回 HTTP 200；本地必要条件失败时返回 HTTP 503。
- 响应使用 JSON，包含状态、运行阶段、活动时间、最近处理结果和本地依赖检查，但不得包含账号、密码、完整 API URL 或配置文件内容。
- 广东边界文件缺失时标记为 `degraded` 并保持 HTTP 200，因为现有业务允许继续生成 CSV 和 NetCDF。
- DEM 或数据目录不可用时返回 HTTP 503。
- Docker HEALTHCHECK 调用 `/health/ready`，配置启动宽限期和连续失败次数。

### 配置与运维文档

- 宿主机配置使用 `/data`、`/data/assets/dem/merged_dem_data.nc`、`/data/assets/gis/guangdong/广东省_省界.shp` 等容器内路径。
- 初始化 `/srv/vis-interpolate/config` 和 `/srv/vis-interpolate/data`，将数据目录授权给 UID/GID `10001`，配置文件由同一用户只读访问。
- 说明本地构建、启动、状态检查、日志查看、单次补跑、升级重建、回滚以及 SQLite 和结果目录备份命令。
- PM2 配置保留为旧部署方式，但文档明确 Docker 部署不再使用 PM2。
- 增加磁盘容量、容器内存峰值和 Docker 日志轮转的观测建议，取得生产数据后再确定资源限额。

## 接口与配置约定

- 新增 CLI：`python -m src.business serve --health-host 0.0.0.0 --health-port 8181`。
- 新增 HTTP 接口：`/health/live`、`/health/ready`；仅允许 GET，其他路径返回 404，非 GET 请求返回 405。
- `.env` 只保存宿主机路径、端口等非密钥部署参数；API 账号密码仍放在不提交的 JSON 配置文件中。
- 生产环境保持单容器单副本；SQLite 和文件锁不支持横向扩容，也不使用 NFS。
- 容器时区设置为 `Asia/Shanghai`，业务资料时间继续使用现有 UTC 与北京时间转换逻辑。

## 测试与验收

### 自动化测试

- 单元测试覆盖健康接口的 200、503、404、405、降级状态、敏感信息脱敏和服务关闭行为。
- 覆盖 SIGTERM 和停止事件：空闲调度能及时退出，处理中的任务不会启动下一轮，健康状态切换为关闭中。
- 运行完整业务测试：

  ```bash
  uv run python -m unittest discover -s tests -p "test_*.py"
  ```

### Docker 验证

- `docker compose config` 成功，构建上下文不包含 `data/`、`.git/`、`.venv/`、PNG、NetCDF 等大文件。
- `docker compose build` 能根据锁文件完成 Python 3.13 Linux 环境安装。
- 使用测试配置启动后容器状态为 `healthy`，宿主机只能通过 `127.0.0.1` 访问健康接口。
- 容器重建后 SQLite 状态和历史输出仍然存在。
- 配置或 DEM 缺失时 readiness 返回 503，并能从日志明确定位原因。
- 在具备真实接口访问条件的服务器执行一次 `run-once`，核验 CSV、NetCDF、PNG、SQLite 状态和中文字体渲染。

## 假设与默认值

- 目标平台为支持 Docker Compose v2 的 Linux x86_64 单机。
- 服务器构建镜像时能够访问基础镜像仓库和 Python 依赖源。
- 宿主机数据根目录默认 `/srv/vis-interpolate/data`，配置文件默认 `/srv/vis-interpolate/config/business.config.json`。
- 服务只承担后台定时处理和健康探测，不新增远程补跑、状态管理或业务数据下载 API。
- 首版保留当前完整依赖集以降低行为差异；镜像瘦身和资源限额待取得生产运行数据后单独实施。
