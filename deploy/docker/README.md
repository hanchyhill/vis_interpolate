# vis-interpolate Docker 服务端部署指南

本文用于将能见度业务流水线部署到 Linux 服务器的 Docker Compose 环境。容器运行的是现有业务入口：

```bash
python -m src.business serve --health-host 0.0.0.0 --health-port 8181
```

它是定时后台服务，不提供业务数据查询或补跑 API；仅在宿主机本机开放健康检查端口 `8181`。

## 1. 部署原则与目录

- 使用单机、单副本 Docker Compose；SQLite 状态库和文件锁不支持横向扩容。
- 配置、DEM、Shapefile、SQLite、日志和业务结果均保留在宿主机 bind mount 中；容器重建不会删除这些文件。
- Docker 与 PM2 是二选一的守护方式。切换前必须停止另一种方式，避免同一上游资料被重复处理。
- 容器内业务用户 UID/GID 固定为 `10001:10001`。

以下示例约定：

| 用途 | 服务器路径 | 容器路径 |
| --- | --- | --- |
| 代码仓库 | `/opt/vis-interpolate` | `/app`（镜像内） |
| 私密配置 | `/srv/vis-interpolate/config/business.config.json` | `/run/config/business.config.json`（只读） |
| DEM、边界、状态、日志和输出 | `/srv/vis-interpolate/data` | `/data`（读写） |
| 健康检查 | 宿主机 `127.0.0.1:8181` | `0.0.0.0:8181` |

`/srv/vis-interpolate/data` 必须位于本地磁盘。不要把 SQLite 状态文件放到 NFS、SMB 等网络文件系统上。

## 2. 前置条件

服务器须为 Linux x86_64，具备 Docker Engine 和 Docker Compose v2，并且构建镜像时能访问 Docker Hub、GHCR 和 Python 包源。

```bash
docker --version
docker compose version
docker info
```

克隆或同步项目代码到服务器，例如：

```bash
sudo mkdir -p /opt
sudo git clone <项目仓库地址> /opt/vis-interpolate
sudo chown -R "$USER":"$USER" /opt/vis-interpolate
cd /opt/vis-interpolate
```

不要把真实账号密码、DEM、NetCDF、PNG 或输出数据提交到仓库。

## 3. 准备配置与数据

创建宿主机持久化目录，并复制容器专用配置模板：

```bash
sudo install -d -m 750 /srv/vis-interpolate/config
sudo install -d -m 750 /srv/vis-interpolate/data
sudo cp src/config/business.docker.config.example.json \
  /srv/vis-interpolate/config/business.config.json
sudo chown -R 10001:10001 /srv/vis-interpolate/config /srv/vis-interpolate/data
sudo chmod 600 /srv/vis-interpolate/config/business.config.json
```

编辑 `/srv/vis-interpolate/config/business.config.json`，填写 `userId`、`pwd`、`baseUrl` 和必要的调度参数。Docker 模板已经使用以下容器内路径，不应改为服务器绝对路径：

```json
{
  "dataRoot": "/data",
  "demPath": "/data/assets/dem/merged_dem_data.nc",
  "guangdongBoundaryPath": "/data/assets/gis/guangdong/广东省_省界.shp"
}
```

将输入资产放入数据目录。边界 Shapefile 的配套文件必须完整：

```text
/srv/vis-interpolate/data/
└── assets/
    ├── dem/
    │   └── merged_dem_data.nc
    └── gis/guangdong/
        ├── 广东省_省界.shp
        ├── 广东省_省界.shx
        ├── 广东省_省界.dbf
        └── 广东省_省界.prj
```

DEM 缺失会使 `/health/ready` 返回 503，服务不会被视为就绪。边界文件缺失不会阻断 CSV 和 NetCDF 生成，但健康响应会标记为 `degraded`，且不会生成 PNG。

## 4. 配置 Compose 环境

在代码仓库根目录创建本机部署变量文件：

```bash
cd /opt/vis-interpolate
cp .env.example .env
```

编辑 `.env` 并使用绝对路径：

```dotenv
VIS_CONFIG_FILE=/srv/vis-interpolate/config/business.config.json
VIS_DATA_DIR=/srv/vis-interpolate/data
VIS_HEALTH_PORT=8181
```

`.env` 不含账号密码，但包含服务器路径；它已被 Git 忽略。先检查 Compose 实际渲染的配置：

```bash
docker compose config
```

确认配置文件挂载为只读，数据目录挂载到 `/data`，且端口发布为 `127.0.0.1:8181`。不要把健康端口直接映射到 `0.0.0.0`；远程运维可通过 SSH 隧道访问：

```bash
ssh -L 8181:127.0.0.1:8181 <user>@<server>
```

## 5. 构建并启动

首次构建会下载 Python 3.13、uv 和科学计算/地理依赖，耗时取决于网络速度：

```bash
cd /opt/vis-interpolate
docker compose build --progress plain
docker compose up -d
docker compose ps
```

等待约 30 秒后检查健康状态：

```bash
curl --fail http://127.0.0.1:8181/health/live
curl --fail http://127.0.0.1:8181/health/ready
docker compose logs --tail=200 vis-interpolate-business
```

`/health/live` 表示健康服务仍可响应；`/health/ready` 还会检查 DEM、数据目录和输出目录。容器的 Docker health 状态可以通过以下命令查看：

```bash
docker inspect --format '{{.State.Health.Status}}' vis-interpolate-business
```

服务采用 `restart: unless-stopped`。服务器重启后 Docker 服务恢复时，容器会自动恢复；若管理员执行过 `docker compose stop`，需要手动执行 `docker compose up -d`。

## 6. 日常运维

查看服务和日志：

```bash
docker compose ps
docker compose logs -f vis-interpolate-business
tail -f /srv/vis-interpolate/data/business/business.log
```

业务输出、SQLite 状态和日志均在 `/srv/vis-interpolate/data` 下。常见目录包括：

```text
business/pipeline_state.sqlite
business/business.log
vis_estimated_base_nation_station/YYYY/MM/DD/
vis_estimated_base_nation_and_regional_station/YYYY/MM/DD/
idw_nc/national/YYYY/MM/DD/
idw_nc/national_and_regional/YYYY/MM/DD/
vis_img/YYYY/MM/DD/
```

更新代码和镜像时，Compose 会向容器发送 SIGTERM；服务会停止接收新的调度任务，并在最多 5 分钟内完成当前任务后退出：

```bash
cd /opt/vis-interpolate
git pull
docker compose build --progress plain
docker compose up -d
docker compose ps
```

不要使用 `docker compose down -v`，也不要执行会删除 `/srv/vis-interpolate/data` 的命令。该服务使用宿主机目录挂载，但业务数据和状态库仍需要单独备份。

建议在升级前备份配置和状态；输出数据量较大时按本地保留策略或备份系统处理：

```bash
sudo tar -C /srv/vis-interpolate -czf \
  /var/backups/vis-interpolate-config-$(date +%F).tar.gz config
sudo cp /srv/vis-interpolate/data/business/pipeline_state.sqlite \
  /var/backups/pipeline_state-$(date +%F).sqlite
```

## 7. 手动单次补跑

不要在常驻容器正常运行时启动第二个补跑进程。需要人工补跑时，先停止常驻服务，再执行单次命令，最后恢复服务：

```bash
cd /opt/vis-interpolate
docker compose stop vis-interpolate-business
docker compose run --rm --no-deps vis-interpolate-business \
  python -m src.business run-once --config /run/config/business.config.json \
  --now 2026-08-05T08:02:00+08:00
docker compose up -d vis-interpolate-business
```

省份、DEM 等临时覆盖参数与原有 CLI 相同。补跑使用相同的数据目录和 SQLite 状态库。

## 8. 故障排查

### Compose 提示 `VIS_CONFIG_FILE` 或 `VIS_DATA_DIR` 缺失

确认当前目录是仓库根目录，且 `.env` 已存在并包含两个绝对路径：

```bash
cat .env
docker compose config
```

### 容器启动后立即退出

先查看容器日志。最常见原因是配置 JSON 缺少账号密码、配置文件没有读取权限，或挂载的 DEM 路径不正确：

```bash
docker compose logs --tail=200 vis-interpolate-business
ls -l /srv/vis-interpolate/config/business.config.json
ls -l /srv/vis-interpolate/data/assets/dem/merged_dem_data.nc
```

### `ready` 返回 503 或 Docker 显示 unhealthy

检查 JSON 中的 `/data` 路径、宿主机目录权限及 DEM 文件。容器用户为 `10001:10001`：

```bash
sudo chown -R 10001:10001 /srv/vis-interpolate/data
sudo chmod -R u+rwX /srv/vis-interpolate/data
curl -i http://127.0.0.1:8181/health/ready
```

### 构建停在拉取基础镜像或没有输出

先使用 plain 输出重试，再检查 Docker 网络、代理和 DNS 设置：

```bash
docker compose build --progress plain
docker info | grep -i -E 'proxy|registry'
docker pull python:3.13-slim-bookworm
docker pull ghcr.io/astral-sh/uv:0.8.22
```

服务器无法访问外网时，应在可联网机器构建镜像并通过 `docker save`、`docker load` 离线导入，或配置可访问的内部镜像仓库和 Python 包源。

## 9. 从 Docker 回退到 PM2

Docker 部署不会修改 [ecosystem.config.cjs](../../ecosystem.config.cjs) 或 PM2 进程定义。Docker 不可用时，按以下顺序回退：

```bash
cd /opt/vis-interpolate
docker compose down

# 按原 PM2 部署所使用的配置路径设置；若已在 ecosystem.config.cjs 固定配置，可省略此行。
export VIS_BUSINESS_CONFIG=/path/to/server.config.json
pm2 start ecosystem.config.cjs
pm2 status
pm2 logs vis-interpolate-business
```

确认 PM2 进程正常后再处理 Docker 镜像。`docker compose down` 不会删除宿主机 bind mount 数据；不要在回退过程中删除 `/srv/vis-interpolate/data`、配置文件或 SQLite 状态库。

当需要从 PM2 切回 Docker 时，先执行 `pm2 stop vis-interpolate-business`，确认无 PM2 进程后再执行 `docker compose up -d`。
