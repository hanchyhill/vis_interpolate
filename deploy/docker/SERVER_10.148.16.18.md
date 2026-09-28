# 10.148.16.18 Docker 部署与运维记录

本文记录 `vis-interpolate` 已在服务器 `10.148.16.18`（SSH 别名：`16.18`）上的实际部署方式和日常运维命令。账号、密码和业务数据不记录在本文或 Git 仓库中。

## 当前部署

| 项目 | 值 |
| --- | --- |
| 操作系统 | 银河麒麟 V10，x86_64 |
| Docker | 26.1.3 |
| Docker Compose | v2.27.0 |
| 登录方式 | `ssh 16.18` |
| 代码目录 | `/opt/vis-interpolate` |
| 私密配置 | `/srv/vis-interpolate/config/business.config.json` |
| 持久化数据（迁移后） | `/data/vis-interpolate` |
| 健康检查 | 仅服务器本机 `127.0.0.1:8181` |
| Compose 服务名 | `vis-interpolate-business` |

容器以 UID/GID `10001:10001` 访问配置和数据。`/data/vis-interpolate` 位于服务器本地 2TB ext4 挂载 `/data`，不能将 SQLite 状态文件置于 NFS、SMB 等网络文件系统。完成下方迁移前，服务仍会使用旧的 `/srv/vis-interpolate/data` 目录。

## 迁移持久化数据到 /data

为避免根分区因业务输出增长而空间不足，迁移全部容器数据（DEM、边界、状态库、日志和输出）到 `/data/vis-interpolate`；容器内挂载点仍为 `/data`，因此 `business.config.json` 中的容器内路径不需要改动。

```bash
ssh 16.18
cd /opt/vis-interpolate

# 先停止调度，避免复制 SQLite 和业务输出时仍有写入。
docker compose stop vis-interpolate-business

# 以下命令需要手动输入 sudo 密码。
sudo install -d -o 10001 -g 10001 -m 750 /data/vis-interpolate
sudo rsync -aHAX --numeric-ids --info=progress2 \
  /srv/vis-interpolate/data/ /data/vis-interpolate/
sudo chown -R 10001:10001 /data/vis-interpolate
sudo chmod -R u+rwX /data/vis-interpolate

# 代码仓库归当前 SSH 用户所有，不需要 sudo。
sed -i 's|^VIS_DATA_DIR=.*|VIS_DATA_DIR=/data/vis-interpolate|' .env
docker compose config --quiet
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:8181/health/live
curl --fail http://127.0.0.1:8181/health/ready
```

健康检查稳定前不要删除旧的 `/srv/vis-interpolate/data`；它可作为快速回退副本。回退时将 `.env` 的 `VIS_DATA_DIR` 改回旧路径后执行 `docker compose up -d`。

## 登录与状态检查

```bash
ssh 16.18
cd /opt/vis-interpolate
docker compose ps
docker compose logs --tail=100 vis-interpolate-business
```

容器的实际名称由 Compose 自动生成，不应硬编码。查看 Docker health 状态时使用 Compose 解析出的容器 ID：

```bash
cd /opt/vis-interpolate
container_id=$(docker compose ps -q vis-interpolate-business)
docker inspect --format '{{.State.Status}} health={{.State.Health.Status}}' "$container_id"
```

## 健康检查

服务端口只绑定到回环地址，不对外网开放：

```bash
curl --fail http://127.0.0.1:8181/health/live
curl --fail http://127.0.0.1:8181/health/ready
```

从本机远程访问时建立 SSH 隧道：

```bash
ssh -L 8181:127.0.0.1:8181 16.18
```

然后在本机访问 `http://127.0.0.1:8181/health/ready`。

`live` 表示服务进程可响应；`ready` 还会验证 DEM、数据目录、输出目录和边界文件。DEM 路径为：

```text
/data/vis-interpolate/assets/dem/merged_dem_data.nc
```

## 浏览器查看业务输出

可使用 [nginx-vis.8080.conf](nginx-vis.8080.conf) 建立只读静态文件站点。该配置仅公开 CSV、NetCDF 和 PNG 输出目录，刻意不公开 `assets/`、`business/`、SQLite 状态库、日志或私密配置。

服务器安装并加载 Nginx 配置后，示例访问地址为：

```text
http://10.148.16.18:8080/vis/national/
http://10.148.16.18:8080/vis/national-and-regional/
http://10.148.16.18:8080/vis/idw-national/
http://10.148.16.18:8080/vis/idw-national-and-regional/
http://10.148.16.18:8080/vis/images/
```

Nginx 工作进程必须具备输出目录的遍历和读取权限；不要通过放宽整个 `/data/vis-interpolate` 的权限来解决。服务器的 `/data` 挂载点为 `root:root`、权限 `700`，因此即使业务文件本身可读，Nginx 仍会因不能穿过父目录而返回 HTTP 403。应仅对 Nginx 运行用户授予下述最小 ACL。

当前服务器已安装 Nginx 1.30.4，工作用户为 `nginx`，配置目录为 `/etc/nginx/conf.d/`。从本机上传配置：

```bash
scp deploy/docker/nginx-vis.8080.conf 16.18:/tmp/nginx-vis.8080.conf
```

然后登录服务器，手动执行以下需要 `sudo` 的命令：

```bash
sudo install -m 644 /tmp/nginx-vis.8080.conf \
  /etc/nginx/conf.d/nginx-vis.8080.conf

# 仅允许 Nginx 穿过父目录，不允许列出 data 的全部内容。
sudo setfacl -m u:nginx:--x /data /data/vis-interpolate

# 赋予既有输出文件和目录的只读/遍历权限，并让未来按日期创建的文件继承 ACL。
for output_dir in \
  /data/vis-interpolate/vis_estimated_base_nation_station \
  /data/vis-interpolate/vis_estimated_base_nation_and_regional_station \
  /data/vis-interpolate/idw_nc/national \
  /data/vis-interpolate/idw_nc/national_and_regional \
  /data/vis-interpolate/vis_img; do
  sudo find "$output_dir" -type d -exec setfacl -m u:nginx:rx,d:u:nginx:rx {} +
  sudo find "$output_dir" -type f -exec setfacl -m u:nginx:r {} +
done

sudo nginx -t
sudo systemctl enable --now nginx
```

如果某个输出目录尚未生成，上述 `find` 会失败；可先跳过该目录，待业务首次生成后对该目录补执行 ACL 命令。ACL 修改会即时生效，不需要重启或重新加载 Nginx；Nginx 配置本身有变更时才使用 `sudo systemctl reload nginx`。

出现 HTTP 403 时，先验证 Nginx 用户是否可以读取目标目录：

```bash
sudo -u nginx ls -la /data/vis-interpolate/vis_img
curl -I http://127.0.0.1:8080/vis/images/
```

若第一条命令提示权限不足，重新执行上方 `/data` 父目录及输出目录的 ACL 命令。不要用 `chmod -R o+rx /data` 或对整个数据目录放开权限。

## 日常运维

实时查看服务日志：

```bash
cd /opt/vis-interpolate
docker compose logs -f vis-interpolate-business
```

业务日志、锁文件和状态库位于：

```text
/data/vis-interpolate/business/business.log
/data/vis-interpolate/business/pipeline_state.sqlite
/data/vis-interpolate/business/pipeline.lock
```

重启服务：

```bash
cd /opt/vis-interpolate
docker compose restart vis-interpolate-business
docker compose ps
```

服务配置为 `restart: unless-stopped`，Docker 服务在机器重启后恢复时会自动拉起容器。

## 更新应用

更新会向服务发送 SIGTERM，并最多等待 5 分钟让正在执行的任务完成。执行前确认没有 PM2 的同名业务进程，以免重复处理上游资料。

```bash
cd /opt/vis-interpolate
git pull
docker compose build --progress plain
docker compose up -d
docker compose ps
curl --fail http://127.0.0.1:8181/health/ready
```

## 配置和数据权限

配置中的账号、密码及上游地址只能保存在服务器私密配置文件中。配置和数据应归属容器用户：

```bash
sudo chown -R 10001:10001 /srv/vis-interpolate/config /data/vis-interpolate
sudo chmod 600 /srv/vis-interpolate/config/business.config.json
sudo chmod -R u+rwX /data/vis-interpolate
```

上述命令需要人工输入 `sudo` 密码。不要将真实配置、DEM、Shapefile、NetCDF、PNG 或业务输出提交到仓库。

## 备份

升级前至少备份私密配置和 SQLite 状态库。以下命令需要 `sudo`：

```bash
sudo tar -C /srv/vis-interpolate -czf \
  /var/backups/vis-interpolate-config-$(date +%F).tar.gz config
sudo cp /data/vis-interpolate/business/pipeline_state.sqlite \
  /var/backups/pipeline_state-$(date +%F).sqlite
```

业务输出量可能较大，应按服务器本地保留策略或备份系统单独处理。

## 常见问题

`ready` 返回 503 时，先检查资产、权限及服务日志：

```bash
ls -l /data/vis-interpolate/assets/dem/merged_dem_data.nc
ls -l /data/vis-interpolate/assets/gis/guangdong/
cd /opt/vis-interpolate
docker compose logs --tail=200 vis-interpolate-business
curl -i http://127.0.0.1:8181/health/ready
```

不要执行 `docker compose down -v`，也不要删除 `/data/vis-interpolate`；这里保存业务结果和 SQLite 状态。

需要切换到 PM2 或人工单次补跑时，先停止 Docker 常驻服务，具体流程见 [README.md](README.md)。
