

把数据流程业务化。
改造流程。
1. 能见度数据流改为从API接口获取
2. 能见度估算分为两个输出文件：1. 仅使用国家站能见度数据的估算其他区域站的结果；2. 使用国家站 + 区域站的能见度数据的估算其他区域站的结果
3. IDW插值结果输出为两个文件：1. 使用国家站能见度数据的插值结果；2. 使用国家站 + 区域站的能见度数据的插值结果
4. 定时执行任务：


接口示例：

1. 区域站数据获取：
```
http://172.22.1.175/di/http.action?userId={userId}&pwd={pwd}&interfaceId=getSurfAwstOrg4Prov&dataFormat=csv&ymdhms=20260805060000&prov=广东
```
其中日期ymdhms格式为：YYYYMMDDHHMMSS，prov为省份名称。其中日期为世界时。
2. 国家站数据获取：
``` 
http://172.22.1.175/di/http.action?userId={userId}&pwd={pwd}&interfaceId=getSurfAutoOrg4Prov&dataFormat=csv&ymdhms=20260805060000&prov=广东
```
数据的采样时间间隔为5分钟，也就是每5分钟一个有效数据。
其中，userId和pwd为接口访问的用户名和密码，从 @src\config\local.config.json 当中获取, 格式如下：
```
{
    "userId":"username",
    "pwd":"password"
}
```
为了保证数据的完整性，定时任务设置为每5分钟执行一次，每次执行检查前30分钟的数据，并记录数据样本的数量，如果有效样本数量大于前一次执行的有效样本数量，则重新进行能见度估算和IDW插值处理，并输出结果文件；否则，跳过本次处理，等待下一次定时任务执行。同时由于采样数据一般会延迟6分钟左右，所以定时的时间可设置为07分，12分，17分、22、27、32、37、42、47、52、57分、02分。同时与当前时间间隔小于5分钟的数据不进行处理，避免数据不完整。
同时注意数据竞争，如果上一个时次任务还没有执行完，当前时次任务就开始执行了，那么当前时次任务将会被跳过，等待下一次定时任务执行。

业务化的脚本保存在全新的目录 @src\business\ 下，避免跟调试的脚本冲突。
netCDF文件保存在 @data\idw_nc\national 和 @data\idw_nc\national_and_regional， 同时为了避免数据文件过多，按照日期创建子文件夹，子文件夹的命名规则如下：
```
YYYY/MM/DD/
```

**数据流各环节一览**：

| 环节 | 脚本 | 输入 → 输出 |
|---|---|---|
| ① DEM 处理 | `src/dem_interpolation.py` | 高分辨率 GeoTIFF 瓦片 → `merged_dem_data.nc`（0.01° 网格，含边界优化） |
| ② 能见度估算 | `src/get_vis_estimated_by_rh.py` | 国家站观测 CSV（网络共享）→ 各站估算能见度 CSV |
| ③ IDW 插值 | `src/vis_dem_dis.py` | 站点 CSV + DEM 海拔 → `visibility_anisotropic_idw_{time}.nc` |
| ④ 调试可视化 | `debug_visibility_visualization.py` | **CLDAS 在线数据 或 本地 IDW NC** + 广东边界 → 综合对比图 |
| ⑤ 模型评估 | `src/evaluate_visibility_model.py` | IDW 结果 vs CLDAS 产品 → 评分 CSV |

本次流程只需要环节②能见度估算和③IDW 插值的改造。
