# 能见度插值算法
包含两个部分，一个是基于站点湿度和距离的站点能见度估计算法。另一个是格点化的各向异性反距离权重插值算法。这是一个python项目
## 站点能见度估计算法
1. 用pandas读取站点列表csv文件，根据是否有能见度数据，得到有能见度的站点station_vis. 在没有能见度的站点中，筛选出有相对湿度数据的站点station_rh。
2. station_rh中，每个站点按照距离匹配station_vis当中的距离最近的站点，得到每个站点的站点列表station_nearest，以及对应的距离station_nearest_distance。
3. 用湿度和距离分别达到station_rh中的能见度估计，分别记为vis_rh, vis_dis.
4. 对应station_rh的每个站点的vis_rh的计算方法如下：
   - 读取station_nearest的各个站点的相对湿度rh_i
   - 计算本站和各个station_nearest站点的相对湿度差值d_rh_i = rh - rh_i
   - 得到各个nearest站点的湿度权重w_rh_i = 1/(d_rh_i)^2
   - 根据权重得到根据相对湿度估计的能见度 vis_rh = sum(w_rh_i * vis_i) / sum(w_rh_i)
5. 用反距离权重法得到vis_dis的步骤如下：
   - 读取station_nearest的各个站点的的经纬度，计算与本站之间的距离distance_i
   - 计算各个nearest站点的距离权重w_dis_i = 1/(distance_i)^2
   - 根据权重得到根据距离估计的能见度 vis_dis = sum(w_dis_i * vis_i) / sum(w_dis_i)
7. 最终，将vis_rh和vis_dis进行加权平均，得到每个站点的最终能见度估计值。vis = vis_rh * 0.5 + vis_dis * 0.5
8. 合并station_vis 和 station_rh，得到最终的表station_vis_all. 得到每个站点的最终能见度，添加一列is_vis_est, station_rh中此列为1, station_vis中此列为0，用于判断能见度是真实值还是估测值。
## 格点化能见度估计算法
使用各向异性反距离权重法，将站点能见度插值到格点上。
### 各向异性距离函数形式：
为了让**垂直方向影响更大**，引入一个缩放因子 $\beta > 1$，表示 z 方向权重的“放大”：
$$
d_i = \sqrt{(x_i - x_0)^2 + (y_i - y_0)^2 + \beta^2 (z_i - z_0)^2}
$$
* $\beta > 1$：增强垂直方向的相关性（beta默认值为10）
* $d_i$：插值点与观测点之间的距离
* $x_i, y_i, z_i$：观测点的经纬度和海拔高度
* $x_0, y_0, z_0$：插值点的经纬度和海拔高度
记住经纬度需要换算欧氏距离
得到距离，距离权重w_i = 1/(d_i)^p, 
$p$：插值权重的幂次，控制衰减速度，默认值为2

用xarray创建指定范围的0.01°经纬度分辨率的网格，再按照上述方法插值到网格上。
## 站点文件格式说明
站点文件为csv格式，需要读取的字段如下
| 中文         | 字段      | 英文           |
|--------------|-----------|----------------|
| 站号         | V01301    | code           |
| 站点名称     | VF01015_CN| name           |
| 所属地市     | V_CITY    | city           |
| 所属县       | V_COUNTY  | county         |
| 站点经度     | V06001    | lon            |
| 站点纬度     | V05001    | lat            |
| 站点海拔高度 | V07001    | altitude       |
| 站点能见度   | V20001    | vis            |
| 相对湿度     | V13003    | rh             |
用pandas读取相应字段，并把字段改为上述表格中的英文名。
## 高程数据说明
文件名格式如下 ASTGTM2_NyyExxx_dem.tif, 其中yy为纬度，xxx为经度，例如：
ASTGTM2_N25E114_dem.tif, 表示25°N, 114°E的高程数据。
