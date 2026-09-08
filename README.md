# constellation-project

Starlink 第一壳层（Shell 1）Walker-Delta 理想星座的 Python 复现。

模型参数为 `1584 / 72 / 39`，包括：

- 圆轨道与两体平均运动；
- 可选的一阶 J2 升交点赤经（RAAN）长期漂移；
- ECI 与简化 ECEF 坐标；
- 球形地球星下点经纬度；
- 3D 星座、星下点、纬度直方图及 Walker 相位图；
- 一个轨道周期的可选动画。

该模型用于复现理想化星座几何，不能代表某个真实 UTC 时刻的 Starlink 实际位置。真实位置计算应使用 TLE + SGP4。

## 运行环境

```powershell
pip install -r requirements.txt
```

当前电脑可以直接使用 Anaconda Python：

```powershell
D:\anaconda\python.exe starlink_shell1.py
```

## 常用命令

只计算并输出数值，不绘图：

```powershell
D:\anaconda\python.exe starlink_shell1.py --no-plots
```

仿真一天后的构型：

```powershell
D:\anaconda\python.exe starlink_shell1.py --time-seconds 86400
```

关闭 J2 漂移：

```powershell
D:\anaconda\python.exe starlink_shell1.py --no-j2
```

播放一个轨道周期的动画：

```powershell
D:\anaconda\python.exe starlink_shell1.py --animate
```

保存静态图片但不打开窗口：

```powershell
D:\anaconda\python.exe starlink_shell1.py --no-show --save-dir output
```
